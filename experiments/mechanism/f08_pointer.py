"""F08 (forward passes): quantitative pointer test, and the digit-0 recency test.

Pointer formula: target_a = pos(+) - 1 - k and target_b = pos(=) - 1 - k, with k = pos(query) - pos(=).
For a position-embedding shift d the formula predicts these moves of the attended digit:
  plus: da=+d, db=0 | eq: da=+d, db=+2d | query: da=-d, db=-d | a_digits: da=-d, db=0 | b_digits: da=0, db=-d | all: 0, 0
Measured move: change in attention-weighted mean position over the operand's digits (offsets -3..+3).

Digit 0 recency test: swap position embeddings of the last two digits of a (or b). If the head reads whichever digit
carries the latest position tag, attention moves to the physically earlier digit, and the output uses it.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {"pointer": {}, "recency": {}}
tok = np.array([model.vocab[c] for c in DIGITS])
CASES = {(2, 1): (1, 0), (2, 2): (1, 0), (2, 3): (1, 0), (8, 1): (3, 3), (8, 2): (0, 0), (10, 1): (3, 3), (10, 2): (0, 0), (16, 1): (3, 3), (16, 2): (0, 0)}
PRED = {"plus": (1, 0), "eq": (1, 2), "query": (-1, -1), "a_digits": (-1, 0), "b_digits": (0, -1), "all": (0, 0)}


def build(base, k, n, seed):
    rng = random.Random(seed); out = []
    while len(out) < n:
        a, b_ = rng.randint(0, 999), rng.randint(0, 999)
        sa, sb = to_base(a, base), to_base(b_, base)
        if len(sa) <= k or len(sb) <= k or len(to_answer(a + b_, base)) <= k:
            continue
        ctx = "\n".join(render(*s, base) for s in sample_shots(rng, (a, b_)))
        s = f"{ctx}\n{sa}+{sb}={to_answer(a + b_, base)[:k]}"
        c = len(ctx) + 1
        out.append(dict(s=s, a=a, b=b_, start=c, plus=c + len(sa), eq=c + len(sa) + 1 + len(sb), q=len(s) - 1))
    return out


def tensors(P):
    seqs = [[model.vocab[c] for c in p["s"]] for p in P]
    L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
    g = lambda n: torch.tensor([p[n] for p in P], device=DEVICE)
    return x.to(DEVICE), g("start"), g("plus"), g("eq"), g("q")


def mean_pos(pat, head, q, lo, hi, J):
    """Attention-weighted mean position over digits in [lo, hi)."""
    row = pat[J, head, q]
    idx = torch.arange(row.shape[1], device=DEVICE)[None]
    mask = (idx >= lo[:, None]) & (idx < hi[:, None])
    m = (row * mask).sum(1).clamp_min(1e-6)
    return ((row * mask * idx).sum(1) / m)


def run_shift(P, k, tgt, d, ra, rb):
    da, db, n = 0.0, 0.0, 0
    ok = 0
    for i in range(0, len(P), 128):
        ch = P[i:i + 128]
        x, st, pl, eq, q = tensors(ch)
        J = torch.arange(len(ch), device=DEVICE)
        idx = torch.arange(x.shape[1], device=DEVICE)[None]
        mask = {"plus": idx == pl[:, None], "eq": idx == eq[:, None], "query": idx == q[:, None], "a_digits": (idx >= st[:, None]) & (idx < pl[:, None]), "b_digits": (idx > pl[:, None]) & (idx < eq[:, None]), "all": (idx >= st[:, None]) & (idx <= q[:, None])}[tgt]
        pos = idx.expand(x.shape)
        delta = (PE[(pos + d).clamp(0, 255)] - PE[pos]) * mask[..., None]
        _, c0 = model.forward(x, capture={"pattern1"})
        lg, c1 = model.forward(x, hooks={"resid_pre0": lambda t, delta=delta: t + delta}, capture={"pattern1"})
        da += (mean_pos(c1["pattern1"], ra, q, st, pl, J) - mean_pos(c0["pattern1"], ra, q, st, pl, J)).sum().item()
        db += (mean_pos(c1["pattern1"], rb, q, pl + 1, eq, J) - mean_pos(c0["pattern1"], rb, q, pl + 1, eq, J)).sum().item()
        n += len(ch)
    return da / n, db / n


X, Y, rows = [], [], []
for (base, k), (ra, rb) in CASES.items():
    P = build(base, k, 300, base * 7 + k)
    row = {}
    for tgt, (pa, pb) in PRED.items():
        for d in (-1, 1):
            ma, mb = run_shift(P, k, tgt, d, ra, rb)
            row[f"{tgt}/{d:+d}"] = {"measured_a": ma, "measured_b": mb, "pred_a": pa * d, "pred_b": pb * d}
            X += [pa * d, pb * d]; Y += [ma, mb]; rows += [(base, k), (base, k)]
    res["pointer"][f"{base}/k{k}"] = row
    print(f"b{base} k{k}:", {t: (round(row[f'{t}/+1']['measured_a'], 2), round(row[f'{t}/+1']['measured_b'], 2), PRED[t]) for t in PRED}, flush=True)
    xs = np.array([x for x, r in zip(X, rows) if r == (base, k)]); ys = np.array([y for y, r in zip(Y, rows) if r == (base, k)])
    r = float(np.corrcoef(xs, ys)[0, 1]); slope = float((xs * ys).sum() / (xs * xs).sum())
    res["pointer"][f"{base}/k{k}"]["fit"] = {"corr": r, "slope": slope}
    print(f"   predicted vs measured: r={r:.2f}, slope={slope:.2f}", flush=True)
    save_json("f08_pointer", res)

# ---- recency at digit 0
for base in (2, 8, 10, 16):
    P = []
    rng = random.Random(base + 99)
    while len(P) < 800:
        a, b_ = rng.randint(0, 999), rng.randint(0, 999)
        sa, sb = to_base(a, base), to_base(b_, base)
        if len(sa) < 2 or len(sb) < 2: continue
        ctx = "\n".join(render(*s, base) for s in sample_shots(rng, (a, b_)))
        c = len(ctx) + 1
        P.append(dict(s=f"{ctx}\n{sa}+{sb}=", a=a, b=b_, start=c, plus=c + len(sa), eq=c + len(sa) + 1 + len(sb), q=c + len(sa) + 1 + len(sb)))
    A = np.array([p["a"] for p in P]); B = np.array([p["b"] for p in P])
    true = (A % base + B % base) % base
    a_swap = ((A // base) % base + B % base) % base
    b_swap = (A % base + (B // base) % base) % base
    out = {}
    for which in ("clean", "a", "b", "both_shift_a"):
        pr, ml, mo = [], 0., 0.
        for i in range(0, len(P), 256):
            ch = P[i:i + 256]; x, st, pl, eq, q = tensors(ch); J = torch.arange(len(ch), device=DEVICE)
            pos = torch.arange(x.shape[1], device=DEVICE)[None].expand(x.shape)
            pos2 = pos.clone()
            if which == "a":
                pos2[J, pl - 1] = pl - 2; pos2[J, pl - 2] = pl - 1
            if which == "b":
                pos2[J, eq - 1] = eq - 2; pos2[J, eq - 2] = eq - 1
            delta = PE[pos2] - PE[pos]
            lg, c = model.forward(x, hooks={"resid_pre0": lambda t, delta=delta: t + delta}, capture={"pattern1"})
            pr.append(lg[J, q].argmax(-1).cpu().numpy())
            H, tp = (1, pl) if which != "b" else (0, eq)
            ml += c["pattern1"][J, H, q, tp - 1].sum().item(); mo += c["pattern1"][J, H, q, tp - 2].sum().item()
        pr = np.concatenate(pr)
        diff_a, diff_b = a_swap != true, b_swap != true
        out[which] = {"attn_on_last_digit": ml / len(P), "attn_on_second_last": mo / len(P), "output_true": float((pr == tok[true]).mean()), "output_uses_swapped_a": float((pr[diff_a] == tok[a_swap[diff_a]]).mean()), "output_uses_swapped_b": float((pr[diff_b] == tok[b_swap[diff_b]]).mean())}
    res["recency"][base] = out
    print("RECENCY", base, {k: {a: round(v, 2) for a, v in r.items()} for k, r in out.items() if k != "both_shift_a"}, flush=True)
save_json("f08_pointer", res)
print("DONE")
