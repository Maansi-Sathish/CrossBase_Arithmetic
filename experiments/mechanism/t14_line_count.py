"""T14 (forward passes). Q7/Q8: how does the model know it is on the sixth line, so it may stop?
At the last answer digit (teacher forced), compare P(stop) = EOS chosen, vs newline chosen.
A. Number of shots 1..8.
B. Absolute position vs line count: shift every position tag by +D (fewer shots look later) or -D (5 shots look earlier).
   Also 5 shots of short examples vs long examples (same line count, different absolute position).
C. Extra blank lines: 4 shots + one empty line (5 newlines before the test line, 4 examples). 5 shots with one '\n\n'.
D. Patch from 5-shot into 3-shot (same test line and answer) at the final position: each L0/L1 head output, L0 MLP, L1 MLP.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {"A": {}, "B": {}, "C": {}, "D": {}}; NL = model.vocab["\n"]
def build(strs):
    seqs = [[model.vocab[x] for x in s] for s in strs]; n = max(map(len, seqs)); X = torch.full((len(seqs), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    return X.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
def stop_rate(strs, hooks=None):
    strs = [x for x in strs if len(x) < 250]
    if not strs: return float('nan'), float('nan')
    X, q = build(strs); lg, _ = model.forward(X, hooks=hooks); r = lg[torch.arange(len(strs), device=DEVICE), q].argmax(-1)
    return float((r == model.eos).float().mean()), float((r == NL).float().mean())
def problems(b, n, seed, k=5, small=False, big=False):
    rng = random.Random(seed); out = []
    for _ in range(n):
        a, c_ = rng.randint(0, 999), rng.randint(0, 999)
        shots = sample_shots(rng, (a, c_), k=k, hi=((b - 1 if b > 2 else 3) if small else 999))
        if big: shots = [(rng.randint(b**2, 999), rng.randint(b**2, 999)) for _ in range(k)]
        out.append((shots, a, c_))
    return out
def text(b, shots, a, c_, sep="\n", extra=""):
    return sep.join(render(*s, b) for s in shots) + extra + f"\n{to_base(a, b)}+{to_base(c_, b)}={to_answer(a + c_, b)}"
def shift_all(D):
    def f(t):
        n = t.shape[1]; idx = torch.arange(n, device=DEVICE); return t + PE[(idx + D).clamp(0, PE.shape[0] - 1)] - PE[idx]
    return f
for b in (2, 8, 10, 16):
    rows = {}
    for k in range(1, 9):
        S = [text(b, *p) for p in problems(b, 300, b * 100 + k, k=k)]; S = [s for s in S if len(s) < 255]
        rows[k] = stop_rate(S); rows[k] = {"stop": rows[k][0], "newline": rows[k][1], "mean length": float(np.mean([len(s) for s in S]))}
    res["A"][b] = rows; print(f"A b{b} stop rate by shots:", {k: (round(v['stop'], 2), round(v['newline'], 2)) for k, v in rows.items()}, flush=True)
    out = {}
    P3 = problems(b, 300, b * 7, k=3); P5 = problems(b, 300, b * 9, k=5)
    S3 = [text(b, *p) for p in P3]; S5 = [text(b, *p) for p in P5]
    len3, len5 = np.mean([len(s) for s in S3]), np.mean([len(s) for s in S5]); D = int(round(len5 - len3))
    out["3 shots"] = stop_rate(S3)[0]; out[f"3 shots, tags shifted +{D} (to 5-shot positions)"] = stop_rate(S3, {"resid_pre0": shift_all(D)})[0]
    out["5 shots"] = stop_rate(S5)[0]; out[f"5 shots, tags shifted -{D} (to 3-shot positions)"] = stop_rate(S5, {"resid_pre0": shift_all(-D)})[0]
    Ss = [text(b, *p) for p in problems(b, 300, b * 11, k=5, small=True)]; out["5 short shots (one-digit operands)"] = stop_rate(Ss)[0]; out["mean length short"] = float(np.mean([len(s) for s in Ss]))
    Sb = [s for s in (text(b, *p) for p in problems(b, 300, b * 13, k=3, big=True)) if len(s) < 255]; out["3 long shots (three-digit operands)"] = stop_rate(Sb)[0]; out["mean length long 3-shot"] = float(np.mean([len(s) for s in Sb])); out["mean length 5-shot"] = float(len5)
    res["B"][b] = out; print(f"B b{b}:", {k: round(v, 2) for k, v in out.items()}, flush=True)
    P4 = problems(b, 300, b * 15, k=4)
    c = {"4 shots": stop_rate([text(b, *p) for p in P4])[0], "4 shots + one blank line": stop_rate([text(b, *p, extra="\n") for p in P4])[0],
         "4 shots, one separator doubled": stop_rate([text(b, p[0][:2], p[1], p[2]).split("\n")[0] + "\n" + render(*p[0][1], b) + "\n\n" + "\n".join(render(*s, b) for s in p[0][2:]) + f"\n{to_base(p[1], b)}+{to_base(p[2], b)}={to_answer(p[1] + p[2], b)}" for p in P4])[0],
         "5 shots": stop_rate([text(b, *p) for p in P5])[0], "5 shots + one blank line": stop_rate([text(b, *p, extra="\n") for p in P5])[0]}
    res["C"][b] = c; print(f"C b{b}:", {k: round(v, 2) for k, v in c.items()}, flush=True)
    # D: same test line, 3 vs 5 shots; patch at final position
    rng = random.Random(b * 17); S3, S5 = [], []
    while len(S3) < 300:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sh = sample_shots(rng, (a, c_), k=5)
        s5 = text(b, sh, a, c_); s3 = text(b, sh[:3], a, c_)
        if len(s5) < 255: S5.append(s5); S3.append(s3)
    X3, q3 = build(S3); X5, q5 = build(S5); J = torch.arange(len(S3), device=DEVICE)
    names = {"z0", "z1", "mlp_post0", "mlp_post1", "resid_pre1"}
    l3, c3 = model.forward(X3, capture=names); l5, c5 = model.forward(X5, capture=names)
    base3 = float((l3[J, q3].argmax(-1) == model.eos).float().mean()); base5 = float((l5[J, q5].argmax(-1) == model.eos).float().mean())
    d = {"3-shot stop": base3, "5-shot stop": base5}
    comps = [(f"L0H{h}", "z0", h) for h in range(4)] + [(f"L1H{h}", "z1", h) for h in range(4)] + [("L0 MLP", "mlp_post0", None), ("L1 MLP", "mlp_post1", None), ("layer 0 output (resid_pre1)", "resid_pre1", None)]
    for nm, hk, h in comps:
        def fn(t, s=c5[hk], h=h):
            t = t.clone()
            if h is None: t[J, q3] = s[J, q5]
            else: t[J, q3, h] = s[J, q5, h]
            return t
        lg, _ = model.forward(X3, hooks={hk: fn}); d[f"3-shot gets 5-shot {nm}"] = float((lg[J, q3].argmax(-1) == model.eos).float().mean())
        def fn2(t, s=c3[hk], h=h):
            t = t.clone()
            if h is None: t[J, q5] = s[J, q3]
            else: t[J, q5, h] = s[J, q3, h]
            return t
        lg, _ = model.forward(X5, hooks={hk: fn2}); d[f"5-shot gets 3-shot {nm}"] = float((lg[J, q5].argmax(-1) == model.eos).float().mean())
    # attention of each head at the final position to newline tokens
    for L in (0, 1):
        _, c = model.forward(X5, capture={f"pattern{L}"}); pat = c[f"pattern{L}"][J, :, q5]
        nlmask = (X5 == NL)[:, None, :]
        d[f"L{L} attention to newlines (5-shot) by head"] = [round(x, 3) for x in (pat * nlmask).sum(-1).mean(0).tolist()]
        _, c = model.forward(X3, capture={f"pattern{L}"}); pat = c[f"pattern{L}"][J, :, q3]; nlmask = (X3 == NL)[:, None, :]
        d[f"L{L} attention to newlines (3-shot) by head"] = [round(x, 3) for x in (pat * nlmask).sum(-1).mean(0).tolist()]
    res["D"][b] = d; print(f"D b{b}:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in d.items()}, flush=True)
    save_json("t14_line_count", res)
print("DONE")
