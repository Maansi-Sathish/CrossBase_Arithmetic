"""T13 (forward passes): addressing.
Q4. Later digits: what sets which operand digit a layer 1 head reads at digit k=1 (L1H3) and k=2 (L1H0)?
    Shift the position tag of one thing by d in {-2,-1,+1,+2}: the query only; '+' only; '=' only; all operand digits.
    Measure where the head's attention over a's and b's digits lands, as an offset (in digits) from the right digit.
Q5. Is L0H3's output at a's digits the recency code the digit-0 reader uses?
    (i) reverse the order of L0H3 outputs over a's digits; (ii) remove the line start from L0H3's attention at a's digits.
Q6. Giving b's digits a's position tags drops accuracy without moving L1H1. Which head breaks? Restore each L1 head's output.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {"Q4": {}, "Q5": {}, "Q6": {}}; tok = np.array([model.vocab[c] for c in DIGITS])
def make(b, n, seed, k, minlen):
    rng = random.Random(seed); S, info = [], []
    while len(S) < n:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) != len(sb) or len(sa) < minlen: continue
        ctx = "\n".join(render(*x, b) for x in sample_shots(rng, (a, c_))); st = len(ctx) + 1; L = len(sa); ans = to_answer(a + c_, b)
        s = f"{ctx}\n{sa}+{sb}={ans[:k]}"
        if len(s) > 250: continue
        S.append(s); info.append((st, st + L, st + 2 * L + 1, L, int(ans[k], 16) if k < len(ans) else -1))
    seqs = [[model.vocab[x] for x in s] for s in S]; n_ = max(map(len, seqs)); X = torch.full((len(S), n_), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    st, pl, eq, L, y = [torch.tensor([x[i] for x in info], device=DEVICE) for i in range(5)]
    return X.to(DEVICE), st, pl, eq, L, y
def pe_hook(shift):  # shift: [B, n] integer tag change per position
    def f(t):
        idx = torch.arange(t.shape[1], device=DEVICE)[None].expand(t.shape[0], -1)
        return t + PE[(idx + shift).clamp(0, PE.shape[0] - 1)] - PE[idx]
    return f
# Q4
for b in (2, 8, 10, 16):
    for k, head in ((1, 3), (2, 0)):
        X, st, pl, eq, L, y = make(b, 400, b * 7 + k, k, 3 if b > 2 else 4); B, n = X.shape; J = torch.arange(B, device=DEVICE); q = eq + k
        idx = torch.arange(n, device=DEVICE)[None]
        ta, tb = pl - 1 - k, eq - 1 - k  # target positions of a_k and b_k
        out = {}
        for what in ("query", "plus", "equals", "operands"):
            for d in (-2, -1, 0, 1, 2):
                if d == 0 and what != "query": continue
                if what == "query": m = idx == q[:, None]
                elif what == "plus": m = idx == pl[:, None]
                elif what == "equals": m = idx == eq[:, None]
                else: m = ((idx >= st[:, None]) & (idx < eq[:, None])) & (idx != pl[:, None])
                shift = torch.where(m, torch.full_like(idx.expand(B, -1), d), torch.zeros_like(idx.expand(B, -1)))
                lg, c = model.forward(X, hooks={"resid_pre0": pe_hook(shift)} if d else None, capture={"pattern1"}); row = c["pattern1"][J, head, q]  # [B, n]
                def centroid(lo, hi, target):
                    m_ = (idx >= lo[:, None]) & (idx < hi[:, None]); w = row * m_; pos = idx.float()
                    return ((w * (pos - target[:, None].float())).sum(1) / w.sum(1).clamp_min(1e-9)).mean().item(), w.sum(1).mean().item()
                ca, ma = centroid(st, pl, ta); cb, mb = centroid(pl + 1, eq, tb)
                acc = float((lg[J, q].argmax(-1).cpu().numpy() == tok[y.clamp_min(0).cpu().numpy()]).mean())
                out[f"{what} {d:+d}"] = {"a read offset": ca, "b read offset": cb, "mass on a": ma, "mass on b": mb, "correct": acc}
        res["Q4"][f"{b}/k{k}"] = out
        print(f"Q4 b{b} digit {k} L1H{head} read offset a/b (positive = toward the more significant digit... sign: +1 position = one digit LEFT? no: +1 = later position = less significant):")
        print("   ", {kk: (round(v['a read offset'], 2), round(v['b read offset'], 2), round(v['correct'], 2)) for kk, v in out.items()}, flush=True)
    save_json("t13_addressing", res)
# Q5 + Q6 at digit 0
for b in BASES:
    X, st, pl, eq, L, y = make(b, 500, b * 3, 0, 2); B, n = X.shape; J = torch.arange(B, device=DEVICE); idx = torch.arange(n, device=DEVICE)[None]
    am = (idx >= st[:, None]) & (idx < pl[:, None]); bm = (idx > pl[:, None]) & (idx < eq[:, None]); truth = y.cpu().numpy()
    def run(hooks, cap={"pattern1"}):
        lg, c = model.forward(X, hooks=hooks, capture=cap); r = c["pattern1"][J, 1, eq]
        return {"H1 on a0": float(r[J, pl - 1].mean()), "H1 on a first": float(r[J, st].mean()), "correct": float((lg[J, eq].argmax(-1).cpu().numpy() == tok[truth]).mean())}, c
    out = {}
    out["clean"], cc = run(None, {"pattern1", "z1"})
    # reversed L0H3 outputs over a's digits: position st+i gets the output of position pl-1-i
    src = torch.where(am, st[:, None] + pl[:, None] - 1 - idx, idx).clamp(0, n - 1)
    def rev(t):
        t = t.clone(); t[:, :, 3] = t[J[:, None], src, 3]; return t
    out["L0H3 outputs reversed over a's digits"], _ = run({"z0": rev})
    def no_nl(t):
        t = t.clone(); keym = idx == (st - 1)[:, None]; rows = am[:, :, None] & keym[:, None, :]
        h = t[:, 3].masked_fill(rows, 0.0); t[:, 3] = h / h.sum(-1, keepdim=True).clamp_min(1e-9); return t
    out["L0H3 cannot see the line start, at a's digits"], _ = run({"pattern0": no_nl})
    res["Q5"][b] = out; print(f"Q5 b{b}:", {kk: {a_: round(v_, 2) for a_, v_ in v.items()} for kk, v in out.items()}, flush=True)
    # Q6: b's digits get a's tags
    shift = torch.where(bm, -(pl + 1 - st)[:, None].expand(B, n), torch.zeros(B, n, dtype=torch.long, device=DEVICE))
    hk = pe_hook(shift); o6 = {}
    lg, c6 = model.forward(X, hooks={"resid_pre0": hk}, capture={"pattern1", "z1"})
    o6["tags swapped"] = float((lg[J, eq].argmax(-1).cpu().numpy() == tok[truth]).mean())
    o6["L1H0 on b0 clean / swapped"] = (float(cc["pattern1"][J, 0, eq][J, eq - 1].mean()), float(c6["pattern1"][J, 0, eq][J, eq - 1].mean()))
    o6["L1H0 on b's first digit clean / swapped"] = (float(cc["pattern1"][J, 0, eq][J, pl + 1].mean()), float(c6["pattern1"][J, 0, eq][J, pl + 1].mean()))
    for h in range(4):
        def restore(t, h=h):
            t = t.clone(); t[J, eq, h] = cc["z1"][J, eq, h]; return t
        lg, _ = model.forward(X, hooks={"resid_pre0": hk, "z1": restore}); o6[f"swapped + restore L1H{h}"] = float((lg[J, eq].argmax(-1).cpu().numpy() == tok[truth]).mean())
    res["Q6"][b] = o6; print(f"Q6 b{b}:", {kk: (tuple(round(x, 2) for x in v) if isinstance(v, tuple) else round(v, 2)) for kk, v in o6.items()}, flush=True)
save_json("t13_addressing", res); print("DONE")
