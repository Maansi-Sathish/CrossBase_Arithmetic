"""T16 (forward passes): binary.
Q11. How is base 2 detected? Test lines with digits 0/1 only and a0=b0=1 (binary writes 0, octal/decimal write 2).
     Same test line under a binary context vs an octal context. Patch each L0 head / layer 0 output on the test line between them.
     Dose: put n copies of a digit 2-7 into the binary context.
Q12. Is binary's reliance on position tags (not L0H3 recency) due to long numbers? Digit-0 reader tests on short (<=3) vs long (>=7)
     binary operands: reverse L0H3 outputs over a's digits; give all of a's digits the same tag.
Q13. Unequal lengths: when b is shorter, accuracy at digits k >= len(b), L1H0's attention to b's digits there, and the effect of
     blocking that attention.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, schoolbook, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; tok = np.array([model.vocab[c] for c in DIGITS]); res = {"Q11": {}, "Q12": {}, "Q13": {}}
def ids(S):
    seqs = [[model.vocab[x] for x in s] for s in S]; n = max(map(len, seqs)); X = torch.full((len(S), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    return X.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
# Q11
rng = random.Random(5); Cb, Co, tl = [], [], []
while len(Cb) < 600:
    a, c_ = rng.randint(1, 63), rng.randint(1, 63); sa, sb = to_base(a, 2), to_base(c_, 2)
    if sa[-1] != "1" or sb[-1] != "1": continue
    test = f"{sa}+{sb}="
    cb = "\n".join(render(*x, 2) for x in sample_shots(rng, (a, c_))); co = "\n".join(render(*x, 8) for x in sample_shots(rng, (a, c_)))
    if len(cb) + len(test) > 250: continue
    # pad shorter context on the left with nothing; align the test line by requiring both contexts' lengths differ -> patch by test-line offset
    Cb.append(cb + "\n" + test); Co.append(co + "\n" + test); tl.append(len(test))
Xb, qb = ids(Cb); Xo, qo = ids(Co); J = torch.arange(len(Cb), device=DEVICE); T = torch.tensor(tl, device=DEVICE)
ZERO, TWO = model.vocab["0"], model.vocab["2"]
def rule(lg, q):
    p = lg[J, q].argmax(-1); return float((p == ZERO).float().mean()), float((p == TWO).float().mean())
names = {"z0", "resid_pre1", "mlp_post0", "pattern0"}
lb, cb_ = model.forward(Xb, capture=names); lo, co_ = model.forward(Xo, capture=names)
out = {"binary context: writes 0 / 2": rule(lb, qb), "octal context: writes 0 / 2": rule(lo, qo)}
off = torch.arange(int(T.max()), device=DEVICE)
def patch(dst_q, src_q, src, name, h):
    def f(t):
        t = t.clone()
        for o in range(int(T.max())):
            m = o < T; jd = J[m]; pd = (dst_q - o)[m]; ps = (src_q - o)[m]
            if h is None: t[jd, pd] = src[name][jd, ps]
            else: t[jd, pd, h] = src[name][jd, ps, h]
        return t
    return f
for nm, name, h in [(f"L0H{h}", "z0", h) for h in range(4)] + [("L0 MLP", "mlp_post0", None), ("layer 0 output (incl. embeddings)", "resid_pre1", None)]:
    lg, _ = model.forward(Xb, hooks={name: patch(qb, qo, co_, name, h)}); out[f"binary gets octal {nm} on test line: 0 / 2"] = rule(lg, qb)
    lg, _ = model.forward(Xo, hooks={name: patch(qo, qb, cb_, name, h)}); out[f"octal gets binary {nm} on test line: 0 / 2"] = rule(lg, qo)
pb = co_["pattern0"]  # unused placeholder
for which, c, X, q in (("binary", cb_, Xb, qb), ("octal", co_, Xo, qo)):
    pat = c["pattern0"][J, 1, q]; idx = torch.arange(X.shape[1], device=DEVICE)[None]; on_line = idx > (q - T)[:, None]
    out[f"L0H1 at '=' attention on test line ({which} context)"] = float((pat * on_line).sum(1).mean())
dose = {}
for d in "234567":
    for n in (1, 2, 4):
        S = []
        r2 = random.Random(ord(d) * 10 + n)
        for s in Cb:
            ctx, test = s.rsplit("\n", 1); cl = list(ctx); P = [i for i, ch in enumerate(cl) if ch in "01"]
            for i in r2.sample(P, n): cl[i] = d
            S.append("".join(cl) + "\n" + test)
        X, q = ids(S); lg, _ = model.forward(X); dose[f"{n} x '{d}'"] = rule(lg, q)
out["dose: n context digits replaced by d -> writes 0 / 2"] = dose
res["Q11"] = out
for k, v in out.items(): print("Q11", k, v if not isinstance(v, dict) else {a: tuple(round(x, 2) for x in b) for a, b in v.items()}, flush=True)
# Q12
def make(n, seed, lo, hi):
    r = random.Random(seed); S, info = [], []
    while len(S) < n:
        a, c_ = r.randint(0, 1023), r.randint(0, 1023); sa, sb = to_base(a, 2), to_base(c_, 2)
        if not (lo <= len(sa) <= hi and lo <= len(sb) <= hi): continue
        ctx = "\n".join(render(*x, 2) for x in sample_shots(r, (a, c_))); st = len(ctx) + 1
        s = f"{ctx}\n{sa}+{sb}="
        if len(s) > 250: continue
        S.append(s); info.append((st, st + len(sa), (a + c_) % 2))
    X, q = ids(S); st, pl, y = [torch.tensor([x[i] for x in info], device=DEVICE) for i in range(3)]
    return X, q, st, pl, y
for lab, lo, hi in (("short (2-3 digits)", 2, 3), ("long (7-10 digits)", 7, 10)):
    X, q, st, pl, y = make(500, len(lab), lo, hi); J2 = torch.arange(len(X), device=DEVICE); n = X.shape[1]; idx = torch.arange(n, device=DEVICE)[None]
    am = (idx >= st[:, None]) & (idx < pl[:, None])
    def meas(hooks):
        lg, c = model.forward(X, hooks=hooks, capture={"pattern1"}); r = c["pattern1"][J2, 1, q]
        return {"H1 on a0": round(float(r[J2, pl - 1].mean()), 2), "H1 on a first": round(float(r[J2, st].mean()), 2), "correct": round(float((lg[J2, q].argmax(-1) == torch.as_tensor(tok, device=DEVICE)[y]).float().mean()), 2)}
    src = torch.where(am, st[:, None] + pl[:, None] - 1 - idx, idx).clamp(0, n - 1)
    def rev(t):
        t = t.clone(); t[:, :, 3] = t[J2[:, None], src, 3]; return t
    flat = torch.where(am, (pl - 1)[:, None] - idx, torch.zeros_like(idx).expand(len(X), -1))
    def flat_h(t):
        i_ = torch.arange(n, device=DEVICE)[None].expand(len(X), -1); return t + PE[(i_ + flat).clamp(0, 255)] - PE[i_]
    o = {"clean": meas(None), "L0H3 outputs reversed over a": meas({"z0": rev}), "all of a's digits get a0's tag": meas({"resid_pre0": flat_h})}
    _, c = model.forward(X, capture={"pattern0"}); p3 = c["pattern0"][:, 3]
    nl = []
    for l in range(hi):
        m = am & (idx - st[:, None] == l)
        if m.sum() > 20: nl.append(round(float(p3[m][:, :].gather(1, (st - 1)[m.nonzero()[:, 0]][:, None]).mean()), 2))
    o["L0H3 attention to line start by digit index"] = nl
    res["Q12"][lab] = o; print("Q12", lab, o, flush=True)
# Q13
r = random.Random(77); S, info = [], []
while len(S) < 2500:
    a, c_ = r.randint(16, 1023), r.randint(1, 1023); sa, sb = to_base(a, 2), to_base(c_, 2)
    if len(sb) >= len(sa): continue
    k = r.randint(len(sb), len(sa) - 1); ctx = "\n".join(render(*x, 2) for x in sample_shots(r, (a, c_))); st = len(ctx) + 1
    s = f"{ctx}\n{sa}+{sb}={to_answer(a + c_, 2)[:k]}"
    if len(s) > 250: continue
    S.append(s); info.append((st + len(sa), st + len(sa) + 1 + len(sb), k - len(sb), schoolbook(a, c_, 2)["o"][k], schoolbook(a, c_, 2)["c_in"][k], schoolbook(a, c_, 2)["a"][k]))
I = np.array(info); X, q = ids(S); J3 = torch.arange(len(S), device=DEVICE); n = X.shape[1]; idx = torch.arange(n, device=DEVICE)[None]
pl, eq = torch.as_tensor(I[:, 0], device=DEVICE), torch.as_tensor(I[:, 1], device=DEVICE); bm = (idx > pl[:, None]) & (idx < eq[:, None]); y = tok[I[:, 3]]
lg, c = model.forward(X, capture={"pattern1"}); pr = lg[J3, q].argmax(-1).cpu().numpy(); ok = pr == y
heads = {h: (c["pattern1"][J3, h, q] * bm).sum(1).cpu().numpy() for h in range(4)}
btop = {h: c["pattern1"][J3, h, q][J3, pl + 1].cpu().numpy() for h in range(4)}
past = I[:, 2]
o13 = {"accuracy by digits past b's end": {int(p): round(float(ok[past == p].mean()), 2) for p in np.unique(past) if (past == p).sum() > 50}}
for h in range(4): o13[f"L1H{h} attention on b's digits (correct / wrong)"] = (round(float(heads[h][ok].mean()), 2), round(float(heads[h][~ok].mean()), 2)); o13[f"L1H{h} on b's leading digit (correct / wrong)"] = (round(float(btop[h][ok].mean()), 2), round(float(btop[h][~ok].mean()), 2))
wrong_pred = pr[~ok]; I_w = I[~ok]
o13["wrong answers equal (a_k + 1 + c) mod 2, i.e. as if b's digit were 1"] = round(float((wrong_pred == tok[(I_w[:, 5] + 1 + I_w[:, 4]) % 2]).mean()), 2)
for h in range(4):
    def blk(t, h=h):
        t = t.clone(); rows = bm[:, None, :] & (torch.arange(n, device=DEVICE)[None, :, None] == q[:, None, None])
        x = t[:, h].masked_fill(rows, 0.0); t[:, h] = x / x.sum(-1, keepdim=True).clamp_min(1e-9); return t
    lg2, _ = model.forward(X, hooks={"pattern1": blk}); p2 = lg2[J3, q].argmax(-1).cpu().numpy(); o13[f"block L1H{h} from b's digits: accuracy"] = round(float((p2 == y).mean()), 2); o13[f"   fixed errors"] = None
    o13[f"block L1H{h} from b's digits: accuracy"] = (round(float((p2 == y).mean()), 2), "errors fixed", round(float((p2 == y)[~ok].mean()), 2))
o13.pop("   fixed errors", None)
o13["baseline accuracy"] = round(float(ok.mean()), 2)
res["Q13"] = o13
for k, v in o13.items(): print("Q13", k, v, flush=True)
save_json("t16_binary", res); print("DONE")
