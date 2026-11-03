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
# Q11b: does the binary rule follow absolute position or operand length? Test line "x+y=" with x, y ending in 1, digits 0/1 only.
rng = random.Random(9); out = {}
def ctx(base, r, a, c_): return "\n".join(render(*x, base) for x in sample_shots(r, (a, c_)))
ZERO, TWO = model.vocab["0"], model.vocab["2"]
def rule(S, hooks=None):
    X, q = ids(S); J = torch.arange(len(S), device=DEVICE); lg, _ = model.forward(X, hooks=hooks); p = lg[J, q].argmax(-1)
    return round(float((p == ZERO).float().mean()), 2), round(float((p == TWO).float().mean()), 2)
def shift(D):
    def f(t):
        i = torch.arange(t.shape[1], device=DEVICE); return t + PE[(i + D).clamp(0, 255)] - PE[i]
    return f
for lab, (lo, hi) in (("short test operands (2-3 binary digits)", (2, 7)), ("long test operands (7-10 binary digits)", (65, 1023))):
    Sb, So, So_long = [], [], []
    while len(Sb) < 400:
        a, c_ = rng.randint(lo, hi), rng.randint(lo, hi); sa, sb = to_base(a, 2), to_base(c_, 2)
        if sa[-1] != "1" or sb[-1] != "1": continue
        t = f"\n{sa}+{sb}="; cb, co = ctx(2, rng, a, c_), ctx(8, rng, a, c_)
        if len(cb) + len(t) > 250: continue
        Sb.append(cb + t); So.append(co + t)
    Lb, Lo = np.mean([len(x) for x in Sb]), np.mean([len(x) for x in So]); D = int(Lb - Lo)
    out[lab] = {"binary context: 0 / 2": rule(Sb), "octal context: 0 / 2": rule(So), f"octal context, tags +{D} (to binary positions): 0 / 2": rule(So, {"resid_pre0": shift(D)}), f"binary context, tags -{D} (to octal positions): 0 / 2": rule(Sb, {"resid_pre0": shift(-D)})}
    print("Q11b", lab, out[lab], flush=True)
res["Q11b"] = out
# Q13
r = random.Random(77); S, info = [], []
while len(S) < 1200:
    a, c_ = r.randint(16, 1023), r.randint(1, 1023); sa, sb = to_base(a, 2), to_base(c_, 2)
    if len(sb) >= len(sa): continue
    k = r.randint(len(sb), len(sa) - 1); ctx = "\n".join(render(*x, 2) for x in sample_shots(r, (a, c_))); st = len(ctx) + 1
    s = f"{ctx}\n{sa}+{sb}={to_answer(a + c_, 2)[:k]}"
    if len(s) > 250: continue
    S.append(s); info.append((st + len(sa), st + len(sa) + 1 + len(sb), k - len(sb), schoolbook(a, c_, 2)["o"][k], schoolbook(a, c_, 2)["c_in"][k], schoolbook(a, c_, 2)["a"][k]))
I = np.array(info); X, q = ids(S); J3 = torch.arange(len(S), device=DEVICE); n = X.shape[1]; idx = torch.arange(n, device=DEVICE)[None]
pl, eq = torch.as_tensor(I[:, 0], device=DEVICE), torch.as_tensor(I[:, 1], device=DEVICE); bm = (idx > pl[:, None]) & (idx < eq[:, None]); y = tok[I[:, 3]]
PR, HD, BT = [], {h: [] for h in range(4)}, {h: [] for h in range(4)}
for i in range(0, len(S), 200):
    sl = slice(i, i + 200); Jc = torch.arange(len(S[sl]), device=DEVICE)
    lg, c = model.forward(X[sl], capture={"pattern1"}); PR.append(lg[Jc, q[sl]].argmax(-1).cpu().numpy())
    for h in range(4):
        r_ = c["pattern1"][Jc, h, q[sl]]; HD[h].append((r_ * bm[sl]).sum(1).cpu().numpy()); BT[h].append(r_[Jc, pl[sl] + 1].cpu().numpy())
    del c, lg
pr = np.concatenate(PR); ok = pr == y; heads = {h: np.concatenate(HD[h]) for h in range(4)}; btop = {h: np.concatenate(BT[h]) for h in range(4)}
past = I[:, 2]
o13 = {"accuracy by digits past b's end": {int(p): round(float(ok[past == p].mean()), 2) for p in np.unique(past) if (past == p).sum() > 50}}
for h in range(4): o13[f"L1H{h} attention on b's digits (correct / wrong)"] = (round(float(heads[h][ok].mean()), 2), round(float(heads[h][~ok].mean()), 2)); o13[f"L1H{h} on b's leading digit (correct / wrong)"] = (round(float(btop[h][ok].mean()), 2), round(float(btop[h][~ok].mean()), 2))
wrong_pred = pr[~ok]; I_w = I[~ok]
o13["wrong answers equal (a_k + 1 + c) mod 2, i.e. as if b's digit were 1"] = round(float((wrong_pred == tok[(I_w[:, 5] + 1 + I_w[:, 4]) % 2]).mean()), 2)
for h in range(4):
    def blk(t, h=h):
        t = t.clone(); rows = bm[:, None, :] & (torch.arange(n, device=DEVICE)[None, :, None] == q[:, None, None])
        x = t[:, h].masked_fill(rows, 0.0); t[:, h] = x / x.sum(-1, keepdim=True).clamp_min(1e-9); return t
    p2 = []
    for i in range(0, len(S), 200):
        sl = slice(i, i + 200); Jc = torch.arange(len(S[sl]), device=DEVICE); qq = q[sl]; bb = bm[sl]
        def blk2(t, h=h, qq=qq, bb=bb):
            t = t.clone(); rows = bb[:, None, :] & (torch.arange(t.shape[2], device=DEVICE)[None, :, None] == qq[:, None, None])
            x = t[:, h].masked_fill(rows, 0.0); t[:, h] = x / x.sum(-1, keepdim=True).clamp_min(1e-9); return t
        lg2, _ = model.forward(X[sl], hooks={"pattern1": blk2}); p2.append(lg2[Jc, qq].argmax(-1).cpu().numpy())
    p2 = np.concatenate(p2); o13[f"block L1H{h} from b's digits: accuracy"] = round(float((p2 == y).mean()), 2); o13[f"   fixed errors"] = None
    o13[f"block L1H{h} from b's digits: accuracy"] = (round(float((p2 == y).mean()), 2), "errors fixed", round(float((p2 == y)[~ok].mean()), 2))
o13.pop("   fixed errors", None)
o13["baseline accuracy"] = round(float(ok.mean()), 2)
res["Q13"] = o13
for k, v in o13.items(): print("Q13", k, v, flush=True)
save_json("t16_binary", res); print("DONE")
