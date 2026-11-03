"""T15c: how does an unseen leading digit break the middle digits? Octal digit 2 and hex digit 1, operands of the longest training
length with leading digit unseen at that position. Counterpart prompt: same everything but leading digits set to 1 (a seen value).
Teacher forcing uses the TRUE answer prefix in both (the counterpart's own sum differs only in the top digit, not digit k).
  (i) attention of the reading heads to the target column vs the leading column
  (ii) restore layer 0 output at the leading digit positions from the counterpart
  (iii) restore each L1 head's output at the query from the counterpart; (iv) restore L1 MLP input pieces via resid_mid1."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, schoolbook, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); tok = np.array([model.vocab[c] for c in DIGITS]); res = {}
def ids(S):
    seqs = [[model.vocab[x] for x in s] for s in S]; n = max(map(len, seqs)); X = torch.full((len(S), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    return X.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
for b, k, readers in ((8, 2, (0, 1, 3)), (16, 1, (1, 3))):
    L = len(to_base(999, b)); top = int(to_base(999, b)[0], 16); rng = random.Random(b * 41); C, D, y, pos = [], [], [], []
    while len(C) < 500:
        da = [rng.randrange(b) for _ in range(L - 1)]; db = [rng.randrange(b) for _ in range(L - 1)]
        ta, tb = rng.randint(top + 1, b - 1), rng.randint(1, b - 1)
        a = ta * b**(L - 1) + sum(d * b**i for i, d in enumerate(da)); c_ = tb * b**(L - 1) + sum(d * b**i for i, d in enumerate(db))
        a1 = 1 * b**(L - 1) + a % b**(L - 1); c1 = 1 * b**(L - 1) + c_ % b**(L - 1)
        ans = to_answer(a + c_, b)[:k]; o = schoolbook(a, c_, b)["o"][k]; assert schoolbook(a1, c1, b)["o"][k] == o
        ctx = "\n".join(render(*x, b) for x in sample_shots(rng, (a, c_))); st = len(ctx) + 1
        C.append(f"{ctx}\n{to_base(a, b)}+{to_base(c_, b)}={ans}"); D.append(f"{ctx}\n{to_base(a1, b)}+{to_base(c1, b)}={ans}"); y.append(o); pos.append((st, st + L + 1, st + 2 * L + 1))
    Xc, q = ids(C); Xd, _ = ids(D); J = torch.arange(len(C), device=DEVICE); y = tok[np.array(y)]; st, bt, eq = [torch.tensor([p[i] for p in pos], device=DEVICE) for i in range(3)]
    ta_, tb_ = st + L - 1 - k, eq - 1 - k  # target columns
    names = {"resid_pre1", "z1", "pattern1", "resid_mid1"}
    lc, cc = model.forward(Xc, capture=names); ld, cd = model.forward(Xd, capture=names)
    acc = lambda lg: float((lg[J, q].argmax(-1).cpu().numpy() == y).mean())
    out = {"unseen leading digit": acc(lc), "counterpart (leading digit 1)": acc(ld)}
    for h in readers:
        pc, pd = cc["pattern1"][J, h, q], cd["pattern1"][J, h, q]
        out[f"L1H{h} attention target a/b column (unseen | counterpart)"] = (round(float(pc[J, ta_].mean() + pc[J, tb_].mean()), 2), round(float(pd[J, ta_].mean() + pd[J, tb_].mean()), 2))
        out[f"L1H{h} attention leading digits (unseen | counterpart)"] = (round(float(pc[J, st].mean() + pc[J, bt].mean()), 2), round(float(pd[J, st].mean() + pd[J, bt].mean()), 2))
    def lead(t):
        t = t.clone(); t[J, st] = cd["resid_pre1"][J, st]; t[J, bt] = cd["resid_pre1"][J, bt]; return t
    def lead_only_not_token(t):  # restore layer 0 output at leading digits but keep the leading digit's own token embedding
        t = t.clone()
        for p in (st, bt): t[J, p] = cd["resid_pre1"][J, p] - model.WE[Xd[J, p]] + model.WE[Xc[J, p]]
        return t
    lg, _ = model.forward(Xc, hooks={"resid_pre1": lead}); out["restore layer 0 output at leading digits"] = acc(lg)
    lg, _ = model.forward(Xc, hooks={"resid_pre1": lead_only_not_token}); out["same, but keep the unseen token embedding"] = acc(lg)
    for h in range(4):
        def fn(t, h=h):
            t = t.clone(); t[J, q, h] = cd["z1"][J, q, h]; return t
        lg, _ = model.forward(Xc, hooks={"z1": fn}); out[f"restore L1H{h} at query"] = acc(lg)
    def allh(t):
        t = t.clone(); t[J, q] = cd["z1"][J, q]; return t
    lg, _ = model.forward(Xc, hooks={"z1": allh}); out["restore all L1 heads at query"] = acc(lg)
    res[f"{b}/k{k}"] = out; print(f"b{b} digit {k}:", out, flush=True)
save_json("t15c_leading_digit", res); print("DONE")
