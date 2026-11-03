"""T16d: which wrong carry rule produces binary's carry errors? For each problem and digit k (teacher forced), the model's implied
carry c_hat = 1 if its digit equals (a_k+b_k+1) mod 2. Candidate rules for the carry:
  true; a_{k-1} AND b_{k-1} (ignores chains); written w_{k-1} based: a_{k-1}+b_{k-1}-w_{k-1} >= 1 (= true);
  read of the wrong column: AND of column k-2; OR of column k-1; majority of (a_{k-1}, b_{k-1}, c_{k-1} of column k-2 AND)
Report each rule's agreement with c_hat on errors, and the prompts' features at errors (operand lengths, k)."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_problems, save_json, schoolbook, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); P = [p for p in sample_problems(6000, 2, seed=93) if len(p.prompt) + 14 <= 256]; tok = np.array([model.vocab[c] for c in DIGITS])
S, info = [], []
for p in P:
    sb = schoolbook(p.a, p.b, 2); ans = p.answer
    for k in range(1, min(len(ans), 6)):
        S.append(p.prompt + ans[:k]); info.append((k, sb["a"][k], sb["b"][k], sb["c_in"][k], sb["a"][k - 1], sb["b"][k - 1], sb["a"][k - 2] if k >= 2 else 0, sb["b"][k - 2] if k >= 2 else 0, sb["c_in"][k - 1], len(to_base(p.a, 2)), len(to_base(p.b, 2))))
pr = []
for i in range(0, len(S), 400):
    seqs = [[model.vocab[x] for x in s] for s in S[i:i + 400]]; n = max(map(len, seqs)); X = torch.full((len(seqs), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); lg, _ = model.forward(X.to(DEVICE)); pr.append(lg[torch.arange(len(seqs), device=DEVICE), q].argmax(-1).cpu().numpy())
pr = np.concatenate(pr); I = np.array(info); k, ak, bk, c, a1, b1, a2, b2, c1, la, lb = I.T
d0 = tok[(ak + bk) % 2]; d1 = tok[(ak + bk + 1) % 2]; valid = (pr == d0) | (pr == d1); chat = (pr == d1).astype(int); err = valid & (chat != c)
rules = {"true carry": c, "AND of column k-1 (ignores incoming chain)": a1 & b1, "OR of column k-1": a1 | b1, "AND of column k-2": a2 & b2, "carry-in of column k-1 (one step stale)": c1, "a_{k-1} only": a1, "b_{k-1} only": b1}
res = {"n": int(valid.sum()), "errors": int(err.sum()), "error rate by k": {int(x): round(float(err[k == x].mean()), 4) for x in np.unique(k)}}
res["agreement with model's carry on errors"] = {nm: round(float((r[err] == chat[err]).mean()), 2) for nm, r in rules.items()}
res["agreement with model's carry overall"] = {nm: round(float((r[valid] == chat[valid]).mean()), 3) for nm, r in rules.items()}
case = {}
for nm, m in (("chain: c_in=1 from a carried carry (a_{k-1}+b_{k-1}=1, c_{k-1}=1)", (a1 + b1 == 1) & (c1 == 1)), ("c_in=1 from a_{k-1}=b_{k-1}=1", (a1 & b1) == 1), ("no carry, a_{k-1}+b_{k-1}=1", (a1 + b1 == 1) & (c1 == 0)), ("no carry, a_{k-1}=b_{k-1}=0", (a1 + b1 == 0))):
    case[nm] = {"n": int((m & valid).sum()), "error rate": round(float(err[m & valid].mean()), 4)}
res["error rate by previous column state"] = case
res["error rate when an operand ends at or before k"] = round(float(err[valid & (np.minimum(la, lb) <= k)].mean()), 4); res["error rate otherwise"] = round(float(err[valid & (np.minimum(la, lb) > k)].mean()), 4)
for kk, v in res.items(): print(kk, v, flush=True)
save_json("t16d_binary_carry_errors", res); print("DONE")
