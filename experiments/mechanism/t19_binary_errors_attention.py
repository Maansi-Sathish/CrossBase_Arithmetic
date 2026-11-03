"""T19 (forward passes): do binary carry errors come from mis-addressed reads? Teacher forced at digits 1-5 on many binary
problems. For each L1 head at the query: attention on columns k-2, k-1, k, k+1 of a and b, and on '+', '=' and the written digits.
Compare errors with correct cases at the same k. Then a causal check on errors: replace each head's output with its mean over
correct cases that share (k, a_{k-1}, b_{k-1}, c_in) - if one head's output is the culprit, that should fix the error."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, sample_problems, save_json, schoolbook, to_base
torch.set_grad_enabled(False)
model = Model(); tok = np.array([model.vocab[c] for c in DIGITS]); res = {}
P = [p for p in sample_problems(16000, 2, seed=95) if len(p.prompt) + 14 <= 256]
S, info = [], []
for p in P:
    sb = schoolbook(p.a, p.b, 2); la, lb = len(to_base(p.a, 2)), len(to_base(p.b, 2)); st = len(p.prompt) - la - lb - 2
    for k in range(1, min(len(p.answer), 6)):
        S.append(p.prompt + p.answer[:k]); info.append((k, sb["a"][k], sb["b"][k], sb["c_in"][k], sb["a"][k - 1], sb["b"][k - 1], st, la, lb))
I = np.array(info); N = len(S); print("cases", N, flush=True)
feats = {h: [] for h in range(4)}; pred = []; Z = []
cols = (-2, -1, 0, 1)
for i in range(0, N, 300):
    ch = S[i:i + 300]; seqs = [[model.vocab[x] for x in s] for s in ch]; n = max(map(len, seqs)); X = torch.full((len(ch), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    X = X.to(DEVICE); q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); J = torch.arange(len(ch), device=DEVICE)
    lg, c = model.forward(X, capture={"pattern1", "z1"}); pred.append(lg[J, q].argmax(-1).cpu().numpy()); Z.append(c["z1"][J, q].cpu())
    Ii = torch.as_tensor(I[i:i + 300], device=DEVICE); k, st, la, lb = Ii[:, 0], Ii[:, 6], Ii[:, 7], Ii[:, 8]; pl = st + la; eq = pl + lb + 1
    for h in range(4):
        r = c["pattern1"][J, h, q]; row = []
        for d in cols:
            kk = k + d; pa = pl - 1 - kk; pb = eq - 1 - kk
            va = (kk >= 0) & (kk < la); vb = (kk >= 0) & (kk < lb)
            row.append(torch.where(va, r[J, pa.clamp(0, n - 1)], torch.zeros_like(r[:, 0]))); row.append(torch.where(vb, r[J, pb.clamp(0, n - 1)], torch.zeros_like(r[:, 0])))
        row.append(r[J, pl]); row.append(r[J, eq]); feats[h].append(torch.stack(row, 1).cpu().numpy())
pred = np.concatenate(pred); Z = torch.cat(Z); k, ak, bk, c_in, a1, b1 = I[:, 0], I[:, 1], I[:, 2], I[:, 3], I[:, 4], I[:, 5]
ok = pred == tok[(ak + bk + c_in) % 2]; err = ~ok; print("errors", int(err.sum()), flush=True)
names = [f"{x}{d:+d}" for d in cols for x in ("a", "b")] + ["+", "="]
for h in range(4):
    F = np.concatenate(feats[h]); res[f"L1H{h}"] = {nm: (round(float(F[ok][:, j].mean()), 3), round(float(F[err][:, j].mean()), 3)) for j, nm in enumerate(names)}
    print(f"L1H{h} attention (correct, error):", res[f"L1H{h}"], flush=True)
# causal: replace a head's output at error cases with the mean output of correct cases sharing (k, a_k, b_k, a_{k-1}, b_{k-1}, c_in)
key = np.c_[k, ak, bk, a1, b1, c_in]; fix = {}
err_idx = np.where(err)[0]
for h in range(4):
    fixed = 0; tot = 0
    for i in range(0, len(err_idx), 100):
        idx = err_idx[i:i + 100]; ch = [S[j] for j in idx]; seqs = [[model.vocab[x] for x in s] for s in ch]; n = max(map(len, seqs)); X = torch.full((len(ch), n), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
        X = X.to(DEVICE); q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); J = torch.arange(len(ch), device=DEVICE)
        rep = torch.stack([Z[np.where(ok & (key == key[j]).all(1))[0]][:, h].mean(0) for j in idx]).to(DEVICE)
        def fn(t, rep=rep, q=q, J=J, h=h):
            t = t.clone(); t[J, q, h] = rep; return t
        lg, _ = model.forward(X, hooks={"z1": fn}); p2 = lg[J, q].argmax(-1).cpu().numpy(); fixed += int((p2 == tok[(ak[idx] + bk[idx] + c_in[idx]) % 2]).sum()); tot += len(idx)
    fix[f"L1H{h}"] = round(fixed / max(tot, 1), 2)
res["errors fixed by replacing one head's output with its typical value"] = fix; res["n errors"] = int(err.sum()); res["n"] = int(N)
print("fixed by head:", fix, flush=True)
save_json("t19_binary_errors_attention", res); print("DONE")
