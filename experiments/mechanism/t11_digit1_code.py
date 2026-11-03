"""T11 (stored + weights). Q1: why digit-1 neurons are not a function of the sum. Q3: why hex digit 0 needs more than two bumps.

Q1. Readout accuracy when each neuron's activity is replaced by a table over different variables (fit on half, test on other half):
    s=a+b+c ; (a+b, c) ; (a, b, c) ; (a+b, c, a0+b0) [graded carry].
    Then the additive pre-activation pre ~ mu + A[a] + B[b] + c*C: is A+B a function of a+b? Compare A and B per neuron.
Q3. Hex digit 0: per-neuron formulas with 2 and 3 bumps + parity, and how many sums each neuron covers.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; res = {"Q1": {}, "Q3": {}}
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; a, bb = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = (model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]).cpu().numpy()
    return dict(a=a, b=bb, c=c, s=a + bb + c, p0=(A % b) + (B % b), post=t.get("mlp1", rows), pre=pre, resid=t.get("resid1", rows), idx=t.meta["idx"][rows], base=b)
def score(d, Xhat, te):
    r = torch.as_tensor(d["resid"][te], device=DEVICE) + (torch.as_tensor(Xhat[te], device=DEVICE, dtype=torch.float32) - torch.as_tensor(d["post"][te], device=DEVICE)) @ Wo
    return float((model.unembed(r).argmax(-1).cpu().numpy() == tok[d["s"][te] % d["base"]]).mean())
def table(d, keys, tr):
    X = d["post"]; T = np.zeros_like(X); key = np.stack(keys, 1); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    fall = X[tr].mean(0); cov = 0
    for g in np.unique(inv):
        m = inv == g; mt = m & tr
        T[m] = X[mt].mean(0) if mt.any() else fall; cov += (m & ~tr).sum() if mt.any() else 0
    return T, cov / (~tr).sum()
for b in (8, 10, 16):
    for k in (0, 1):
        d = load(b, k); tr = d["idx"] % 2 == 0; te = ~tr; out = {}
        for nm, keys in (("s = a+b+c", [d["s"]]), ("(a+b, c)", [d["a"] + d["b"], d["c"]]), ("(a, b, c)", [d["a"], d["b"], d["c"]]), ("(a+b, c, a0+b0)", [d["a"] + d["b"], d["c"], d["p0"]])):
            T, cov = table(d, keys, tr); out[nm] = score(d, T, te); out[nm + " test coverage"] = float(cov)
        # additive fit of pre-activation; is A[a]+B[b] a function of a+b?
        P = d["pre"]; Xd = np.c_[np.eye(b)[d["a"]], np.eye(b)[d["b"]][:, 1:], d["c"]]
        co = np.linalg.lstsq(np.c_[Xd, np.ones(len(P))], P, rcond=None)[0]; Acode = co[:b]; Bcode = np.r_[np.zeros((1, 256)), co[b:2 * b - 1]]
        Acode -= Acode.mean(0); Bcode -= Bcode.mean(0)
        imp = np.abs(d["post"] - d["post"].mean(0)).mean(0) * np.linalg.norm(Wo.cpu().numpy(), axis=1); w = imp / imp.sum()
        cosAB = np.array([np.dot(Acode[:, n], Bcode[:, n]) / (np.linalg.norm(Acode[:, n]) * np.linalg.norm(Bcode[:, n]) + 1e-9) for n in range(256)])
        # share of the variance of A[a]+B[b] over all (a,b) pairs that is a function of a+b
        AA, BB = np.meshgrid(np.arange(b), np.arange(b), indexing="ij"); S = (AA + BB).ravel()
        F = Acode[AA.ravel()] + Bcode[BB.ravel()]
        fs = np.stack([F[S == v].mean(0) for v in range(2 * b - 1)])[S]
        share = 1 - ((F - fs) ** 2).sum(0) / ((F - F.mean(0)) ** 2).sum(0).clip(1e-9)
        out["weighted cos(A code, B code)"] = float((w * cosAB).sum()); out["weighted share of A+B that depends only on a+b"] = float((w * share).sum())
        res["Q1"][f"{b}/k{k}"] = out
        print(f"Q1 b{b} digit {k}:", {kk: round(v, 3) for kk, v in out.items() if "coverage" not in kk}, flush=True)
# Q3
def bumps_pred(d, tr, nb, oo, ee):
    s = d["s"].astype(float); X = d["post"]; sv = np.arange(s.max() + 1)
    Ms = np.stack([X[tr & (d["s"] == v)].mean(0) if (tr & (d["s"] == v)).any() else X[tr].mean(0) for v in sv])
    grid = [(c, w) for c in np.arange(0, sv.max() + 1, 1.0) for w in (0.5, 0.8, 1.2, 2.0, 3.5, 6.0)]
    P = np.zeros_like(X)
    for n in range(256):
        chosen = []
        for _ in range(nb):
            best = (np.inf, None)
            for g in grid:
                D = np.column_stack([np.ones(len(sv))] + [np.exp(-0.5 * ((sv - c) / w_) ** 2) for c, w_ in chosen + [g]])
                co = np.linalg.lstsq(D, Ms[:, n], rcond=None)[0]; e = ((Ms[:, n] - D @ co) ** 2).sum()
                if e < best[0]: best = (e, g)
            chosen.append(best[1])
        D = np.column_stack([np.ones(len(s))] + [np.exp(-0.5 * ((s - c) / w_) ** 2) for c, w_ in chosen] + [oo, ee])
        P[:, n] = D @ np.linalg.lstsq(D[tr], X[tr, n], rcond=None)[0]
    return P
for b in (10, 16):
    d = load(b, 0); tr = d["idx"] % 2 == 0; te = ~tr
    oo = ((d["a"] % 2 == 1) & (d["b"] % 2 == 1)).astype(float); ee = ((d["a"] % 2 == 0) & (d["b"] % 2 == 0)).astype(float)
    out = {f"{nb} bumps + parity": score(d, bumps_pred(d, tr, nb, oo, ee), te) for nb in (2, 3, 4)}
    X = d["post"]; Ms = np.stack([X[d["s"] == v].mean(0) for v in range(2 * b - 1)])
    imp = np.abs(X - X.mean(0)).mean(0); top = np.argsort(-imp)[:60]
    act = (Ms[:, top] > 0.3 * Ms[:, top].max(0)).sum(0)
    out["sums covered per top-60 neuron (median)"] = float(np.median(act)); out["sums covered as share of range"] = float(np.median(act) / (2 * b - 1))
    out["neurons strongly active per sum (median)"] = float(np.median((Ms > 0.5).sum(1)))
    res["Q3"][b] = out; print(f"Q3 b{b} digit 0:", {kk: round(v, 3) for kk, v in out.items()}, flush=True)
save_json("t11_digit1_code", res); print("DONE")
