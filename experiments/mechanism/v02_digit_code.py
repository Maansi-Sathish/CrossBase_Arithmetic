"""V02 (stored + weights): what does the digit code contain, and what is the smallest code that still works?

A. Fit pre = mu + A[a] + B[b] + c*C + V[base] on bases 8, 10, 16. SVD of the digit tables: what is each component, as a function of digit?
B. Ladder of smaller codes: replace A, B with simple features, regenerate neuron inputs, run GELU + readout, score on held-out prompts.
C. How parity gets computed: neuron tables by (parity of a, parity of b). XOR needs firing on both even+even and odd+odd.
D. Is pre a function of a+b? Share of A,B variance on a linear digit axis, whether A and B use the same axes.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; ak, bk = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    return dict(a=ak, b=bk, c=c, pre=pre.cpu().numpy(), mid=mid.cpu().numpy(), idx=t.meta["idx"][rows], base=np.full(len(ak), b))
def pooled(k):
    ds = [load(b, k) for b in (8, 10, 16)]
    return {f: np.concatenate([x[f] for x in ds]) for f in ds[0]}
def score(d, X, tr, te):
    coef = np.linalg.lstsq(X[tr].astype(np.float32), d["pre"][tr], rcond=None)[0]; pred = X[te] @ coef
    m = np.zeros_like(d["mid"][te])
    for b in (8, 10, 16):
        s = d["base"][te] == b; m[s] = d["mid"][tr][d["base"][tr] == b].mean(0)
    lg = model.unembed(torch.as_tensor(m, device=DEVICE, dtype=torch.float32) + gelu_new(torch.as_tensor(pred, device=DEVICE, dtype=torch.float32)) @ Wo + bo).argmax(-1).cpu().numpy()
    s = d["a"][te] + d["b"][te] + d["c"][te]
    return {"params": X.shape[1], "pre_r2": float(1 - ((pred - d["pre"][te]) ** 2).sum() / ((d["pre"][te] - d["pre"][te].mean(0)) ** 2).sum()), **{f"b{b}": float((lg[d["base"][te] == b] == tok[s[d["base"][te] == b] % b]).mean()) for b in (8, 10, 16)}}
for k in (0, 1, 2):
    d = pooled(k); tr = d["idx"] % 2 == 0; te = ~tr
    n = len(d["a"]); i = np.ones((n, 1)); oh = lambda v: np.eye(16)[v][:, 1:]
    bv = np.stack([d["base"] == 10, d["base"] == 16], 1).astype(float); c = d["c"][:, None].astype(float)
    X = np.c_[i, oh(d["a"]), oh(d["b"]), c, bv]
    coef = np.linalg.lstsq(X[tr].astype(np.float32), d["pre"][tr], rcond=None)[0]
    TA = np.r_[np.zeros((1, 256)), coef[1:16]]; TB = np.r_[np.zeros((1, 256)), coef[16:31]]
    TAc, TBc = TA - TA.mean(0), TB - TB.mean(0)
    U, S, Vt = np.linalg.svd(np.r_[TAc, TBc], full_matrices=False)
    share = (S**2 / (S**2).sum())
    digs = np.arange(16.0); z = (digs - digs.mean()) / digs.std(); par = (digs % 2 == 0).astype(float) - .5
    prof = {}
    for j in range(5):
        pa = TAc @ Vt[j]; pb = TBc @ Vt[j]
        fit = {"linear": np.corrcoef(pa, z)[0, 1] ** 2, "parity": np.corrcoef(pa, par)[0, 1] ** 2, "is letter": np.corrcoef(pa, digs >= 10)[0, 1] ** 2, "mod4": np.corrcoef(pa, (digs % 4 >= 2).astype(float))[0, 1] ** 2}
        prof[j] = {"share": float(share[j]), "a_profile": pa.tolist(), "corr_a_b_profile": float(np.corrcoef(pa, pb)[0, 1]), **{a2: float(v) for a2, v in fit.items()}}
    res["A"][k] = {"components": prof, "share_top8": share[:8].tolist()}
    print(f"A k{k} variance share {np.round(share[:8], 3).tolist()}")
    for j, v in prof.items():
        print(f"   PC{j}: share {v['share']:.2f}, a~b {v['corr_a_b_profile']:+.2f}, R2 linear {v['linear']:.2f} parity {v['parity']:.2f} letter {v['is letter']:.2f} mod4 {v['mod4']:.2f}; by digit {np.round(np.array(v['a_profile']), 1).tolist()}", flush=True)
    # D same axes?
    res["D"][k] = {"cos_mean_A_B_tables": float(np.corrcoef(TAc.ravel(), TBc.ravel())[0, 1])}
    # B ladder
    a_, b_ = d["a"].astype(float), d["b"].astype(float); pa_, pb_ = (a_ % 2 == 0).astype(float), (b_ % 2 == 0).astype(float)
    f4 = lambda v: np.c_[(v % 4 == 0), (v % 4 == 1), (v % 4 == 2)].astype(float)
    let = lambda v: (v >= 10).astype(float)
    ladder = {
        "linear a+b only": [i, a_ + b_, c, bv],
        "linear a, b (separate)": [i, a_, b_, c, bv],
        "+ parity": [i, a_, b_, pa_, pb_, c, bv],
        "+ parity + squares": [i, a_, b_, pa_, pb_, a_**2, b_**2, c, bv],
        "+ parity + squares + mod 4": [i, a_, b_, pa_, pb_, a_**2, b_**2, f4(a_), f4(b_), c, bv],
        "+ parity x linear": [i, a_, b_, pa_, pb_, pa_ * a_, pb_ * b_, a_**2, b_**2, c, bv],
        "full tables (15+15)": [i, oh(d["a"]), oh(d["b"]), c, bv],
    }
    res["B"][k] = {}
    for nm, cols in ladder.items():
        res["B"][k][nm] = score(d, np.column_stack(cols), tr, te)
    for r in (1, 2, 3, 4, 6, 8, 12):
        Va = TAc @ Vt[:r].T; Vb = TBc @ Vt[:r].T
        X2 = np.c_[i, Va[d["a"]], Vb[d["b"]], c, bv]
        res["B"][k][f"top {r} components of the digit code"] = score(d, X2, tr, te)
    print(f"B k{k} accuracy (8,10,16):")
    for nm, r in res["B"][k].items():
        print(f"   {nm:38s} params {r['params']:3d} preR2 {r['pre_r2']:.2f} acc {r['b8']:.3f} {r['b10']:.3f} {r['b16']:.3f}", flush=True)
# C parity computation, digit 0
for b in (2, 8, 10, 16):
    d = load(b, 0); Q = gelu_new(torch.as_tensor(d["pre"], device=DEVICE)).cpu().numpy()
    a, c_ = d["a"] % 2, d["b"] % 2
    M = np.zeros((2, 2, 256))
    for x in (0, 1):
        for y in (0, 1):
            M[x, y] = Q[(a == x) & (c_ == y)].mean(0)
    even = (M[0, 0] + M[1, 1]) / 2; odd = (M[0, 1] + M[1, 0]) / 2
    xor = even - odd
    top = np.argsort(-np.abs(xor))[:10]
    both = float(np.mean([(M[0, 0, n] - odd[n]) * (M[1, 1, n] - odd[n]) > 0 for n in top]))
    res["C"][b] = {"top_neurons": top.tolist(), "fraction_top10_same_sign_on_EE_and_OO": both, "table_top": M[:, :, top[:6]].tolist(), "xor_share_of_parity_variance": float((xor[top] ** 2).sum() / (((M - M.mean((0, 1), keepdims=True))[:, :, top]) ** 2).sum() * 1)}
    print(f"C b{b}: top parity neurons {top[:6].tolist()}, fire same way on even+even and odd+odd in {both:.2f} of top 10; tables (EE,EO,OE,OO):", [np.round([M[0, 0, n], M[0, 1, n], M[1, 0, n], M[1, 1, n]], 2).tolist() for n in top[:4]], flush=True)
save_json("v02_digit_code", res); print("DONE")
