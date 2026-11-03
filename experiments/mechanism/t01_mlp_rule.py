"""T01 (stored + weights): what rule do layer 1 neurons implement?

Hypothesis: each neuron reads a smooth "magnitude" feature of each digit, plus parity; it fires once the pair's total crosses a
threshold. The population then builds a piecewise function of the sum s, and the base vector shifts thresholds.
A. Digit code: replace each neuron's digit profile by its projection on smooth bases (cosine basis over digit value), with or without
   parity and an "is letter" step. Run the additive theory; compare with random bases of equal size and with the SVD reference.
B. Thresholds: per neuron, mean input by s; where it crosses zero (turns on); does it rise with s? Where do thresholds sit relative
   to the base, weighted by output importance?
C. Base vector: shift each neuron's threshold. Is the shift equal to the base difference for neurons that matter for wrapping?
D. Carry: at digit 1, fit input = alpha + beta*(a+b) + gamma*c + parity. Is gamma = beta, i.e. carry is +1 on the same sum line?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]
outnorm = Wo.norm(dim=1).cpu().numpy()
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; a, c_ = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    return dict(a=a, b=c_, c=c, s=a + c_ + c, pre=pre.cpu().numpy(), mid=mid.cpu().numpy(), idx=t.meta["idx"][rows], base=np.full(len(a), b))
D = {(b, k): load(b, k) for b in (2, 8, 10, 16) for k in (0, 1, 2)}
def pooled(k, bases=(8, 10, 16)):
    return {f: np.concatenate([D[(b, k)][f] for b in bases]) for f in D[(8, k)]}
def acc_from_pre(d, pred, tr, te):
    m = np.zeros_like(d["mid"][te])
    for b in np.unique(d["base"]):
        s = d["base"][te] == b; m[s] = d["mid"][tr][d["base"][tr] == b].mean(0)
    lg = model.unembed(torch.as_tensor(m, device=DEVICE, dtype=torch.float32) + gelu_new(torch.as_tensor(pred, device=DEVICE, dtype=torch.float32)) @ Wo + bo).argmax(-1).cpu().numpy()
    s = d["s"][te]; return {int(b): float((lg[d["base"][te] == b] == tok[s[d["base"][te] == b] % b]).mean()) for b in np.unique(d["base"])}
# ---------------- A
dv = np.arange(16.0)
def dct(m): return np.stack([np.cos(np.pi * j * (dv + .5) / 16) for j in range(m)], 1)
for k in (0, 1, 2):
    d = pooled(k); tr = d["idx"] % 2 == 0; te = ~tr; n = len(d["a"])
    oh = lambda v: np.eye(16)[v]
    bv = np.stack([d["base"] == 10, d["base"] == 16], 1).astype(np.float32)
    X = np.c_[np.ones(n), oh(d["a"]), oh(d["b"]), d["c"], bv].astype(np.float32)
    co = np.linalg.lstsq(X[tr], d["pre"][tr], rcond=None)[0]
    A_, B_ = co[1:17], co[17:33]  # digit profiles, 16 x 256 each
    rest = lambda Ap, Bp: X[:, [0, 33, 34, 35]] @ co[[0, 33, 34, 35]] + Ap[d["a"]] + Bp[d["b"]]
    out = {"full tables": acc_from_pre(d, rest(A_, B_)[te], tr, te)}
    rng = np.random.default_rng(0)
    for m in (2, 3, 4, 6, 8):
        for nm, Bs in ((f"smooth {m}", dct(m)), (f"smooth {m} + parity", np.c_[dct(m), dv % 2]), (f"smooth {m} + parity + letter", np.c_[dct(m), dv % 2, dv >= 10])):
            P = Bs @ np.linalg.pinv(Bs)
            out[nm] = acc_from_pre(d, rest(P @ A_, P @ B_)[te], tr, te)
        nb = m + 2
        rs = []
        for _ in range(5):
            R = np.c_[np.ones(16), rng.standard_normal((16, nb - 1))]; P = R @ np.linalg.pinv(R); rs.append(acc_from_pre(d, rest(P @ A_, P @ B_)[te], tr, te))
        out[f"random {nb} dims"] = {b: float(np.mean([r[b] for r in rs])) for b in rs[0]}
    res["A"][k] = out
    print(f"A digit {k}: accuracy (8,10,16)")
    for nm, r in out.items(): print(f"   {nm:32s}", [round(r[b], 3) for b in (8, 10, 16)], flush=True)
save_json("t01_mlp_rule", res)
# ---------------- B, C
gate_res = {}
for b in (8, 10, 16):
    d = D[(b, 0)]; s = d["s"]; ns = 2 * b - 1
    M = np.stack([d["pre"][s == v].mean(0) for v in range(ns)])  # [s, neuron]
    post = gelu_new(torch.as_tensor(d["pre"], device=DEVICE)).cpu().numpy()
    imp = post.std(0) * outnorm
    rho = np.array([np.corrcoef(np.arange(ns), M[:, n])[0, 1] for n in range(256)])
    th = np.full(256, np.nan)
    for n in range(256):
        y = M[:, n]
        cross = np.where(np.sign(y[:-1]) != np.sign(y[1:]))[0]
        if len(cross) == 1:
            i = cross[0]; th[n] = i + y[i] / (y[i] - y[i + 1])
    mono_up = (rho > .9) & ~np.isnan(th); mono_dn = (rho < -.9) & ~np.isnan(th)
    w = imp / imp.sum()
    res["B"][b] = {"share_importance_monotone_rising": float(w[rho > .9].sum()), "share_importance_monotone_falling": float(w[rho < -.9].sum()), "share_importance_nonmonotone": float(w[np.abs(rho) <= .9].sum()), "thresholds_rising": th[mono_up].tolist(), "thresholds_falling": th[mono_dn].tolist(), "importance_rising": imp[mono_up].tolist()}
    hist = np.histogram(th[mono_up], bins=np.arange(0, ns + 1, 2))[0]
    print(f"B b{b}: importance share rising {res['B'][b]['share_importance_monotone_rising']:.2f}, falling {res['B'][b]['share_importance_monotone_falling']:.2f}, non-monotone {res['B'][b]['share_importance_nonmonotone']:.2f}; rising turn-on points by s (bins of 2): {hist.tolist()}; near base (b-1..b+1): {int(((th[mono_up] > b - 1.5) & (th[mono_up] < b + 1.5)).sum())} of {int(mono_up.sum())}", flush=True)
    gate_res[b] = (M, th, rho, imp)
for x, y in [(8, 10), (10, 8), (10, 16), (16, 10), (8, 16), (16, 8)]:
    dx, dy = D[(x, 0)], D[(y, 0)]; lim = min(x, y); ds = []
    for i in range(lim):
        for j in range(lim):
            m1 = (dx["a"] == i) & (dx["b"] == j); m2 = (dy["a"] == i) & (dy["b"] == j)
            if m1.sum() and m2.sum(): ds.append(dy["pre"][m2].mean(0) - dx["pre"][m1].mean(0))
    v = np.stack(ds).mean(0)
    M, th, rho, imp = gate_res[x]; ns = 2 * x - 1
    slope = np.array([np.polyfit(np.arange(ns), M[:, n], 1)[0] for n in range(256)])
    dth = -v / np.where(np.abs(slope) > 1e-3, slope, np.nan)
    sel = (rho > .9) & ~np.isnan(th) & (th > 2) & (th < ns - 2)
    top = sel & (imp > np.quantile(imp[sel], .5)) if sel.sum() > 4 else sel
    res["C"][f"{x}->{y}"] = {"n": int(sel.sum()), "median_threshold_shift": float(np.nanmedian(dth[sel])), "median_shift_important_half": float(np.nanmedian(dth[top])), "expected": y - x, "iqr": [float(np.nanpercentile(dth[sel], 25)), float(np.nanpercentile(dth[sel], 75))], "corr_shift_with_threshold": float(np.corrcoef(th[sel], dth[sel])[0, 1]) if sel.sum() > 4 else None}
    print(f"C {x}->{y}: rising neurons {int(sel.sum())}, median turn-on shift {np.nanmedian(dth[sel]):+.2f} (important half {np.nanmedian(dth[top]):+.2f}; IQR {np.round(res['C'][f'{x}->{y}']['iqr'], 2).tolist()}), base difference {y - x:+d}", flush=True)
# ---------------- D
for b in (2, 8, 10, 16):
    d = D[(b, 1)]; X = np.c_[np.ones(len(d["a"])), d["a"] + d["b"], d["c"], d["a"] % 2 + d["b"] % 2].astype(float)
    co, *_ = np.linalg.lstsq(X, d["pre"], rcond=None)
    beta, gamma = co[1], co[2]
    sel = np.abs(beta) > np.quantile(np.abs(beta), .5)
    ratio = gamma[sel] / beta[sel]
    cos = float(beta @ gamma / (np.linalg.norm(beta) * np.linalg.norm(gamma)))
    Xs = np.c_[np.ones(len(d["a"])), d["s"], d["a"] % 2 + d["b"] % 2]
    r_sep = 1 - ((d["pre"] - X @ co) ** 2).sum() / ((d["pre"] - d["pre"].mean(0)) ** 2).sum()
    co2, *_ = np.linalg.lstsq(Xs, d["pre"], rcond=None); r_one = 1 - ((d["pre"] - Xs @ co2) ** 2).sum() / ((d["pre"] - d["pre"].mean(0)) ** 2).sum()
    res["D"][b] = {"cos_sum_slope_vs_carry": cos, "median_carry_over_sum_slope": float(np.median(ratio)), "iqr": [float(np.percentile(ratio, 25)), float(np.percentile(ratio, 75))], "r2_separate": float(r_sep), "r2_single_s": float(r_one)}
    print(f"D b{b} digit 1: cos(sum slope, carry) {cos:.2f}, carry / sum slope median {np.median(ratio):.2f} (IQR {np.round(res['D'][b]['iqr'], 2).tolist()}), R2 with a+b and c separate {r_sep:.3f} vs single s {r_one:.3f}", flush=True)
save_json("t01_mlp_rule", res); print("DONE")
