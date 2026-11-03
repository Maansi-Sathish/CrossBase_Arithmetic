"""F16 (stored + weights): sufficiency test. Replace all real activations with a tiny additive theory.

pre(k, base, a, b, c) = mu_k + A_k[a] + B_k[b] + c * C_k + V_k[base]    (fitted by least squares on half the prompts)
Then logits = unembed(mid_k_mean + GELU(pre) @ W_out + bias). Nothing from the real prompt is used except a, b, c, base, k.

Variants: full theory; shared digit tables with per-base tables as upper reference; dropping each part (carry, base, b...).
Parameters are counted to show how small the theory is. Also feeds unseen combinations: digit pairs never valid in that base's
training are not used; held-out prompts only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {}
tok = np.array([model.vocab[c] for c in DIGITS])
Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; ak, bk = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    return dict(a=ak, b=bk, c=c, pre=pre.cpu().numpy(), mid=mid.cpu().numpy(), idx=t.meta["idx"][rows], base=np.full(len(ak), b))
def design(d, parts):
    n = len(d["a"]); cols = [np.ones((n, 1))]
    if "a" in parts: cols.append(np.eye(16)[d["a"]][:, 1:])
    if "b" in parts: cols.append(np.eye(16)[d["b"]][:, 1:])
    if "c" in parts: cols.append(d["c"][:, None].astype(float))
    if "base" in parts: cols.append(np.stack([d["base"] == x for x in BASES[1:]], 1).astype(float))
    if "cxb" in parts: cols.append(np.stack([(d["base"] == x) * d["c"] for x in BASES[1:]], 1).astype(float))
    return np.concatenate(cols, 1)
def pooled(k, bases):
    ds = [load(b, k) for b in bases]
    return {f: np.concatenate([d[f] for d in ds]) for f in ds[0]}
def evaluate(d, parts, label):
    tr = d["idx"] % 2 == 0; te = ~tr
    X = design(d, parts)
    coef, *_ = np.linalg.lstsq(X[tr].astype(np.float32), d["pre"][tr], rcond=None)
    pred = X[te] @ coef
    r2 = 1 - ((pred - d["pre"][te]) ** 2).sum() / ((d["pre"][te] - d["pre"][te].mean(0)) ** 2).sum()
    # mid: reduced to its mean per (base), no digits
    m = np.zeros_like(d["mid"][te])
    for b in BASES:
        sel = d["base"][te] == b
        if sel.any(): m[sel] = d["mid"][tr][d["base"][tr] == b].mean(0)
    pr = torch.as_tensor(pred, device=DEVICE, dtype=torch.float32); mt = torch.as_tensor(m, device=DEVICE, dtype=torch.float32)
    lg = model.unembed(mt + gelu_new(pr) @ Wo + bo).argmax(-1).cpu().numpy()
    out = {"pre_r2": float(r2), "params": X.shape[1]}
    s = d["a"][te] + d["b"][te] + d["c"][te]
    for b in np.unique(d["base"]):
        sel = d["base"][te] == b
        out[f"acc_b{b}"] = float((lg[sel] == tok[s[sel] % b]).mean())
    return out
for k in (0, 1, 2):
    res[k] = {}
    d = pooled(k, BASES)
    for label, parts in (("full theory: code(a)+code(b)+carry+base", {"a", "b", "c", "base"}), ("no carry", {"a", "b", "base"}), ("no base vector", {"a", "b", "c"}), ("no b", {"a", "c", "base"}), ("no a", {"b", "c", "base"}), ("carry depends on base", {"a", "b", "c", "base", "cxb"})):
        if k == 0 and "c" not in parts and label == "no carry": continue
        r = evaluate(d, parts, label); res[k][label] = r
        print(f"k{k} {label:42s} params {r['params']:3d} preR2 {r['pre_r2']:.3f} acc", {b: round(r[f'acc_b{b}'], 3) for b in BASES if f'acc_b{b}' in r}, flush=True)
    d2 = pooled(k, [8, 10, 16])
    r = evaluate(d2, {"a", "b", "c", "base"}, "x"); res[k]["no binary in fit"] = r
    print(f"k{k} {'theory fit on 8,10,16 only':42s} params {r['params']:3d} preR2 {r['pre_r2']:.3f} acc", {b: round(r[f'acc_b{b}'], 3) for b in (8, 10, 16)}, flush=True)
    for b in BASES:
        db = load(b, k)
        r = evaluate(db, {"a", "b", "c"}, "per base"); res[k][f"per-base tables b{b}"] = r
        print(f"k{k} {'tables fit only on base ' + str(b):42s} params {r['params']:3d} preR2 {r['pre_r2']:.3f} acc", round(r[f'acc_b{b}'], 3), flush=True)
    save_json("f16_synthesis", res)
print("DONE")
