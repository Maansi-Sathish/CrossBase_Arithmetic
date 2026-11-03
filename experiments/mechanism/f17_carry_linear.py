"""F17 (forward passes + stored): is the carry computed linearly as a + b - written?

A. Random previous digits and written digit (including impossible combinations). Carry evidence = logit(s+1) - logit(s).
   Fit it as alpha*a + beta*b + gamma*w. Prediction: alpha ~ beta > 0, gamma < 0, and a threshold rule on a+b-w explains outputs.
B. Theory without carry input: pre = mu + code(a_k) + code(b_k) + code(a_{k-1}) + code(b_{k-1}) + code(o_{k-1}) + base vector. Does it reach
   high accuracy? If yes, carry is computed inside the layer 1 MLP from additive parts.
C. Binary: count model, pre as a function of s only; thresholds; extra terms at digit 1.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, render, sample_shots, save_json, to_base
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]
def run(strs, bs=256):
    out = []
    for i in range(0, len(strs), bs):
        seqs = [[model.vocab[c] for c in s] for s in strs[i:i + bs]]
        L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
        lg, _ = model.forward(x.to(DEVICE)); out.append(lg[torch.arange(len(seqs), device=DEVICE), q].cpu().numpy())
    return np.concatenate(out)
for base in BASES:
    for k in (1, 2):
        rng = random.Random(base * 37 + k); S, rec = [], []
        M = base ** (k - 1) if k >= 1 else 1
        while len(S) < 6000:
            a, b_ = rng.randint(base**k, 999), rng.randint(base**k, 999)
            if base**k > 999: continue
            ap, bp = rng.randrange(base), rng.randrange(base); w = rng.randrange(base)
            def setd(n, j, d): return n + (d - (n // base**j) % base) * base**j
            A, B = setd(a, k - 1, ap), setd(b_, k - 1, bp)
            sa, sb = to_base(A, base), to_base(B, base)
            if len(sa) <= k or len(sb) <= k: continue
            true_low = to_base(A + B, base)[::-1]
            pref = (true_low[:k - 1] + DIGITS[w]) if k >= 2 else DIGITS[w]
            cur = ((A // base**k) % base + (B // base**k) % base)
            S.append("\n".join(render(*s, base) for s in sample_shots(rng, (A, B))) + f"\n{sa}+{sb}={pref}")
            rec.append((ap, bp, w, cur))
        L = run(S); rec = np.array(rec); ap, bp, w, cur = rec.T
        c0 = tok[cur % base]; c1 = tok[(cur + 1) % base]
        ev = L[np.arange(len(L)), c1] - L[np.arange(len(L)), c0]
        pr = L.argmax(1)
        chose1 = pr == c1; chose0 = pr == c0
        valid = chose0 | chose1
        X = np.c_[ap, bp, w, np.ones(len(ap))].astype(float)
        coef, *_ = np.linalg.lstsq(X, ev, rcond=None)
        r2 = 1 - ((ev - X @ coef) ** 2).sum() / ((ev - ev.mean()) ** 2).sum()
        # one-hot additive fit, then saturated
        D = np.c_[np.eye(base)[ap][:, 1:], np.eye(base)[bp][:, 1:], np.eye(base)[w][:, 1:], np.ones(len(ap))]
        c2, *_ = np.linalg.lstsq(D, ev, rcond=None); r2add = 1 - ((ev - D @ c2) ** 2).sum() / ((ev - ev.mean()) ** 2).sum()
        cell = ap * 400 + bp * 20 + w; m = np.zeros(len(ev))
        for cv in np.unique(cell): m[cell == cv] = ev[cell == cv].mean()
        r2sat = 1 - ((ev - m) ** 2).sum() / ((ev - ev.mean()) ** 2).sum()
        lin = ap + bp - w
        best = max(((float((((lin > th).astype(int) == chose1)[valid]).mean()), th) for th in np.arange(-base, 2 * base) + .5))
        res["A"][f"{base}/k{k}"] = {"alpha": coef[0], "beta": coef[1], "gamma": coef[2], "linear_r2": float(r2), "additive_onehot_r2": float(r2add), "saturated_r2": float(r2sat), "threshold_rule_acc": best[0], "best_threshold": best[1], "best_threshold_over_base": best[1] / base, "valid": float(valid.mean())}
        print(f"A b{base} k{k}: a {coef[0]:+.2f} b {coef[1]:+.2f} written {coef[2]:+.2f} | R2 linear {r2:.2f}, additive {r2add:.2f}, full table {r2sat:.2f} | rule a+b-w > {best[1]} ({best[1] / base:.2f} base): {best[0]:.3f}", flush=True)
    save_json("f17_carry_linear", res)
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; pm = b ** (k - 1)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    c = (((A % M) + (B % M)) >= M).astype(int)
    return dict(a=(A // M) % b, b=(B // M) % b, ap=(A // pm) % b, bp=(B // pm) % b, w=((A + B) // pm) % b, c=c, pre=pre.cpu().numpy(), mid=mid.cpu().numpy(), idx=t.meta["idx"][rows], base=np.full(len(A), b))
def fit(d, cols, lab, bases):
    tr = d["idx"] % 2 == 0; te = ~tr; X = np.concatenate(cols, 1).astype(np.float32)
    coef, *_ = np.linalg.lstsq(X[tr], d["pre"][tr], rcond=None); pred = X[te] @ coef
    m = np.zeros_like(d["mid"][te])
    for b in bases:
        s = d["base"][te] == b
        if s.any(): m[s] = d["mid"][tr][d["base"][tr] == b].mean(0)
    lg = model.unembed(torch.as_tensor(m, device=DEVICE, dtype=torch.float32) + gelu_new(torch.as_tensor(pred, device=DEVICE, dtype=torch.float32)) @ Wo + bo).argmax(-1).cpu().numpy()
    s = d["a"][te] + d["b"][te] + d["c"][te]
    r2 = 1 - ((pred - d["pre"][te]) ** 2).sum() / ((d["pre"][te] - d["pre"][te].mean(0)) ** 2).sum()
    return {"params": X.shape[1], "pre_r2": float(r2), **{f"acc_b{b}": float((lg[d["base"][te] == b] == tok[s[d["base"][te] == b] % b]).mean()) for b in bases}}
for k in (1, 2):
    ds = [load(b, k) for b in (8, 10, 16)]; d = {f: np.concatenate([x[f] for x in ds]) for f in ds[0]}
    one = lambda v: np.eye(16)[v][:, 1:]
    i = np.ones((len(d["a"]), 1)); bv = np.stack([d["base"] == 10, d["base"] == 16], 1)
    for name, cols in (("with carry input", [i, one(d["a"]), one(d["b"]), d["c"][:, None], bv]), ("prev digits + written, no carry input", [i, one(d["a"]), one(d["b"]), one(d["ap"]), one(d["bp"]), one(d["w"]), bv]), ("linear a+b-w axis only (1 number)", [i, one(d["a"]), one(d["b"]), (d["ap"] + d["bp"] - d["w"])[:, None] / d["base"][:, None], bv]), ("prev digits only", [i, one(d["a"]), one(d["b"]), one(d["ap"]), one(d["bp"]), bv]), ("written only", [i, one(d["a"]), one(d["b"]), one(d["w"]), bv])):
        res["B"][f"k{k}/{name}"] = fit(d, cols, name, (8, 10, 16))
        print(f"B k{k} {name:40s}", {a: round(v, 3) for a, v in res['B'][f'k{k}/{name}'].items()}, flush=True)
for k in (0, 1, 2, 3, 4):
    d = load(2, k); s = d["a"] + d["b"] + d["c"]; i = np.ones((len(s), 1)); e = lambda v, n=2: np.eye(n)[v][:, 1:]
    variants = {"count s only": [i, np.eye(4)[s][:, 1:]], "a,b,c": [i, e(d["a"]), e(d["b"]), d["c"][:, None]]}
    if k > 0:
        variants["a,b,c + prev + written"] = [i, e(d["a"]), e(d["b"]), d["c"][:, None], e(d["ap"]), e(d["bp"]), e(d["w"])]
        variants["a,b, prev, written (no c)"] = [i, e(d["a"]), e(d["b"]), e(d["ap"]), e(d["bp"]), e(d["w"])]
        variants["full interaction of a,b,c"] = [i, np.eye(8)[d["a"] * 4 + d["b"] * 2 + d["c"]][:, 1:]]
    for nm, cols in variants.items():
        res["C"][f"k{k}/{nm}"] = fit(d, cols, nm, (2,))
        print(f"C binary k{k} {nm:32s}", {a: round(v, 3) for a, v in res['C'][f'k{k}/{nm}'].items()}, flush=True)
save_json("f17_carry_linear", res); print("DONE")
