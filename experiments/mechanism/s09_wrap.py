"""S09 (stored + weights, CPU): how wrapping works. Step-threshold neurons, the non-additive part of the digit
tables, and linear translation between bases fitted on no-wrap pairs and tested on wrap pairs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from core import BASES, DIGITS, Model, save_json
from probe_lib import QUERY, Table

model = Model()
Wo = model.sd["h.1.mlp.c_proj.weight"].numpy()
tok = [model.vocab[c] for c in DIGITS]
write = Wo @ model.WE[tok].numpy().T
res = {"step": {}, "profile": {}, "nonadd": {}, "map": {}}


def site(base, k, f):
    t = Table(base)
    rows = t.select(QUERY, k)
    a, b = t.meta["a"][rows], t.meta["b"][rows]
    keep = ((a + b) >= base**k) & (t.meta["correct"][rows] == 1)
    rows, a, b = rows[keep], a[keep], b[keep]
    m = base**k
    c = (((a % m) + (b % m)) >= m).astype(int) if k else np.zeros(len(a), int)
    return t.get(f, rows), (a // m) % base, (b // m) % base, c


# ---------- step neurons at digit 0
S = {}
for base in BASES:
    X, a, b, c = site(base, 0, "mlp1")
    s = a + b
    n_s = 2 * base - 1
    T = np.stack([X[s == v].mean(0) for v in range(n_s)])
    W = np.array([(s == v).sum() for v in range(n_s)], float)
    var = X.var(0)
    thr = np.full(256, -1)
    fit = np.zeros(256)
    between = ((T - np.average(T, axis=0, weights=W)) ** 2 * W[:, None]).sum(0)
    for n in range(256):
        best = -1
        for t in range(1, n_s):
            lo, hi = np.average(T[:t, n], weights=W[:t]), np.average(T[t:, n], weights=W[t:])
            sse = (W[:t] * (T[:t, n] - lo) ** 2).sum() + (W[t:] * (T[t:, n] - hi) ** 2).sum()
            r = 1 - sse / (between[n] + 1e-9)
            if r > best:
                best, thr[n] = r, t
        fit[n] = best
    active = var > 0.01
    good = active & (fit >= 0.8)
    S[base] = (thr, fit, active, T, var)
    hist = {int(t): int((thr[good] == t).sum()) for t in np.unique(thr[good])}
    res["step"][base] = {"n_active": int(active.sum()), "n_step": int(good.sum()), "threshold_hist": hist, "at_base": int((thr[good] == base).sum()), "drop_var_share_step": float(var[good].sum() / var[active].sum())}
    print("STEP", base, res["step"][base], flush=True)
for x, y in [(8, 10), (10, 16), (8, 16)]:
    tx, fx, ax, *_ = S[x]
    ty, fy, ay, *_ = S[y]
    both = ax & ay & (fx >= 0.8) & (fy >= 0.8)
    res["step"][f"{x}v{y}"] = {"both_step": int(both.sum()), "same_threshold": int((tx[both] == ty[both]).sum()), "each_at_own_base": int(((tx == x) & (ty == y) & both).sum()), "neurons_each_at_own_base": np.where((tx == x) & (ty == y) & both)[0].tolist()}
    print("STEP pair", x, y, res["step"][f"{x}v{y}"], flush=True)

# ---------- key neurons: firing by s, what they write
for n in (177, 22, 119, 174, 198):
    row = {"write_norm": float(np.linalg.norm(Wo[n])), "write_norm_rank": int((np.linalg.norm(Wo, axis=1) > np.linalg.norm(Wo[n])).sum()) + 1, "writes_top": [DIGITS[i] for i in np.argsort(-write[n])[:4]], "writes_bottom": [DIGITS[i] for i in np.argsort(write[n])[:4]]}
    for base in BASES:
        row[f"by_s_b{base}"] = np.round(S[base][3][:, n], 2).tolist()
        K = {}
        for k in (0, 1, 2):
            X, a, b, c = site(base, k, "mlp1")
            K[k] = (float(X[:, n].mean()), float(X[:, n].std()))
        row[f"by_k_b{base}"] = K
    res["profile"][n] = row
    print("NEURON", n, {k: v for k, v in row.items() if k.startswith(("write", "by_k"))}, "\n   s b10", row["by_s_b10"], "\n   s b16", row["by_s_b16"], flush=True)

# ---------- non-additive part
for base in (8, 10, 16):
    res["nonadd"][base] = {}
    for L in ("mlp1",):
        X, a, b, c = site(base, 0, L)
        tab = np.zeros((base, base, X.shape[1]))
        ok = np.zeros((base, base), bool)
        for i in range(base):
            for j in range(base):
                m = (a == i) & (b == j)
                if m.sum() >= 2:
                    tab[i, j] = X[m].mean(0)
                    ok[i, j] = True
        assert ok.all()
        t = tab - tab.mean((0, 1), keepdims=True)
        add = t.mean(1, keepdims=True) + t.mean(0, keepdims=True)
        I = t - add  # interaction
        share = float((I**2).sum() / (t**2).sum())
        s = np.add.outer(np.arange(base), np.arange(base))
        U, sv, Vt = np.linalg.svd(I.reshape(base * base, -1), full_matrices=False)
        comp = U[:, :6] * sv[:6]

        def cv_r2(labels, Y):
            # leave-one-cell-out prediction from the other cells with the same label
            pred = np.zeros_like(Y)
            lab = labels.ravel()
            for c_ in range(len(lab)):
                others = (lab == lab[c_]) & (np.arange(len(lab)) != c_)
                pred[c_] = Y[others].mean(0) if others.any() else 0
            return float(1 - ((Y - pred) ** 2).sum() / (Y**2).sum())

        Y = I.reshape(base * base, -1)
        feats = {"s": s, "s mod base": s % base, "s >= base": (s >= base).astype(int), "s bucket: output + carry": s}
        res["nonadd"][base] = {"interaction_share": share, "sv_share_top6": (sv[:6] ** 2 / (sv**2).sum()), "cv_r2": {k: cv_r2(v, Y) for k, v in feats.items() if k != "s bucket: output + carry"}, "cv_r2_top_comp": {k: [cv_r2(v, comp[:, [i]]) for i in range(4)] for k, v in feats.items() if k != "s bucket: output + carry"}}
        print("NONADD", base, {a2: (np.round(b2, 3).tolist() if not isinstance(b2, (dict, float)) else (round(b2, 3) if isinstance(b2, float) else {k: np.round(v, 2).tolist() for k, v in b2.items()})) for a2, b2 in res["nonadd"][base].items()}, flush=True)

# ---------- translation between bases on digit-pair cells
rng = np.random.default_rng(0)


def cells(base, f):
    X, a, b, c = site(base, 0, f)
    out = {}
    for i in range(base):
        for j in range(base):
            m = (a == i) & (b == j)
            if m.sum() >= 2:
                out[(i, j)] = X[m].mean(0)
    return out


def fit_eval(xtr, ytr, xte, yte, lam=1e-1, center=None):
    mx, my = xtr.mean(0), ytr.mean(0)
    A = xtr - mx
    W = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ (ytr - my))
    pred = (xte - mx) @ W + my
    base_pred = np.tile(my, (len(yte), 1))
    return float(1 - ((pred - yte) ** 2).sum() / ((base_pred - yte) ** 2).sum())


for f in ("resid0", "resid1", "mlp1"):
    C = {b: cells(b, f) for b in (8, 10, 16)}
    allv = np.concatenate([np.stack(list(C[b].values())) for b in C])
    Vt = np.linalg.svd(allv - allv.mean(0), full_matrices=False)[2][:12]
    for x, y in [(8, 10), (10, 8), (10, 16), (16, 10), (8, 16), (16, 8)]:
        lim = min(x, y)
        nw = [p for p in C[x] if p in C[y] and p[0] < lim and p[1] < lim and sum(p) < lim]
        wr = [p for p in C[x] if p in C[y] and p[0] < lim and p[1] < lim and sum(p) >= lim]
        vals, wrapv, nulls = [], [], []
        for seed in range(20):
            r = np.random.default_rng(seed)
            perm = r.permutation(len(nw))
            cut = int(0.7 * len(nw))
            tr, te = [nw[i] for i in perm[:cut]], [nw[i] for i in perm[cut:]]
            Xt = (np.stack([C[x][p] for p in tr])) @ Vt.T
            Yt = (np.stack([C[y][p] for p in tr])) @ Vt.T
            vals.append(fit_eval(Xt, Yt, np.stack([C[x][p] for p in te]) @ Vt.T, np.stack([C[y][p] for p in te]) @ Vt.T))
            wrapv.append(fit_eval(Xt, Yt, np.stack([C[x][p] for p in wr]) @ Vt.T, np.stack([C[y][p] for p in wr]) @ Vt.T))
            nulls.append(fit_eval(Xt, Yt[r.permutation(len(Yt))], np.stack([C[x][p] for p in te]) @ Vt.T, np.stack([C[y][p] for p in te]) @ Vt.T))
        res["map"][f"{f}/{x}->{y}"] = {"held_out_no_wrap_r2": float(np.mean(vals)), "wrap_r2": float(np.mean(wrapv)), "shuffled_pairs_r2": float(np.mean(nulls)), "n_no_wrap": len(nw), "n_wrap": len(wr)}
        print("MAP", f, x, "->", y, {k: round(v, 3) if isinstance(v, float) else v for k, v in res["map"][f"{f}/{x}->{y}"].items()}, flush=True)
save_json("s09_wrap", res)
print("DONE")
