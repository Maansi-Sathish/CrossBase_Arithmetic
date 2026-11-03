"""S10 (stored + weights, CPU, no generation): mean ablation, carry logic, margins, absent digits, symmetry,
copying, EOS, weight analyses, shared subspaces, neuron rules."""
import io, pickle, re, sys, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.tree import DecisionTreeRegressor, export_text
from core import BASES, DIGITS, Model, save_json
from probe_lib import QUERY, Ridge, Table, onehot

torch.set_grad_enabled(False)
model = Model()
sd = model.sd
tok = np.array([model.vocab[c] for c in DIGITS])
tok_t = torch.as_tensor(tok)
WE = model.WE
Wo1 = sd["h.1.mlp.c_proj.weight"]
res = {}
T = {b: Table(b) for b in BASES}


def qrows(base, correct=True):
    t = T[base]
    m = t.meta
    q = np.where((m["kind"] == QUERY) & ((m["correct"] == 1) | (not correct)))[0]
    a, b, k, idx = m["a"][q].astype(np.int64), m["b"][q].astype(np.int64), m["j"][q].astype(np.int64), m["idx"][q]
    M = base**k
    ak, bk = (a // M) % base, (b // M) % base
    c = (((a % M) + (b % M)) >= M).astype(int)
    la = np.array([len(np.base_repr(int(x), base)) for x in a])
    lb = np.array([len(np.base_repr(int(x), base)) for x in b])
    ans = (a + b) // M
    eos = ((a + b) < M) & ~((a + b == 0) & (k == 0))
    return dict(q=q, a=a, b=b, k=k, idx=idx, ak=ak, bk=bk, c=c, s=ak + bk + c, o=(ak + bk + c) % base, la=la, lb=lb, eos=eos)


Q = {b: qrows(b) for b in BASES}


def margin(resid, truth):
    lg = model.unembed(torch.as_tensor(resid))
    idx = torch.arange(len(lg))
    t = lg[idx, truth].clone()
    lg2 = lg.clone()
    lg2[idx, truth] = -1e9
    return (t - lg2.max(-1).values).numpy()


# ---- A. mean vs zero ablation for all L1 neurons
res["A"] = {}
for b in BASES:
    d = Q[b]
    post = torch.as_tensor(T[b].get("mlp1", d["q"]))
    resid = torch.as_tensor(T[b].get("resid1", d["q"]))
    truth = tok[d["o"]].copy()
    truth[d["eos"]] = model.eos
    uniq = np.unique(d["idx"])
    pos = np.searchsorted(uniq, d["idx"])
    ex = lambda ok: float((np.bincount(pos, weights=(~ok).astype(float), minlength=len(uniq)) == 0).mean())
    base = ex(model.unembed(resid).argmax(-1).numpy() == truth)
    mean = post.mean(0)
    z, mn = np.zeros(256), np.zeros(256)
    for n in range(256):
        z[n] = base - ex(model.unembed(resid - post[:, n : n + 1] * Wo1[n]).argmax(-1).numpy() == truth)
        mn[n] = base - ex(model.unembed(resid - (post[:, n : n + 1] - mean[n]) * Wo1[n]).argmax(-1).numpy() == truth)
    res["A"][b] = {"zero": z, "mean": mn, "corr": float(np.corrcoef(z, mn)[0, 1]), "n_zero_gt05": int((z > .05).sum()), "n_mean_gt05": int((mn > .05).sum()), "n177_zero": z[177], "n177_mean": mn[177], "top_mean": [(int(i), round(float(mn[i]), 3)) for i in np.argsort(-mn)[:8]]}
    print("A", b, {k: v for k, v in res["A"][b].items() if k not in ("zero", "mean")}, flush=True)

# ---- B. generate / propagate / kill at digit 2
res["B"] = {}
for b in BASES:
    d = Q[b]
    sel = (d["k"] == 2) & ~d["eos"]
    q = d["q"][sel]
    a, bb = d["a"][sel], d["b"][sel]
    raw = (a // b) % b + (bb // b) % b
    cls = np.where(raw >= b, "G", np.where(raw == b - 1, "P", "K"))
    c = d["c"][sel]
    te = d["idx"][sel] % 5 == 0
    resid = T[b].get("resid1", q)
    mg = margin(resid, tok[d["o"][sel]])
    row = {}
    for f in ("resid0", "mlp0", "mlp1"):
        X = T[b].get(f, q)
        p = Ridge(X[~te])
        p.fit(onehot(c[~te], 2))
        pred = p.predict(X[te]).argmax(1).numpy()
        row[f] = {g: float((pred[cls[te] == g] == c[te][cls[te] == g]).mean()) for g in "GPK" if (cls[te] == g).sum() >= 20}
    row["margin"] = {g: float(mg[cls == g].mean()) for g in "GPK"}
    row["margin_P_by_carry"] = {int(x): float(mg[(cls == "P") & (c == x)].mean()) for x in (0, 1)} if (cls == "P").sum() else {}
    row["n"] = {g: int((cls == g).sum()) for g in "GPK"}
    res["B"][b] = row
    print("B", b, row, flush=True)

# ---- C. margin anatomy + binary errors
res["C"] = {}
for b in BASES:
    d = Q[b]
    mask = ~d["eos"]
    resid = T[b].get("resid1", d["q"][mask])
    mg = margin(resid, tok[d["o"][mask]])
    k, c, s, la, lb = d["k"][mask], d["c"][mask], d["s"][mask], d["la"][mask], d["lb"][mask]
    shorter = k >= np.minimum(la, lb)
    res["C"][b] = {"k": {int(x): float(mg[k == x].mean()) for x in range(5) if (k == x).any()}, "carry_in": {int(x): float(mg[c == x].mean()) for x in (0, 1)}, "wraps": {int(x): float(mg[(s >= b) == x].mean()) for x in (0, 1)}, "past_shorter_operand": {int(x): float(mg[shorter == x].mean()) for x in (0, 1)}, "len_equal": {int(x): float(mg[(la == lb) == x].mean()) for x in (0, 1)}}
    print("C", b, {k2: {a2: round(v, 2) for a2, v in v2.items()} for k2, v2 in res["C"][b].items()}, flush=True)
# exact accuracy by operand-length gap (from all rows at q0)
for b in BASES:
    t = T[b]
    rows = t.select(QUERY, 0)
    a, bb, ok = t.meta["a"][rows], t.meta["b"][rows], t.meta["correct"][rows]
    la = np.array([len(np.base_repr(int(x), b)) for x in a]); lb = np.array([len(np.base_repr(int(x), b)) for x in bb])
    gap = np.abs(la - lb)
    res["C"][f"gap{b}"] = {int(g): (float(ok[gap == g].mean()), int((gap == g).sum())) for g in np.unique(gap)}
    small = a + bb < b**2
    res["C"][f"small{b}"] = (float(ok[small].mean()) if small.any() else None, int(small.sum()), float(ok[~small].mean()))
    print("C gap", b, {g: (round(v[0], 3), v[1]) for g, v in res["C"][f"gap{b}"].items()}, "small sum", res["C"][f"small{b}"], flush=True)

# ---- D. absent digit vs real zero
res["D"] = {}
for b in BASES:
    d = Q[b]
    out = {}
    for k in (1, 2):
        sel = (d["k"] == k) & ~d["eos"]
        ab = sel & (d["la"] <= k) & (d["lb"] > k)
        zr = sel & (d["ak"] == 0) & (d["la"] > k) & (d["lb"] > k)
        if ab.sum() < 30 or zr.sum() < 30:
            continue
        X = T[b].get("mlp1", d["q"])
        same, other = [], []
        for x in range(b):
            for cv in (0, 1):
                m1, m2 = ab & (d["bk"] == x) & (d["c"] == cv), zr & (d["bk"] == x) & (d["c"] == cv)
                if m1.sum() >= 5 and m2.sum() >= 5:
                    same.append(np.linalg.norm(X[m1].mean(0) - X[m2].mean(0)))
                for y in range(b):
                    m3 = zr & (d["bk"] == y) & (d["c"] == cv)
                    if y != x and m1.sum() >= 5 and m3.sum() >= 5:
                        other.append(np.linalg.norm(X[m1].mean(0) - X[m3].mean(0)))
        out[k] = {"dist_absent_vs_zero_same_digit": float(np.mean(same)), "dist_absent_vs_zero_other_digit": float(np.mean(other)), "n_absent": int(ab.sum())}
    res["D"][b] = out
    print("D", b, out, flush=True)

# ---- E. symmetry of tables
res["E"] = {}
for b in (8, 10, 16):
    t = T[b]
    rows = t.select(QUERY, 0)
    a, bb = t.meta["a"][rows] % b, t.meta["b"][rows] % b
    row = {}
    for f in ("mlp0", "mlp1"):
        X = t.get(f, rows)
        tab = np.stack([[X[(a == i) & (bb == j)].mean(0) for j in range(b)] for i in range(b)])
        tc = tab - tab.mean((0, 1), keepdims=True)
        anti = (tc - tc.transpose(1, 0, 2)) / 2
        row[f] = float((anti**2).sum() / (tc**2).sum())
    res["E"][b] = row
    print("E antisymmetric variance share", b, row, flush=True)

# ---- F. copying from a few-shot example
res["F"] = {}
for b in BASES:
    t = T[b]
    rows = t.select(QUERY, 0)
    idx = t.meta["idx"][rows]
    a0, b0 = t.meta["a"][rows] % b, t.meta["b"][rows] % b
    sh = t.shots[idx]
    match = ((sh[:, :, 0] % b == a0[:, None]) & (sh[:, :, 1] % b == b0[:, None])).any(1)
    d = (a0 + b0) % b
    mg = margin(t.get("resid1", rows), tok[d])
    res["F"][b] = {"matched": float(mg[match].mean()), "unmatched": float(mg[~match].mean()), "n_matched": int(match.sum()), "se": float(mg[match].std() / np.sqrt(max(match.sum(), 1)))}
    print("F", b, res["F"][b], flush=True)

# ---- G. EOS
res["G"] = {}
eos_write = (Wo1 @ WE[model.eos]).numpy() - (Wo1 @ WE[torch.as_tensor(tok)].T).numpy().mean(1)
for b in BASES:
    d = Q[b]
    te = d["idx"] % 5 == 0
    y = d["eos"].astype(int)
    row = {"rate": float(y.mean())}
    for f in ("resid0", "mlp0", "mlp1"):
        X = T[b].get(f, d["q"])
        p = Ridge(X[~te]); p.fit(onehot(y[~te], 2))
        pr = p.predict(X[te]).argmax(1).numpy()
        conflict = (d["k"][te] == np.maximum(d["la"], d["lb"])[te])  # all operand digits used: EOS unless final carry
        row[f] = {"acc": float((pr == y[te]).mean()), "conflict_acc": float((pr[conflict] == y[te][conflict]).mean()), "conflict_n": int(conflict.sum())}
    post = T[b].get("mlp1", d["q"])
    contrib = post.mean(0) * 0
    m1, m0 = post[y == 1].mean(0), post[y == 0].mean(0)
    top = np.argsort(-((m1 - m0) * eos_write))[:5]
    row["top_neurons"] = [int(i) for i in top]
    res["G"][b] = row
    print("G", b, row, flush=True)

# ---- H. L0 writes to L1 query/key/value vs importance
class _S:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == "collections" and n == "OrderedDict":
            import collections
            return collections.OrderedDict
        return _S
    def persistent_load(self, pid): return None
drops = {}
for f in (Path(__file__).resolve().parents[2] / "ablations").iterdir():
    z = zipfile.ZipFile(f)
    dd = _U(io.BytesIO(z.read([n for n in z.namelist() if n.endswith("data.pkl")][0]))).load()
    drops[int(re.search(r"b=(\d+)", f.name).group(1))] = {x["local_idx"]: x["accuracy_drop"] for x in dd["ablations"] if x["layer_idx"] == 0 and x["type"] == "mlp"}
Wo0 = sd["h.0.mlp.c_proj.weight"]
ca = sd["h.1.attn.c_attn.weight"]
parts = {"q": ca[:, :64], "k": ca[:, 64:128], "v": ca[:, 128:192]}
norms = {nm: (Wo0 @ w).norm(dim=1).numpy() for nm, w in parts.items()}
head_q = np.stack([(Wo0 @ parts["q"][:, 16 * h : 16 * h + 16]).norm(dim=1).numpy() for h in range(4)], 1)
res["H"] = {}
for b in BASES:
    ids = sorted(drops[b])
    dr = np.array([drops[b][i] for i in ids])
    std = T[b].get("mlp0", np.where(T[b].meta["kind"] == QUERY)[0][:20000]).std(0)[ids]
    row = {nm: float(np.corrcoef(norms[nm][ids] * std, dr)[0, 1]) for nm in norms}
    row["raw_write_norm"] = float(np.corrcoef(Wo0.norm(dim=1).numpy()[ids] * std, dr)[0, 1])
    frac = (norms["q"] / (norms["q"] + norms["k"] + norms["v"]))[ids]
    hi = dr > np.quantile(dr, 0.9)
    row["query_share_high_drop"] = float(frac[hi].mean())
    row["query_share_other"] = float(frac[~hi].mean())
    row["n_high"] = int(hi.sum())
    res["H"][b] = row
    print("H", b, {k: round(v, 3) for k, v in row.items()}, flush=True)

# ---- I. where base shift reaches L1 neurons: residual vs each head; J. sublayer logit split
Win = sd["h.1.mlp.c_fc.weight"]
cp = sd["h.1.attn.c_proj.weight"]
res["I"] = {}
cell_means = {}
for b in BASES:
    t = T[b]
    rows = t.select(QUERY, 0)
    a0, b0 = t.meta["a"][rows] % b, t.meta["b"][rows] % b
    cell_means[b] = (torch.as_tensor(t.get("resid0", rows)), torch.as_tensor(t.get("z1", rows)), a0, b0)
for x, y in [(8, 10), (10, 16), (8, 16)]:
    Rx, Zx, ax, bx = cell_means[x]; Ry, Zy, ay, by = cell_means[y]
    lim = min(x, y)
    dR, dH = [], []
    for i in range(lim):
        for j in range(lim):
            mx, my = torch.as_tensor((ax == i) & (bx == j)), torch.as_tensor((ay == i) & (by == j))
            if mx.sum() < 3 or my.sum() < 3:
                continue
            dR.append(Ry[my].mean(0) - Rx[mx].mean(0))
            dZ = Zy[my].mean(0) - Zx[mx].mean(0)
            hs = []
            for h in range(4):
                v = torch.zeros(64); v[16 * h : 16 * h + 16] = dZ[16 * h : 16 * h + 16]
                hs.append(v @ cp)
            dH.append(torch.stack(hs))
    dR = torch.stack(dR); dH = torch.stack(dH)
    pR = (dR @ Win).pow(2).sum().item()
    pH = [(dH[:, h] @ Win).pow(2).sum().item() for h in range(4)]
    tot = pR + sum(pH)
    res["I"][f"{x}v{y}"] = {"residual": pR / tot, "heads": [p / tot for p in pH]}
    print("I", x, y, {"residual": round(pR / tot, 3), "heads": [round(p / tot, 3) for p in pH]}, flush=True)
res["J"] = {}
Wa = cp
for b in BASES:
    d = Q[b]
    for k in (0, 1):
        sel = np.where((d["k"] == k) & ~d["eos"])[0]
        q = d["q"][sel]
        pre = torch.as_tensor(T[b].get("resid0", q))
        z = torch.as_tensor(T[b].get("z1", q))
        post = torch.as_tensor(T[b].get("mlp1", q))
        attn = z @ Wa + sd["h.1.attn.c_proj.bias"]
        mo = post @ Wo1 + sd["h.1.mlp.c_proj.bias"]
        res_all = pre + attn + mo
        scale = sd["ln_f.weight"] / (res_all.std(-1, keepdim=True, unbiased=False) + 1e-5)
        o = torch.as_tensor(d["o"][sel])
        digs = tok_t[:b]
        out = {}
        for nm, v in (("L0 residual", pre), ("L1 attn", attn), ("L1 mlp", mo)):
            lg = ((v - v.mean(-1, keepdim=True)) * scale) @ WE.T
            true = lg[torch.arange(len(o)), tok_t[o]]
            others = lg[:, digs].mean(1)
            out[nm] = float((true - others).mean())
            # lens accuracy from this component alone
            out[nm + " top1"] = float((lg[:, digs].argmax(1) == o).float().mean())
        out["mid top1"] = float((model.unembed(pre + attn)[:, digs].argmax(1) == o).float().mean())
        out["pre top1"] = float((model.unembed(pre)[:, digs].argmax(1) == o).float().mean())
        res["J"][f"{b}/k{k}"] = out
        print("J", b, k, {a: round(v, 2) for a, v in out.items()}, flush=True)

# ---- K. shared vs private subspaces; L. neuron rules
res["K"] = {}
for f in ("resid1", "mlp1", "resid0"):
    Vs = {}
    for b in BASES:
        rows = T[b].select(QUERY, 0)
        X = T[b].get(f, rows)
        Vs[b] = np.linalg.svd(X - X.mean(0), full_matrices=False)
    tab = {}
    for dim in (4, 8, 16):
        ov = np.zeros((4, 4))
        for i, x in enumerate(BASES):
            for j, y in enumerate(BASES):
                ov[i, j] = (np.linalg.norm(Vs[x][2][:dim] @ Vs[y][2][:dim].T) ** 2) / dim
        d_full = Vs[8][2].shape[1]
        tab[dim] = {"overlap": ov, "random_overlap": dim / d_full, "var_captured": [float((Vs[b][1][:dim] ** 2).sum() / (Vs[b][1] ** 2).sum()) for b in BASES]}
        print("K", f, dim, "overlap 2-8,8-10,10-16,8-16,2-16", np.round([ov[0, 1], ov[1, 2], ov[2, 3], ov[1, 3], ov[0, 3]], 2), "random", round(dim / d_full, 2), flush=True)
    res["K"][f] = tab
res["L"] = {}
feat_names = ["k", "a_k", "b_k", "carry", "s", "o", "s>=base", "is_even_s", "base2", "base8", "base10", "base16"]
Xf, Ys, Gs = [], [], []
for bi, b in enumerate(BASES):
    d = Q[b]
    sel = (d["k"] <= 2) & ~d["eos"]
    sb = np.where(sel)[0]
    sb = sb[np.random.default_rng(b).permutation(len(sb))[:15000]]
    Xf.append(np.c_[d["k"][sb], d["ak"][sb], d["bk"][sb], d["c"][sb], d["s"][sb], d["o"][sb], d["s"][sb] >= b, d["s"][sb] % 2 == 0, *[np.full(len(sb), bi == i) for i in range(4)]])
    Ys.append(T[b].get("mlp1", d["q"][sb]))
    Gs.append(d["idx"][sb] % 5 == 0)
Xf, Ys, Gs = np.concatenate(Xf).astype(float), np.concatenate(Ys), np.concatenate(Gs)
mean_drop = np.mean([res["A"][b]["mean"] for b in BASES], 0)
zero_drop = np.mean([res["A"][b]["zero"] for b in BASES], 0)
for n in list(np.argsort(-mean_drop)[:8]) + [177]:
    n = int(n)
    t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=50, random_state=0).fit(Xf[~Gs], Ys[~Gs, n])
    r = t.score(Xf[Gs], Ys[Gs, n])
    res["L"][n] = {"r2": r, "mean_drop": float(mean_drop[n]), "zero_drop": float(zero_drop[n]), "tree": export_text(t, feature_names=feat_names, decimals=2)}
    print(f"L neuron {n} mean drop {mean_drop[n]:.3f} tree R2 {r:.2f}\n{res['L'][n]['tree']}", flush=True)
save_json("s10_rest", res)
print("DONE")
