"""E6: the digit-level hypothesis. Is schoolbook state (a_k, b_k, carry, raw sum, output) represented,
and is that representation SHARED across bases?

  A. Probes for each schoolbook variable at query q_k, within base and transferred to other bases.
  B. Conflict trials for the raw sum s. Where s >= 10, base 10 emits s-10 and base 16 emits the letter s.
     A probe trained on hex that still reads s in decimal rows is evidence for a base-free sum.
  C. Variance decomposition of MLP neurons: how much of each neuron's variance does each variable explain?
     Links back to the team's ablation results: what do causally important neurons encode?
  D. Digit-pair tables: does a neuron depend on (a_k, b_k) only through a_k+b_k+c? 2-D DFT tests the
     "clock" prediction that power lies on the sum diagonal f_a == f_b.
  E. Cross-base similarity of those tables for digit pairs that are valid in both bases.
"""

import io
import pickle
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from core import BASES, save_json, schoolbook
from probe_lib import QUERY, Ridge, Table, onehot

tables = {b: Table(b) for b in BASES}
VARS = ["a", "b", "c_in", "s", "o", "c_out"]
N_CLASS = {"a": 16, "b": 16, "c_in": 2, "s": 32, "o": 16, "c_out": 2}
res: dict = {}


def school_vars(a: np.ndarray, b: np.ndarray, base: int, k: int) -> dict[str, np.ndarray]:
    """Schoolbook variables at digit k, vectorised."""
    out = {v: np.zeros(len(a), dtype=np.int64) for v in VARS}
    for i, (x, y) in enumerate(zip(a.tolist(), b.tolist())):
        sb = schoolbook(int(x), int(y), base)
        for v in VARS:
            out[v][i] = sb[v][k]
    return out


# only keep rows where digit k exists in the answer, so the output digit is a real digit, not EOS
def load_site(base: int, k: int, layer: str) -> dict:
    """Features and labels at q_k, with a prompt-level train/test split shared across bases."""
    t = tables[base]
    rows = t.select(QUERY, k)
    a, b = t.meta["a"][rows], t.meta["b"][rows]
    keep = (a + b) >= base**k
    rows, a, b = rows[keep], a[keep], b[keep]
    X = t.get(layer, rows)
    v = school_vars(a, b, base, k)
    idx = t.meta["idx"][rows]
    return {"X": X, "vars": v, "idx": idx, "a": a, "b": b, "test": idx % 5 == 0}


res["A"] = {}
for layer in ["resid0", "resid1", "mlp0", "mlp1"]:
    res["A"][layer] = {}
    for k in [0, 1, 2]:
        sites = {b: load_site(b, k, layer) for b in BASES}
        res["A"][layer][k] = {}
        for var in VARS:
            mat = np.full((4, 4), np.nan)
            chance = np.full(4, np.nan)
            for i, bx in enumerate(BASES):
                sx = sites[bx]
                tr = ~sx["test"]
                ytr = sx["vars"][var][tr]
                if len(np.unique(ytr)) < 2:
                    continue
                p = Ridge(sx["X"][tr])
                p.fit(onehot(ytr, N_CLASS[var]))
                for jj, by in enumerate(BASES):
                    sy = sites[by]
                    te = sy["test"]
                    yte = sy["vars"][var][te]
                    # score only on labels the source base could have produced
                    valid = np.isin(yte, np.unique(ytr))
                    if valid.sum() < 30:
                        continue
                    pred = p.predict(sy["X"][te][valid]).argmax(1).cpu().numpy()
                    mat[i, jj] = (pred == yte[valid]).mean()
                chance[i] = np.bincount(sx["vars"][var][sx["test"]]).max() / sx["test"].sum()
            res["A"][layer][k][var] = {"transfer": mat, "majority_chance": chance}
            if layer in ("resid0", "resid1"):
                print(f"A {layer} k={k} {var}: diag {np.round(np.diag(mat),2)}  chance {np.round(chance,2)}\n{np.round(mat,2)}", flush=True)

# ---------------------------------------------------------------------------------- B: conflict trials
res["B"] = {}
for layer in ["resid0", "resid1", "mlp0", "mlp1"]:
    res["B"][layer] = {}
    for k in [0, 1]:
        S = {b: load_site(b, k, layer) for b in [10, 16]}
        # hex-trained probe for raw sum. Evaluate on decimal rows where s in 10..15: output digit s-10 in
        # base 10 with carry, versus digit s (a letter) in base 16.
        sx = S[16]
        tr = ~sx["test"]
        p_s = Ridge(sx["X"][tr])
        p_s.fit(onehot(sx["vars"]["s"][tr], 32))
        sy = S[10]
        te = sy["test"] & (sy["vars"]["s"] >= 10) & (sy["vars"]["s"] <= 15)
        pred = p_s.predict(sy["X"][te]).argmax(1).cpu().numpy()
        s_true = sy["vars"]["s"][te]
        o_true = sy["vars"]["o"][te]
        hits_s = float((pred == s_true).mean())
        hits_o = float((pred == o_true).mean())
        # the reverse: decimal-trained probe on hex rows with s in 10..15
        sx2 = S[10]
        tr2 = ~sx2["test"]
        p2 = Ridge(sx2["X"][tr2])
        p2.fit(onehot(sx2["vars"]["s"][tr2], 32))
        sy2 = S[16]
        te2 = sy2["test"] & (sy2["vars"]["s"] >= 10) & (sy2["vars"]["s"] <= 15)
        pred2 = p2.predict(sy2["X"][te2]).argmax(1).cpu().numpy()
        # within-base reference
        within = float((p_s.predict(sx["X"][sx["test"] & (sx["vars"]["s"] >= 10) & (sx["vars"]["s"] <= 15)]).argmax(1).cpu().numpy() == sx["vars"]["s"][sx["test"] & (sx["vars"]["s"] >= 10) & (sx["vars"]["s"] <= 15)]).mean())
        res["B"][layer][k] = {
            "hex_probe_on_decimal_s10to15": {"reads_s": hits_s, "reads_decimal_output_digit": hits_o, "n": int(te.sum()), "chance_uniform_6": 1 / 6},
            "decimal_probe_on_hex_s10to15": {"reads_s": float((pred2 == sy2["vars"]["s"][te2]).mean()), "n": int(te2.sum())},
            "within_hex": within,
        }
        print("B", layer, k, res["B"][layer][k], flush=True)


# ---------------------------------------------------------------------------------- C: neuron tuning
def eta2(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Between-group variance fraction for each column (ANOVA eta squared)."""
    total = x.var(0) * len(x) + 1e-9
    between = np.zeros(x.shape[1])
    g = x.mean(0)
    for lab in np.unique(labels):
        m = labels == lab
        between += m.sum() * (x[m].mean(0) - g) ** 2
    return between / total


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == "collections" and n == "OrderedDict":
            import collections

            return collections.OrderedDict
        return _Stub

    def persistent_load(self, pid):
        return None


# the team's ablation sweep (repo): accuracy drop per component, per base
drops = {}
for f in map(str, (Path(__file__).resolve().parents[2] / "ablations").iterdir()):
    z = zipfile.ZipFile(f)
    name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    d = _U(io.BytesIO(z.read(name))).load()
    base = int(re.search(r"b=(\d+)", f).group(1))
    drops[base] = {(x["layer_idx"], x["type"], x["local_idx"]): x["accuracy_drop"] for x in d["ablations"]}

res["C"] = {}
tuning = {}
for base in BASES:
    res["C"][base] = {}
    t = tables[base]
    for layer_i, layer in enumerate(["mlp0", "mlp1"]):
        X, V, K = [], [], []
        for k in [0, 1, 2]:
            s = load_site(base, k, layer)
            X.append(s["X"])
            V.append(s["vars"])
            K.append(np.full(len(s["X"]), k))
        Xa = np.concatenate(X)
        Ka = np.concatenate(K)
        Va = {v: np.concatenate([u[v] for u in V]) for v in VARS}
        scores = {v: eta2(Xa, Va[v]) for v in VARS}
        scores["k (digit index)"] = eta2(Xa, Ka)
        scores["s mod base"] = eta2(Xa, Va["s"] % base)
        mean_activity = Xa.std(0)
        tuning[(base, layer_i)] = {"scores": scores, "std": mean_activity}
        # top tuning variable per neuron and link to ablation drop
        names = ["a", "b", "c_in", "s", "o", "c_out", "k (digit index)"]
        mat = np.stack([scores[n] for n in names], 1)
        best = mat.argmax(1)
        by_var = {}
        for vi, n in enumerate(names):
            m = best == vi
            # only neurons ablated by the team have a measured drop
            dr = [drops[base][(layer_i, "mlp", int(i))] for i in np.where(m)[0] if (layer_i, "mlp", int(i)) in drops[base]]
            by_var[n] = {"count": int(m.sum()), "mean_abl_drop": float(np.mean(dr)) if dr else None, "n_ablated": len(dr), "mean_eta2": float(mat[m, vi].mean()) if m.any() else None}
        res["C"][base][layer] = {"mean_eta2": {n: float(scores[n].mean()) for n in scores}, "by_top_variable": by_var}
        print("C", base, layer, {n: round(float(scores[n].mean()), 3) for n in scores}, flush=True)
        print("   top var:", {k2: (v2["count"], None if v2["mean_abl_drop"] is None else round(v2["mean_abl_drop"], 3)) for k2, v2 in by_var.items()}, flush=True)

# correlation: tuning strength vs causal importance, across neurons
res["C"]["tuning_vs_drop"] = {}
for base in BASES:
    for layer_i in range(2):
        s = tuning[(base, layer_i)]["scores"]
        ids = [i for i in range(256) if (layer_i, "mlp", i) in drops[base]]
        dr = np.array([drops[base][(layer_i, "mlp", i)] for i in ids])
        row = {n: float(np.corrcoef(s[n][ids], dr)[0, 1]) for n in s}
        res["C"]["tuning_vs_drop"][f"{base}/L{layer_i}"] = row
        print("tuning-drop corr", base, layer_i, {k: round(v, 2) for k, v in row.items()}, flush=True)
# same neuron's tuning across bases (is the neuron's "job" conserved?)
res["C"]["tuning_conservation"] = {}
for layer_i in range(2):
    for n in ["a", "b", "c_in", "s", "o", "c_out", "k (digit index)"]:
        m = np.corrcoef(np.stack([tuning[(b, layer_i)]["scores"][n] for b in BASES]))
        res["C"]["tuning_conservation"][f"L{layer_i}/{n}"] = m
    print("tuning conservation L", layer_i, {n: round(float(res["C"]["tuning_conservation"][f"L{layer_i}/{n}"][np.triu_indices(4, 1)].mean()), 2) for n in ["a", "b", "c_in", "s", "o", "c_out", "k (digit index)"]}, flush=True)


# ---------------------------------------------------------------------------------- D, E: digit tables
def digit_tables(base: int, k: int, layer: str) -> np.ndarray:
    """Mean activation by (a_k, b_k), on no-carry-in rows, [base, base, dim]."""
    s = load_site(base, k, layer)
    m = s["vars"]["c_in"] == 0
    X, ax, bx = s["X"][m], s["vars"]["a"][m], s["vars"]["b"][m]
    tab = np.full((base, base, X.shape[1]), np.nan)
    for i in range(base):
        for j in range(base):
            w = (ax == i) & (bx == j)
            if w.sum() >= 3:
                tab[i, j] = X[w].mean(0)
    return tab


res["D"] = {}
store = {}
for base in [8, 10, 16]:
    res["D"][base] = {}
    for layer in ["mlp0", "mlp1"]:
        tab = digit_tables(base, 0, layer)
        store[(base, layer)] = tab
        filled = np.where(np.isnan(tab), np.nanmean(tab, axis=(0, 1), keepdims=True), tab)
        fm = filled - filled.mean((0, 1), keepdims=True)
        # function-of-sum: variance of each neuron's table explained by a_k+b_k
        I, J = np.meshgrid(np.arange(base), np.arange(base), indexing="ij")
        S = I + J
        sst = (fm**2).sum((0, 1))
        sse_s = np.zeros(fm.shape[-1])
        sse_mod = np.zeros(fm.shape[-1])
        sse_add = np.zeros(fm.shape[-1])
        for s in np.unique(S):
            w = S == s
            sse_s += ((fm[w] - fm[w].mean(0)) ** 2).sum(0)
        for s in np.unique(S % base):
            w = (S % base) == s
            sse_mod += ((fm[w] - fm[w].mean(0)) ** 2).sum(0)
        # additive model: f(a) + g(b), so that sum-collapse is not credited for plain additivity
        add = fm.mean(1, keepdims=True) + fm.mean(0, keepdims=True)
        sse_add = ((fm - add) ** 2).sum((0, 1))
        wt = sst / sst.sum()
        # 2-D DFT: fraction of non-DC power at f_a == f_b (sum frequencies) and on the axes
        F = np.abs(np.fft.fft2(fm, axes=(0, 1))) ** 2
        fa, fb = np.meshgrid(np.arange(base), np.arange(base), indexing="ij")
        diag = (fa == fb) & (fa > 0)
        axes_ = ((fa == 0) | (fb == 0)) & ~((fa == 0) & (fb == 0))
        tot = F.sum((0, 1)) - F[0, 0]
        res["D"][base][layer] = {
            "power_weighted_r2_sum": float((wt * (1 - sse_s / (sst + 1e-9))).sum()),
            "power_weighted_r2_sum_mod_base": float((wt * (1 - sse_mod / (sst + 1e-9))).sum()),
            "power_weighted_r2_additive": float((wt * (1 - sse_add / (sst + 1e-9))).sum()),
            "dft_frac_on_sum_diagonal": float(F[diag].sum() / tot.sum()),
            "dft_frac_on_axes": float(F[axes_].sum() / tot.sum()),
            "dft_frac_uniform_expectation_diag": float(diag.sum() / (base * base - 1)),
            "dominant_frequencies": sorted(
                [(int(i), round(float(F[i, i].sum() / tot.sum()), 3)) for i in range(1, base // 2 + 1)],
                key=lambda z: -z[1],
            )[:4],
        }
        print("D", base, layer, res["D"][base][layer], flush=True)

# E: same neuron, same digit pair, different base. Compare wrap and non-wrap pairs separately.
res["E"] = {}
for layer in ["mlp0", "mlp1"]:
    t10, t16, t8 = store[(10, layer)], store[(16, layer)], store[(8, layer)]
    r = {}
    for name, ta, tb, nd, lim in [("10v16", t10, t16, 10, 10), ("8v16", t8, t16, 8, 8), ("8v10", t8, t10, 8, 8)]:
        I, J = np.meshgrid(np.arange(nd), np.arange(nd), indexing="ij")
        S = I + J
        A = ta[:nd, :nd]
        B = tb[:nd, :nd]
        A = A - np.nanmean(A, (0, 1), keepdims=True)
        B = B - np.nanmean(B, (0, 1), keepdims=True)

        def corr(mask: np.ndarray) -> float:
            xa, xb = A[mask].ravel(), B[mask].ravel()
            w = ~np.isnan(xa) & ~np.isnan(xb)
            return float(np.corrcoef(xa[w], xb[w])[0, 1])

        r[name] = {"no_wrap_in_either": corr(S < lim), "wraps_in_smaller_base_only": corr(S >= lim)}
    res["E"][layer] = r
    print("E", layer, r, flush=True)

save_json("s03_digit_vars", res)
print("DONE")
