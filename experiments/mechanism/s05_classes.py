"""S05 (stored data only, CPU): the team's percentile classes. Do survivors differ, and do classes mark bases?

Uses the team's accuracy drops for 321 shortlisted components (same rule as plot_circuits.compute_survivors)
and stored activations. No forward passes, so no class knockouts.
"""
import io
import pickle
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from scipy.stats import spearmanr
from core import BASES, save_json
from probe_lib import QUERY, Ridge, Table, onehot

ROOT = Path(__file__).resolve().parents[2]


class _S:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass


class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == "collections" and n == "OrderedDict":
            import collections

            return collections.OrderedDict
        return _S

    def persistent_load(self, pid):
        return None


drops = {}
for f in (ROOT / "ablations").iterdir():
    z = zipfile.ZipFile(f)
    d = _U(io.BytesIO(z.read([n for n in z.namelist() if n.endswith("data.pkl")][0]))).load()
    drops[int(re.search(r"b=(\d+)", f.name).group(1))] = {(x["layer_idx"], x["type"], x["local_idx"]): x["accuracy_drop"] for x in d["ablations"]}
comps = sorted(drops[10])
print("shortlisted", len(comps), {L: sum(c[0] == L for c in comps) for L in range(2)}, "heads", sum(c[1] == "head" for c in comps), flush=True)
res = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}


def survivors(base, L, p):
    ids = [c for c in comps if c[0] == L]
    v = np.array([drops[base][c] for c in ids])
    thr = np.percentile(v, p)
    return {c for c, x in zip(ids, v) if x >= thr}, float(thr)


# A: membership. Thresholds are relative, so report them.
for p in (90, 95, 99):
    for L in range(2):
        S = {b: survivors(b, L, p) for b in BASES}
        sets = [S[b][0] for b in BASES]
        counts = {}
        for s in sets:
            for c in s:
                counts[c] = counts.get(c, 0) + 1
        res["A"][f"p{p}/L{L}"] = {"size": [len(s) for s in sets], "threshold_drop": [S[b][1] for b in BASES], "heads_in_class": [sum(c[1] == "head" for c in s) for s in sets], "in_all_4": sum(v == 4 for v in counts.values()), "in_exactly_1": sum(v == 1 for v in counts.values()), "members_all_4": sorted(c for c, v in counts.items() if v == 4), "exactly_1_by_base": {b: sorted(c for c in S[b][0] if counts[c] == 1) for b in BASES}}
        r = res["A"][f"p{p}/L{L}"]
        print(f"A p{p} L{L}: sizes {r['size']} thresholds {np.round(r['threshold_drop'], 3).tolist()} heads {r['heads_in_class']} all4 {r['in_all_4']} exactly1 {r['in_exactly_1']}", flush=True)

# D: continuous agreement of effects across bases (no thresholds)
for L in range(2):
    ids = [c for c in comps if c[0] == L and c[1] == "mlp"]
    M = np.array([[drops[b][c] for c in ids] for b in BASES])
    res["D"][f"L{L}"] = {"spearman": spearmanr(M.T).statistic, "mean_abs_drop": np.abs(M).mean(1), "n_drop_gt_05": (M > 0.05).sum(1)}
    print(f"D L{L} drop rank correlation (2,8,10,16):\n", np.round(res["D"][f"L{L}"]["spearman"], 2), "n>0.05", res["D"][f"L{L}"]["n_drop_gt_05"], flush=True)

# B + C need activations at q_0..q_2
VARS = {"a": 16, "b": 16, "c": 2, "s": 32, "o": 16, "c_out": 2}


def labels(a, b, base, k):
    ak, bk = (a // base**k) % base, (b // base**k) % base
    m = base**k
    c = (((a % m) + (b % m)) >= m).astype(int) if k else np.zeros(len(a), int)
    s = ak + bk + c
    return {"a": ak, "b": bk, "c": c, "s": s, "o": s % base, "c_out": (s >= base).astype(int)}


def eta2(x, y):
    g = x.mean(0)
    between = np.zeros(x.shape[1])
    for v in np.unique(y):
        m = y == v
        between += m.sum() * (x[m].mean(0) - g) ** 2
    return between / (((x - g) ** 2).sum(0) + 1e-9)


rng = np.random.default_rng(0)
tab = {b: Table(b) for b in BASES}
activity = {}
for base in BASES:
    t = tab[base]
    for L in range(2):
        X, K, V = [], [], {v: [] for v in VARS}
        per_k = {}
        for k in (0, 1, 2):
            rows = t.select(QUERY, k)
            a, b = t.meta["a"][rows], t.meta["b"][rows]
            keep = (a + b) >= base**k
            rows, a, b = rows[keep], a[keep], b[keep]
            x = t.get(f"mlp{L}", rows)
            lab = labels(a, b, base, k)
            per_k[k] = (x, lab, t.meta["idx"][rows] % 5 == 0, a + b)
            X.append(x)
            K.append(np.full(len(x), k))
            for v in VARS:
                V[v].append(lab[v])
        Xa, Ka = np.concatenate(X), np.concatenate(K)
        Va = {v: np.concatenate(u) for v, u in V.items()}
        eta = {v: eta2(Xa, Va[v]) for v in VARS}
        eta["k"] = eta2(Xa, Ka)
        activity[(base, L)] = Xa.std(0)
        groups = {}
        for p in (90, 95, 99):
            groups[f"p{p}"] = sorted(c[2] for c in survivors(base, L, p)[0] if c[1] == "mlp")
        s90 = set(groups["p90"])
        groups["shortlisted, below p90"] = sorted(c[2] for c in comps if c[0] == L and c[1] == "mlp" and c[2] not in s90)
        shortlisted = {c[2] for c in comps if c[0] == L and c[1] == "mlp"}
        groups["not shortlisted"] = sorted(set(range(256)) - shortlisted)
        res["B"][f"{base}/L{L}"] = {g: {"n": len(i), **{v: float(eta[v][i].mean()) for v in eta}, "activity_std": float(Xa.std(0)[i].mean())} for g, i in groups.items() if i}
        print(f"B b{base} L{L}", {g: (r["n"], {v: round(r[v], 2) for v in ("a", "b", "c", "s", "o", "k")}, round(r["activity_std"], 2)) for g, r in res["B"][f"{base}/L{L}"].items()}, flush=True)

        def probe(ids, x, y, te, n):
            p = Ridge(x[~te][:, ids])
            p.fit(onehot(y[~te], n))
            return float((p.predict(x[te][:, ids]).argmax(1).cpu().numpy() == y[te]).mean())

        for k in (0, 1):
            x, lab, te, ab = per_k[k]
            for gname in ("p90", "p95", "p99"):
                ids = groups[gname]
                if len(ids) < 2:
                    continue
                row = {}
                for v, n in VARS.items():
                    if len(np.unique(lab[v][~te])) < 2:
                        continue
                    obs = probe(ids, x, lab[v], te, n)
                    null = [probe(list(rng.choice(256, len(ids), replace=False)), x, lab[v], te, n) for _ in range(10)]
                    row[v] = {"class": obs, "random_mean": float(np.mean(null)), "random_max": float(np.max(null))}
                res["C"][f"{base}/L{L}/k{k}/{gname}"] = {"n": len(ids), **row}
                print(f"C b{base} L{L} k{k} {gname} n={len(ids)}", {v: (round(r["class"], 2), round(r["random_mean"], 2)) for v, r in row.items()}, flush=True)

# E: does a neuron that survives in exactly one base fire more in that base? (Does the class mark a base?)
for L in range(2):
    S = {b: {c[2] for c in survivors(b, L, 90)[0] if c[1] == "mlp"} for b in BASES}
    A = np.stack([activity[(b, L)] for b in BASES])  # [bases, neurons]
    rows = {}
    for i, b in enumerate(BASES):
        only = sorted(S[b] - set().union(*[S[x] for x in BASES if x != b]))
        if only:
            rows[b] = {"n": len(only), "own_base_activity_rank_mean": float(np.mean([(A[:, n] > A[i, n]).sum() + 1 for n in only])), "activity_std_by_base": A[:, only].mean(1)}
    res["E"][f"L{L}"] = rows
    print(f"E L{L} (rank 1 = fires most in its own base; 2.5 = chance)", {b: (r["n"], round(r["own_base_activity_rank_mean"], 2), np.round(r["activity_std_by_base"], 2).tolist()) for b, r in rows.items()}, flush=True)
save_json("s05_classes", res)
print("DONE")
