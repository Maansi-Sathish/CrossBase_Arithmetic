"""S06 (stored activations only, CPU): what each head's output carries, and where the base is readable.

Part A: probes on each stored head output z (16-d) at q_0..q_2 for schoolbook variables, per base and transferred.
Part B: pooled 4-way base probe at each test-line position and layer, on integers never seen in training.
Base is partly visible in operand tokens, so also scored on prompts whose test-line tokens are all 0-1 (ambiguous).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from core import BASES, save_json
from probe_lib import A_DIGIT, B_DIGIT, PLUS, QUERY, Ridge, Table, onehot, input_onehot

T = {b: Table(b) for b in BASES}
VARS = {"a": 16, "b": 16, "c": 2, "s": 32, "o": 16, "c_out": 2}
res = {"A": {}, "B": {}}


def labels(a, b, base, k):
    ak, bk = (a // base**k) % base, (b // base**k) % base
    m = base**k
    c = (((a % m) + (b % m)) >= m).astype(int) if k else np.zeros(len(a), int)
    s = ak + bk + c
    return {"a": ak, "b": bk, "c": c, "s": s, "o": s % base, "c_out": (s >= base).astype(int)}


def site(base, k, L):
    t = T[base]
    rows = t.select(QUERY, k)
    a, b = t.meta["a"][rows], t.meta["b"][rows]
    keep = (a + b) >= base**k
    rows, a, b = rows[keep], a[keep], b[keep]
    return t.get(f"z{L}", rows), labels(a, b, base, k), t.meta["idx"][rows] % 5 == 0


for L in range(2):
    for k in (0, 1, 2):
        S = {b: site(b, k, L) for b in BASES}
        for h in range(4):
            d = range(h * 16, (h + 1) * 16)
            row = {}
            for v, n in VARS.items():
                m = np.full((4, 4), np.nan)
                for i, bx in enumerate(BASES):
                    Xx, lx, te = S[bx]
                    if len(np.unique(lx[v][~te])) < 2:
                        continue
                    p = Ridge(Xx[~te][:, list(d)])
                    p.fit(onehot(lx[v][~te], n))
                    seen = np.unique(lx[v][~te])
                    for j, by in enumerate(BASES):
                        Xy, ly, tey = S[by]
                        val = np.isin(ly[v][tey], seen)
                        if val.sum() >= 30:
                            m[i, j] = (p.predict(Xy[tey][val][:, list(d)]).argmax(1).cpu().numpy() == ly[v][tey][val]).mean()
                row[v] = m
            res["A"][f"L{L}H{h}/k{k}"] = row
            print(f"A L{L}H{h} k{k} within-base", {v: np.round(np.diag(m), 2).tolist() for v, m in row.items() if v in ("a", "b", "c", "s", "o")}, flush=True)

SITES = {"a_0": (A_DIGIT, 0), "+": (PLUS, 0), "b_0": (B_DIGIT, 0), "=": (QUERY, 0), "q1": (QUERY, 1)}
for name, (kind, j) in SITES.items():
    for f in ("resid0", "resid1"):
        X, y, idx, amb = [], [], [], []
        for i, b in enumerate(BASES):
            t = T[b]
            rows = t.select(kind, j)
            X.append(t.get(f, rows))
            y.append(np.full(len(rows), i))
            idx.append(t.meta["idx"][rows])
            A, B = t.meta["a"][rows], t.meta["b"][rows]
            amb.append(np.array([set(np.base_repr(int(x), b) + np.base_repr(int(z), b)) <= {"0", "1"} for x, z in zip(A, B)]))
        X, y, idx, amb = map(np.concatenate, (X, y, idx, amb))
        te = idx % 5 == 0
        p = Ridge(X[~te])
        p.fit(onehot(y[~te], 4))
        pred = p.predict(X[te]).argmax(1).cpu().numpy()
        ok = pred == y[te]
        # ambiguous lines occur almost only in binary, so also report non-binary ones
        nb = amb[te] & (y[te] != 0)
        res["B"][f"{name}/{f}"] = {"acc": float(ok.mean()), "per_base": [float(ok[y[te] == i].mean()) for i in range(4)], "ambiguous_nonbinary_acc": float(ok[nb].mean()) if nb.sum() else None, "ambiguous_nonbinary_n": int(nb.sum())}
        print("B", name, f, res["B"][f"{name}/{f}"], flush=True)
save_json("s06_heads_base", res)
print("DONE")
