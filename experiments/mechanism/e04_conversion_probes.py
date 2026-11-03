"""E4: does the model hold the integer in any form other than its own base's digits?

The key decoding targets are residues n mod M for M that do NOT follow from the digit tokens in the
prompt's base. Example: n mod 10 in a hex prompt, or n mod 7 in a decimal prompt. Reading those needs
a real base conversion. Note that every base here is even, and 2, 8 and 16 are powers of 2, so:
  - parity is trivial everywhere; n mod 2^m is trivial for bases 2, 8 and 16 once enough low digits are read;
  - the only "hard" targets are odd moduli and base-10-specific residues.

Run: probes at each semantic position for {a, b, a+b}, linear and MLP, with controls.
Also runs integer-value probes, per base and transferred across bases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from core import BASES, save_json
from probe_lib import (
    A_DIGIT,
    B_DIGIT,
    QUERY,
    Ridge,
    Table,
    classify,
    input_onehot,
    integer_split,
    mlp_classify,
    r2,
    random_features,
)

MODULI = [2, 3, 4, 5, 7, 8, 9, 10, 16]
POSITIONS = {
    "a_0": (A_DIGIT, 0, ["a"]),
    "b_0": (B_DIGIT, 0, ["a", "b"]),
    "=": (QUERY, 0, ["a", "b", "a+b"]),
    "q1": (QUERY, 1, ["a", "b", "a+b"]),
    "q2": (QUERY, 2, ["a", "b", "a+b"]),
}
FEATURES = ["resid0", "resid1", "mlp0", "mlp1"]
res: dict = {"moduli": MODULI, "linear": {}, "mlp": {}, "value": {}}
tables = {b: Table(b) for b in BASES}


def target_values(name: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The integer a probe is asked about."""
    return {"a": a, "b": b, "a+b": a + b}[name]


cache_feats: dict = {}
for base in BASES:
    t = tables[base]
    res["linear"][base] = {}
    for pname, (kind, j, targets) in POSITIONS.items():
        rows = t.select(kind, j)
        idx = t.meta["idx"][rows]
        a, b = t.meta["a"][rows], t.meta["b"][rows]
        feats = {f: t.get(f, rows) for f in FEATURES}
        feats["input_onehot"] = input_onehot(a, b, base)
        probs = t.problems(idx)
        feats["random_init_resid1"] = random_features(base, probs, kind, j)
        cache_feats[(base, pname)] = (feats, a, b)
        res["linear"][base][pname] = {}
        for tname in targets:
            n = target_values(tname, a, b)
            tr, te = integer_split(n, seed=0)
            res["linear"][base][pname][tname] = {}
            for fname, X in feats.items():
                row = {}
                for M in MODULI:
                    y = n % M
                    row[M] = classify(X[tr], y[tr], X[te], y[te], M)
                res["linear"][base][pname][tname][fname] = row
        line = {
            M: (
                round(res["linear"][base][pname][targets[-1]]["resid1"][M]["acc"], 3),
                round(res["linear"][base][pname][targets[-1]]["input_onehot"][M]["acc"], 3),
                round(res["linear"][base][pname][targets[-1]]["random_init_resid1"][M]["acc"], 3),
            )
            for M in MODULI
        }
        print(f"base {base} {pname} target {targets[-1]} (resid1, input, randinit):", line, flush=True)

# ---- non-linear probes on the hard moduli, at the positions that have seen the full problem -------
print("MLP probes")
for base in BASES:
    res["mlp"][base] = {}
    for pname, tname in [("=", "a+b"), ("=", "a"), ("q1", "a+b")]:
        feats, a, b = cache_feats[(base, pname)]
        n = target_values(tname, a, b)
        tr, te = integer_split(n, seed=0)
        res["mlp"][base][f"{pname}/{tname}"] = {}
        for fname in ["resid1", "mlp1", "input_onehot", "random_init_resid1"]:
            X = feats[fname]
            r = {}
            for M in [3, 5, 7, 9, 10, 16]:
                y = n % M
                r[M] = mlp_classify(X[tr], y[tr], X[te], y[te], M)
            res["mlp"][base][f"{pname}/{tname}"][fname] = r
            print(base, pname, tname, fname, {M: round(v["acc"], 3) for M, v in r.items()}, flush=True)

# ---- integer value and log value: per base and transferred -------------------------------------
print("Value probes")
for pname, tname in [("a_0", "a"), ("=", "a"), ("=", "b"), ("=", "a+b"), ("q1", "a+b")]:
    res["value"][f"{pname}/{tname}"] = {}
    for rng_name, lo, hi in [("all", 0, 10000), ("fixed_length_512_999", 512, 999)]:
        if tname == "a+b" and rng_name != "all":
            lo, hi = 1024, 1998
        for fname in ["resid1", "input_onehot", "random_init_resid1"]:
            mat = np.zeros((4, 4))
            for i, bx in enumerate(BASES):
                fx, ax, bbx = cache_feats[(bx, pname)]
                nx = target_values(tname, ax, bbx)
                mask = (nx >= lo) & (nx <= hi)
                tr, te = integer_split(nx, seed=1)
                if fname == "input_onehot":
                    # the input encoding is base-specific and has different dimensions per base
                    p = Ridge(fx[fname][tr & mask])
                    p.fit(nx[tr & mask].astype(np.float64) / 1000)
                    mat[i, i] = r2(p.predict(fx[fname][te & mask]), nx[te & mask].astype(np.float64) / 1000)
                    for jj in range(4):
                        if jj != i:
                            mat[i, jj] = np.nan
                    continue
                p = Ridge(fx[fname][tr & mask])
                p.fit(nx[tr & mask].astype(np.float64) / 1000)
                for jj, by in enumerate(BASES):
                    fy, ay, bby = cache_feats[(by, pname)]
                    ny = target_values(tname, ay, bby)
                    _, tey = integer_split(ny, seed=1)  # same integers held out in every base
                    masky = (ny >= lo) & (ny <= hi)
                    # raw transfer: the probe's intercept and scale are fixed from the source base
                    mat[i, jj] = r2(p.predict(fy[fname][tey & masky]), ny[tey & masky].astype(np.float64) / 1000)
            res["value"][f"{pname}/{tname}"].setdefault(rng_name, {})[fname] = mat
            print(pname, tname, rng_name, fname, "\n", np.round(mat, 2), flush=True)

save_json("e04_conversion_probes", res)
