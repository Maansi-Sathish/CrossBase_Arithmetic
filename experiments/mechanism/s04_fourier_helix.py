"""E5: Fourier probes, conditional spectra and helix fits, in the style of Kantamneni & Tegmark.

Three views of "is there periodic structure over the integer?":
  A. Fourier probe spectrum: R^2 on HELD-OUT integers predicting (cos 2*pi*n/T, sin 2*pi*n/T) from
     activations, for many periods T. The input baseline shows which periods come for free.
  B. Conditional-mean spectrum: Lomb-Scargle-style power of the mean activation by integer, with a
     label-permutation null, so peaks don't depend on a probe.
  C. Helix fit: regress activations on a basis of [n, cos, sin at a set of periods], testing on held-out
     integers. Compares own-base periods, decimal periods, binary periods, random periods and a plain
     additive digit model.
A pivot-base mechanism predicts peaks at a foreign period, e.g. T = 10 in hex prompts.
A digit-local mechanism predicts peaks only at powers of the prompt's own base.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, DEVICE, save_json
from probe_lib import A_DIGIT, B_DIGIT, QUERY, Ridge, Table, input_onehot, integer_split

periods = np.unique(
    np.r_[
        np.arange(2, 65, dtype=float),
        np.array([100, 125, 128, 200, 250, 256, 343, 500, 512, 625, 729, 1000, 1024, 2000, 2048], dtype=float),
        np.geomspace(64, 2048, 60),
    ]
)
NAMED = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 20, 32, 64, 100, 128, 256, 512, 1000, 1024]
tables = {b: Table(b) for b in BASES}
SITES = [("a_0", A_DIGIT, 0, "a"), ("b_0", B_DIGIT, 0, "b"), ("=", QUERY, 0, "a"), ("=", QUERY, 0, "b"),
         ("=", QUERY, 0, "a+b"), ("q1", QUERY, 1, "a+b"), ("q2", QUERY, 2, "a+b")]
res: dict = {"periods": periods, "named": NAMED, "A": {}, "B": {}, "C": {}}


def fourier_targets(n: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """[N, 2*len(ts)] cos/sin targets."""
    ang = 2 * np.pi * n[:, None] / ts[None, :]
    return np.concatenate([np.cos(ang), np.sin(ang)], axis=1)


@torch.no_grad()
def spectrum(xtr: np.ndarray, ntr: np.ndarray, xte: np.ndarray, nte: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Held-out R^2 of the 2-D Fourier target at each period."""
    p = Ridge(xtr)
    Y = fourier_targets(ntr.astype(np.float64), ts)
    p.fit(Y)
    pred = p.predict(xte)
    Yt = torch.as_tensor(fourier_targets(nte.astype(np.float64), ts), device=DEVICE, dtype=torch.float64)
    m = len(ts)
    sse = ((pred - Yt) ** 2)
    sst = (Yt - Yt.mean(0)) ** 2
    return (1 - (sse[:, :m].sum(0) + sse[:, m:].sum(0)) / (sst[:, :m].sum(0) + sst[:, m:].sum(0)).clamp_min(1e-9)).cpu().numpy()


for base in BASES:
    t = tables[base]
    res["A"][base] = {}
    for site, kind, j, tname in SITES:
        rows = t.select(kind, j)
        a, b = t.meta["a"][rows], t.meta["b"][rows]
        n = {"a": a, "b": b, "a+b": a + b}[tname]
        tr, te = integer_split(n, seed=2)
        out = {}
        feats = {
            "resid0": t.get("resid0", rows),
            "resid1": t.get("resid1", rows),
            "mlp1": t.get("mlp1", rows),
            "input_onehot": input_onehot(a, b, base),
        }
        for f, X in feats.items():
            out[f] = spectrum(X[tr], n[tr], X[te], n[te], periods)
        res["A"][base][f"{site}/{tname}"] = out
        at = {T: float(out["resid1"][np.argmin(np.abs(periods - T))]) for T in NAMED}
        at_in = {T: float(out["input_onehot"][np.argmin(np.abs(periods - T))]) for T in NAMED}
        print(f"A base {base} {site}/{tname} resid1:", {T: round(v, 2) for T, v in at.items() if v > 0.05}, "| input:", {T: round(v, 2) for T, v in at_in.items() if v > 0.05}, flush=True)


# ---------------------------------------------------------------------------- B: conditional means
def cond_mean(X: np.ndarray, n: np.ndarray, nmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean activation per integer value; returns (values present, means)."""
    vals = np.unique(n[n <= nmax])
    out = np.stack([X[n == v].mean(0) for v in vals])
    return vals, out


def power(vals: np.ndarray, means: np.ndarray, ts: np.ndarray) -> np.ndarray:
    """Normalised periodogram at arbitrary periods, after removing a cubic trend (magnitude)."""
    V = np.vander((vals - vals.mean()) / vals.std(), 4)
    y = means - V @ np.linalg.lstsq(V, means, rcond=None)[0]
    total = (y**2).sum()
    out = np.zeros(len(ts))
    for i, T in enumerate(ts):
        ang = 2 * np.pi * vals / T
        c, s = np.cos(ang), np.sin(ang)
        c -= c.mean()
        s -= s.mean()
        out[i] = ((c @ y) ** 2).sum() / (c @ c + 1e-9) + ((s @ y) ** 2).sum() / (s @ s + 1e-9)
    return out / total


test_ts = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 20, 25, 32, 50, 64, 100, 128, 256], dtype=float)
res["B"]["periods"] = test_ts
for base in BASES:
    t = tables[base]
    res["B"][base] = {}
    for site, kind, j, tname in [("a_0", A_DIGIT, 0, "a"), ("=", QUERY, 0, "a"), ("=", QUERY, 0, "a+b"), ("q1", QUERY, 1, "a+b")]:
        rows = t.select(kind, j)
        a, b = t.meta["a"][rows], t.meta["b"][rows]
        n = {"a": a, "a+b": a + b}[tname]
        X = t.get("resid1", rows)
        X = (X - X.mean(0)) @ np.linalg.svd(X - X.mean(0), full_matrices=False)[2][:20].T
        nmax = 999 if tname == "a" else 1700
        vals, m = cond_mean(X, n, nmax)
        obs = power(vals, m, test_ts)
        rng = np.random.default_rng(0)
        null = np.stack([power(vals, cond_mean(X, rng.permutation(n), nmax)[1], test_ts) for _ in range(30)])
        z = (obs - null.mean(0)) / (null.std(0) + 1e-12)
        res["B"][base][f"{site}/{tname}"] = {"power": obs, "null_mean": null.mean(0), "null_sd": null.std(0), "z": z}
        top = np.argsort(-z)[:6]
        print(f"B base {base} {site}/{tname} top z:", [(int(test_ts[i]), round(float(z[i]), 1), round(float(obs[i]), 3)) for i in top], flush=True)


# ---------------------------------------------------------------------------- C: helix fits
def helix_basis(n: np.ndarray, ts: list[float], linear: bool = True) -> np.ndarray:
    """[n, cos, sin at each period]."""
    cols = [n[:, None] / 1000.0] if linear else []
    for T in ts:
        ang = 2 * np.pi * n / T
        cols += [np.cos(ang)[:, None], np.sin(ang)[:, None]]
    return np.concatenate(cols, axis=1) if cols else np.zeros((len(n), 1))


def forward_r2(B_tr: np.ndarray, Y_tr: np.ndarray, B_te: np.ndarray, Y_te: np.ndarray) -> float:
    """Held-out R^2 of a basis predicting activation PCs."""
    p = Ridge(B_tr)
    p.fit(Y_tr, lams=(1e-3,))
    pred = p.predict(B_te).cpu().numpy()
    return float(1 - ((pred - Y_te) ** 2).sum() / ((Y_te - Y_te.mean(0)) ** 2).sum())


def powers(b: int, maxv: int = 1100) -> list[float]:
    """Powers of b up to the operand range."""
    out, x = [], b
    while x <= maxv:
        out.append(float(x))
        x *= b
    return out


rng = np.random.default_rng(1)
for base in BASES:
    t = tables[base]
    res["C"][base] = {}
    for site, kind, j, tname in [("a_0", A_DIGIT, 0, "a"), ("=", QUERY, 0, "a"), ("=", QUERY, 0, "a+b")]:
        for layer in ["resid0", "resid1"]:
            rows = t.select(kind, j)
            a, b = t.meta["a"][rows], t.meta["b"][rows]
            n = {"a": a, "a+b": a + b}[tname].astype(np.float64)
            X = t.get(layer, rows)
            mu = X.mean(0)
            U, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
            K = 16
            Y = (X - mu) @ Vt[:K].T
            tr, te = integer_split(n, seed=3)
            own = powers(base)
            cand = {
                "linear only": helix_basis(n, []),
                "own-base periods": helix_basis(n, own),
                "decimal periods": helix_basis(n, powers(10)),
                "binary periods": helix_basis(n, powers(2)),
                "own + decimal": helix_basis(n, sorted(set(own + powers(10)))),
            }
            out = {name: forward_r2(B[tr], Y[tr], B[te], Y[te]) for name, B in cand.items()}
            # random periods, same count as own-base, as the null
            nulls = []
            for _ in range(30):
                ts = list(np.exp(rng.uniform(np.log(2.5), np.log(1100), len(own))))
                B = helix_basis(n, ts)
                nulls.append(forward_r2(B[tr], Y[tr], B[te], Y[te]))
            out["random periods (mean)"] = float(np.mean(nulls))
            out["random periods (95th pct)"] = float(np.quantile(nulls, 0.95))
            # Surface hypothesis: additive one-hot digits, no periodic assumptions
            if tname == "a":
                D = input_onehot(a, np.zeros_like(a), base)
            else:
                D = input_onehot(a, b, base)
            out["additive digit one-hot"] = forward_r2(D[tr], Y[tr], D[te], Y[te])
            full_cov = float((S[:K] ** 2).sum() / (S**2).sum())
            out["pc_variance_covered"] = full_cov
            res["C"][base][f"{site}/{tname}/{layer}"] = out
            print(f"C base {base} {site}/{tname}/{layer}:", {k: round(v, 3) for k, v in out.items()}, flush=True)

save_json("s04_fourier_helix", res)
print("DONE")
