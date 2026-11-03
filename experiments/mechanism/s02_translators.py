"""E10: are there translator neurons? Neuron-level input/output anatomy across bases.

A neuron's scalar input and output are tied by a fixed curve (GELU), so comparing those says nothing. The
meaningful pair is WHEN it fires (its tuning) and WHAT it writes (its output weights). Tests:

  A. Foreign tuning. Within a cell of fixed own-base digits (a_k, b_k, c_k), does the neuron still vary with
     a digit of the same integer in ANOTHER base? Only a neuron holding a converted number can do that.
     Permutation within cells gives a family-wise null over neurons. Positive control: own-base higher
     digits. (Stored activations only; no forward passes, so no untrained-model control.)
  B. Read/write table. For L1 neurons: mean firing by raw sum s, times the logit each neuron writes for
     each token. Does a neuron's vote follow its own base's rule or another base's, on conflict values of s?
  C. Wrap index: eta^2(activation ~ o) / eta^2(activation ~ s). Near 1 means the neuron only sees s through
     the wrapped digit o, a post-modulus code. Lower means it still separates s from s + base.
  D. Weight anatomy: for each neuron, the digit pattern of its input and output weights, in embedding space.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, DEVICE, DIGITS, Model, save_json, schoolbook
from probe_lib import QUERY, Table

model = Model()
tables = {b: Table(b) for b in BASES}
res: dict = {"A": {}, "B": {}, "C": {}, "D": {}}


def digit(n: np.ndarray, base: int, k: int) -> np.ndarray:
    """k-th LSD digit of n in a base."""
    return (n // base**k) % base


def own_labels(a: np.ndarray, b: np.ndarray, base: int, k: int) -> dict:
    """Schoolbook labels at digit k."""
    out = {"a": digit(a, base, k), "b": digit(b, base, k)}
    # carry into digit k
    m = base**k
    c = ((a % m) + (b % m)) >= m if k > 0 else np.zeros(len(a), dtype=bool)
    out["c"] = c.astype(int)
    out["s"] = out["a"] + out["b"] + out["c"]
    out["o"] = out["s"] % base
    return out


@torch.no_grad()
def eta2_gpu(x: torch.Tensor, labels: torch.Tensor, n_lab: int) -> torch.Tensor:
    """Eta^2 for each column of x, against integer labels. x must be centered per column."""
    one = torch.nn.functional.one_hot(labels, n_lab).float()
    cnt = one.sum(0).clamp_min(1)
    means = (one.T @ x) / cnt[:, None]
    between = (means**2 * cnt[:, None]).sum(0)
    total = (x**2).sum(0).clamp_min(1e-8)
    return between / total


def cell_residual(x: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Remove the mean of every own-digit cell, column by column."""
    out = x.copy()
    for c in np.unique(cell):
        m = cell == c
        out[m] -= out[m].mean(0)
    return out


def within_cell_perm(labels: np.ndarray, cell: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Shuffle labels within each cell: preserves anything explained by the own-base digits."""
    first = np.lexsort((rng.random(len(cell)), cell))
    second = np.lexsort((rng.random(len(cell)), cell))
    out = labels.copy()
    out[first] = labels[second]
    return out


# ------------------------------------------------------------------------------------------- A
# First design (fixing only digit k) was confounded: e.g. parity of the sum IS the digit already written.
# Corrected design: remove an additive model of ALL own-base operand digits and all answer digits already
# written, then test targets that require conversion (odd residues), with effect-size reporting.
from probe_lib import input_onehot

PERMS = 100
MIN_SHARE = 0.01
for base in BASES:
    t = tables[base]
    res["A"][base] = {}
    for k in [0, 1, 2]:
        rows = t.select(QUERY, k)
        a0, b0 = t.meta["a"][rows], t.meta["b"][rows]
        keep = (a0 + b0) >= base**k
        rows, a, b = rows[keep], a0[keep], b0[keep]
        own = own_labels(a, b, base, k)
        cell = own["a"] * 400 + own["b"] * 20 + own["c"]
        design = [input_onehot(a, b, base)]
        for j in range(k):  # answer digits already written
            design.append(np.eye(base, dtype=np.float32)[digit(a + b, base, j)])
        D = np.concatenate(design, axis=1)
        D = np.c_[D, np.ones(len(D), dtype=np.float32)]
        targets = {f"{nm} mod {M}": n % M for M in [3, 5, 7, 9] for nm, n in (("a", a), ("b", b), ("a+b", a + b)) if not (M == 5 and base == 10)}
        ctrl = {}
        if k + 1 < 11:
            ctrl[f"own: next sum digit"] = digit(a + b, base, k + 1)
        out = {}
        rng = np.random.default_rng(base * 10 + k)
        for fname in ["mlp0", "mlp1"]:
            X = t.get(fname, rows).astype(np.float64)
            tv = X.var(0)
            coef, *_ = np.linalg.lstsq(D, X, rcond=None)
            R = X - D @ coef
            additive_r2 = float(1 - R.var(0).sum() / tv.sum())
            cell_R = cell_residual(X, cell)
            Rg = torch.as_tensor(R, device=DEVICE, dtype=torch.float32)
            Rg = Rg - Rg.mean(0)
            frow = {"additive_own_digits_r2": additive_r2, "cell_digit_k_r2": float(1 - cell_R.var(0).sum() / tv.sum()), "targets": {}, "controls": {}}
            for group, tg in (("targets", targets), ("controls", ctrl)):
                for tname, lab in tg.items():
                    nlab = int(lab.max()) + 1
                    if nlab < 2:
                        continue
                    obs = eta2_gpu(Rg, torch.as_tensor(lab, dtype=torch.long, device=DEVICE), nlab).cpu().numpy()
                    maxima = [float(eta2_gpu(Rg, torch.as_tensor(within_cell_perm(lab, cell, rng), dtype=torch.long, device=DEVICE), nlab).max()) for _ in range(PERMS)]
                    thr = float(np.quantile(maxima, 0.95))
                    share = obs * R.var(0) / (tv + 1e-9)
                    frow[group][tname] = {"n_sig": int((obs > thr).sum()), "n_sig_and_large": int(((obs > thr) & (share >= MIN_SHARE)).sum()), "max_share_total_var": float(share.max()), "max_partial_eta2": float(obs.max()), "null95": thr, "top": int(share.argmax())}
            out[fname] = frow
            print(f"A b{base} k{k} {fname}: digit-k cell R2 {frow['cell_digit_k_r2']:.3f}, all-own-digits additive R2 {additive_r2:.3f} | odd residues: sig {sum(v['n_sig'] for v in frow['targets'].values())}, sig+>=1% var {sum(v['n_sig_and_large'] for v in frow['targets'].values())}, max share {max(v['max_share_total_var'] for v in frow['targets'].values()):.4f} | ctrl {[(v['n_sig'], v['n_sig_and_large'], round(v['max_share_total_var'], 3)) for v in frow['controls'].values()]}", flush=True)
        res["A"][base][k] = out
    save_json("s02_translators", res)

# ------------------------------------------------------------------------------------------- B, C
digit_ids = torch.tensor([model.vocab[c] for c in DIGITS], device=DEVICE)
Wout = {L: model.sd[f"h.{L}.mlp.c_proj.weight"] for L in range(2)}  # [256, 64]
Win = {L: model.sd[f"h.{L}.mlp.c_fc.weight"] for L in range(2)}  # [64, 256]
write = {L: (Wout[L] @ model.WE[digit_ids].T).cpu().numpy() for L in range(2)}  # [neurons, 16 digit tokens]
write_centered = {L: write[L] - write[L].mean(1, keepdims=True) for L in range(2)}
res["D"]["write_top_token_counts"] = {L: np.bincount(write[L].argmax(1), minlength=16) for L in range(2)}

tuning_s: dict = {}
for base in BASES:
    res["B"][base] = {}
    res["C"][base] = {}
    t = tables[base]
    for L in range(2):
        for k in [0, 1, 2]:
            rows = t.select(QUERY, k)
            a0, b0 = t.meta["a"][rows], t.meta["b"][rows]
            keep = (a0 + b0) >= base**k
            rows, a, b = rows[keep], a0[keep], b0[keep]
            own = own_labels(a, b, base, k)
            X = t.get(f"mlp{L}", rows)
            Xc = torch.as_tensor(X - X.mean(0), device=DEVICE)
            s = own["s"]
            o = own["o"]
            eta_s = eta2_gpu(Xc, torch.as_tensor(s, device=DEVICE, dtype=torch.long), int(s.max()) + 1).cpu().numpy()
            eta_o = eta2_gpu(Xc, torch.as_tensor(o, device=DEVICE, dtype=torch.long), base).cpu().numpy()
            var = X.var(0)
            weights = var * eta_s
            wrap_idx = eta_o / np.maximum(eta_s, 1e-6)
            res["C"][base][f"L{L}/k{k}"] = {
                "wrap_index_weighted": float((wrap_idx * weights).sum() / (weights.sum() + 1e-9)),
                "eta_s_weighted": float((eta_s * var).sum() / var.sum()),
                "eta_o_weighted": float((eta_o * var).sum() / var.sum()),
            }
            # mean firing by s
            max_s = 2 * base
            tun = np.full((max_s, X.shape[1]), np.nan)
            for v in range(max_s):
                m = s == v
                if m.sum() >= 5:
                    tun[v] = X[m].mean(0)
            tuning_s[(base, L, k)] = tun
            # read/write table: mean written logit for each token, as a function of s
            valid = ~np.isnan(tun[:, 0])
            table = np.full((max_s, 16), np.nan)
            table[valid] = (tun[valid] - X.mean(0)) @ write_centered[L]
            # score each neuron's vote: corr across s of (tuning, its written logit for the token the rule needs)
            scores = {"own": [], **{f"b{T}": [] for T in BASES if T != base}}
            ss = np.where(valid)[0]
            for n in np.where(var > 1e-3)[0]:
                x = tun[ss, n] - tun[ss, n].mean()
                if x.std() < 1e-6:
                    continue
                for tag, mod in [("own", base), *[(f"b{T}", T) for T in BASES if T != base]]:
                    # conflict values only, where the rules disagree. Own-rule scored on the same subset.
                    other = base if tag == "own" else mod
                    y = write_centered[L][n, ss % mod]
                    if y.std() < 1e-6:
                        continue
                    scores[tag].append(float(np.corrcoef(x, y)[0, 1] * x.std()))
            res["B"][base][f"L{L}/k{k}"] = {
                "readout_table": table,
                "mean_weighted_vote": {tag: float(np.mean(v)) for tag, v in scores.items()},
                "table_top_is_rule": {
                    tag: float(np.mean([table[v].argmax() == v % mod for v in ss]))
                    for tag, mod in [("own", base), *[(f"b{T}", T) for T in BASES if T != base]]
                },
            }
            print(f"B/C b{base} L{L} k{k}: wrap idx {res['C'][base][f'L{L}/k{k}']['wrap_index_weighted']:.2f}, eta_s {res['C'][base][f'L{L}/k{k}']['eta_s_weighted']:.3f}, eta_o {res['C'][base][f'L{L}/k{k}']['eta_o_weighted']:.3f}, vote {({a2: round(b2, 3) for a2, b2 in res['B'][base][f'L{L}/k{k}']['mean_weighted_vote'].items()})}, top-token={res['B'][base][f'L{L}/k{k}']['table_top_is_rule']}", flush=True)

# ------------------------------------------------------------------------------------------- D
# Weight anatomy: what does each neuron read from a digit's embedding, and what does it write?
E = model.WE[digit_ids]  # [16, 64]
read = {L: (E @ Win[L]).cpu().numpy().T for L in range(2)}  # [neurons, 16]
res["D"]["read_write_corr_all_neurons"] = {}
for L in range(2):
    r = read[L] - read[L].mean(1, keepdims=True)
    w = write_centered[L]
    corr_same = (r * w).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(w, axis=1) + 1e-9)
    # shift map: reads digit d, writes digit d + shift. Compare against a permutation of tokens.
    shifts = {}
    for sh in range(-3, 4):
        w2 = np.roll(w, sh, axis=1)
        shifts[sh] = float(((r * w2).sum(1) / (np.linalg.norm(r, axis=1) * np.linalg.norm(w2, axis=1) + 1e-9)).mean())
    res["D"]["read_write_corr_all_neurons"][L] = {"same_token_mean": float(corr_same.mean()), "by_shift": shifts}
    print("D L", L, "read-write same token corr", round(float(corr_same.mean()), 3), {k2: round(v2, 3) for k2, v2 in shifts.items()}, flush=True)
    # Fourier content of write vectors over 16 digit tokens
    F = np.abs(np.fft.rfft(w, axis=1)) ** 2
    res["D"][f"write_fourier_L{L}"] = (F[:, 1:].sum(0) / F[:, 1:].sum()).tolist()
    F2 = np.abs(np.fft.rfft(r, axis=1)) ** 2
    res["D"][f"read_fourier_L{L}"] = (F2[:, 1:].sum(0) / F2[:, 1:].sum()).tolist()
    print("  write fourier (T=16/f)", np.round(res["D"][f"write_fourier_L{L}"], 3), "\n  read fourier", np.round(res["D"][f"read_fourier_L{L}"], 3), flush=True)

# Virtual L0 -> L1 neuron weights (ignores LayerNorm scale): the structural "wiring" between neurons
v = (Wout[0] @ Win[1]).cpu().numpy()
np.save(Path(__file__).resolve().parent / "data" / "s02_virtual_L0_L1.npy", v)
np.savez(
    Path(__file__).resolve().parent / "data" / "s02_tuning.npz",
    **{f"tun_b{b}_L{L}_k{k}": t for (b, L, k), t in tuning_s.items()},
    write0=write[0],
    write1=write[1],
)
save_json("s02_translators", res)
print("DONE")
