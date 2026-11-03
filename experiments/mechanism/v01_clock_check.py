"""V01 (stored + weights): is the "clock" real, or a by-product of tied embeddings?

With tied embeddings, if the final residual is close to the correct token's embedding, then logit(t | o) is about
alpha * E[t].E[o], which is symmetric whatever the arithmetic is. So symmetry and "peaked" spectra are not evidence alone.

A. Fit off-target logits with competing models (leave-one-output-out):
   circular offset g((t-o) mod b); signed difference h(t-o); distance h(|t-o|); embedding kernel a*E[t].E[o]+c.
B. Wrap-around test: circular neighbours that are far apart as tokens, e.g. (o=b-1, t=0), against true neighbours.
   A clock treats them alike. A distance-based kernel does not.
C. Periodicity across the wrap: fit each neuron's tuning on sums below the base, and predict sums at or above it, by period T.
D. Unselected alignment: over ALL active neurons, is the write vector's phase aligned with the sum tuning's phase? Null: shuffled neurons.
E. Does a population jump sit at s = base? Does the base vector move it?
F. Shape: fit each neuron over s with polynomial, parity, parity x polynomial and Fourier families.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table

torch.set_grad_enabled(False)
model = Model()
sd = model.sd
res = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}, "F": {}}
tok = np.array([model.vocab[c] for c in DIGITS])
tt = torch.as_tensor(tok, device=DEVICE)
WE = model.WE[tt].cpu().numpy()
Wo = sd["h.1.mlp.c_proj.weight"]
bo = sd["h.1.mlp.c_proj.bias"]
W_tok = (Wo @ model.WE[tt].T).cpu().numpy()


def load(b, k=0):
    t = Table(b)
    rows = t.select(QUERY, k)
    A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1)
    rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k
    ak, bk = (A // M) % b, (B // M) % b
    c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE)
    Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    lg = model.unembed(torch.as_tensor(t.get("resid1", rows), device=DEVICE)).cpu().numpy()[:, tok]
    return dict(a=ak, b=bk, c=c, s=ak + bk + c, pre=pre, mid=mid, lg=lg, post=gelu_new(pre).cpu().numpy())


def loo_r2(table, offs, mask):
    """Leave-one-output-out R^2 for a model that predicts entries by a grouping label."""
    n = table.shape[0]
    pred = np.zeros_like(table)
    for o in range(n):
        other = np.arange(n) != o
        for lab in np.unique(offs[o]):
            m = (offs[other] == lab) & mask[other]
            pred[o, offs[o] == lab] = table[other][m].mean() if m.any() else 0.0
    return float(1 - ((table[mask] - pred[mask]) ** 2).sum() / ((table[mask] - table[mask].mean()) ** 2).sum())


for b in (8, 10, 16):
    d = load(b)
    o = d["s"] % b
    Lg = d["lg"][:, :b]
    Lc = Lg - Lg.mean(1, keepdims=True)
    T = np.stack([Lc[o == v].mean(0) for v in range(b)])  # [output, token]
    tv = np.arange(b)
    diff = tv[None] - tv[:, None]
    mask = diff != 0
    circ = diff % b
    dist = np.abs(diff)
    circ_dist = np.minimum(circ, b - circ)
    E = WE[:b]
    G = E @ E.T
    x, y = G[mask], T[mask]
    slope, inter = np.polyfit(x, y, 1)
    # Leave-one-output-out for the kernel
    pred = np.zeros_like(T)
    for v in range(b):
        other = (np.arange(b) != v)
        mk = mask.copy()
        mk[v] = False
        s2, i2 = np.polyfit(G[mk], T[mk], 1)
        pred[v] = s2 * G[v] + i2
    kernel = float(1 - ((T[mask] - pred[mask]) ** 2).sum() / ((T[mask] - T[mask].mean()) ** 2).sum())
    res["A"][b] = {
        "circular offset": loo_r2(T, circ, mask),
        "circular distance": loo_r2(T, circ_dist, mask),
        "signed difference": loo_r2(T, diff, mask),
        "token distance": loo_r2(T, dist, mask),
        "embedding kernel": kernel,
        "parity of difference only": loo_r2(T, diff % 2, mask),
        "corr_logits_vs_kernel": float(np.corrcoef(x, y)[0, 1]),
    }
    print("A", b, {k: round(v, 3) for k, v in res["A"][b].items()}, flush=True)
    # B wrap-around
    rows = {}
    for cd in (1, 2, 3):
        near = (circ_dist == cd) & (dist == cd)
        far = (circ_dist == cd) & (dist == b - cd)
        rows[cd] = {"non_wrap_mean": float(T[near].mean()), "wrap_mean": float(T[far].mean()), "gram_non_wrap": float(G[near].mean()), "gram_wrap": float(G[far].mean())}
    other = T[mask].mean()
    res["B"][b] = {"circular_distance": rows, "offtarget_mean": float(other), "offtarget_sd": float(T[mask].std())}
    print("B", b, {k: {a: round(c, 2) for a, c in v.items()} for k, v in rows.items()}, "offtarget sd", round(float(T[mask].std()), 2), flush=True)

    # C periodicity across wrap, on all neurons
    ns = 2 * b - 1
    Xs = np.stack([d["post"][d["s"] == v].mean(0) if (d["s"] == v).any() else d["post"].mean(0) for v in range(ns)])
    w = np.array([(d["s"] == v).sum() for v in range(ns)], float)
    var = Xs.var(0)
    act = var > 0.01 * var.max()
    Y = Xs[:, act]
    sv = np.arange(ns, dtype=float)
    low = sv < b
    scan = {}
    for Tp in np.r_[np.arange(2, 21), [24, 32]]:
        def basis(z):
            return np.c_[np.ones_like(z), z / ns, np.cos(2 * np.pi * z / Tp), np.sin(2 * np.pi * z / Tp)]

        B_ = basis(sv)
        coef = np.linalg.lstsq(B_[low], Y[low], rcond=None)[0]
        p = B_[~low] @ coef
        scan[int(Tp)] = float(1 - ((Y[~low] - p) ** 2).sum() / ((Y[~low] - Y[~low].mean(0)) ** 2).sum())
    res["C"][b] = {"cross_wrap_r2_by_period": scan}
    print("C", b, "predict sums >= base from sums < base, by period:", {k: round(v, 2) for k, v in scan.items() if k in (2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20)}, flush=True)

    # D unselected alignment over all active neurons, on s mod b tuning
    Tm = np.stack([d["post"][o == v].mean(0) for v in range(b)])
    Tm = Tm - Tm.mean(0)
    FT = np.fft.fft(Tm, axis=0)
    wr = W_tok[:, :b] - W_tok[:, :b].mean(1, keepdims=True)
    FW = np.fft.fft(wr, axis=1)
    rng = np.random.default_rng(0)
    out = {}
    for f in range(1, b // 2 + 1):
        a_ = FT[f]
        c_ = FW[:, f]
        contrib = float(np.real(a_ * np.conj(c_)).sum())
        amp = float((np.abs(a_) * np.abs(c_)).sum())
        nulls = [float(np.real(a_ * np.conj(c_[rng.permutation(256)])).sum()) for _ in range(500)]
        out[f] = {"coherence": contrib / amp, "z": (contrib - np.mean(nulls)) / (np.std(nulls) + 1e-9), "share_tuning_power": float((np.abs(a_) ** 2).sum() / (np.abs(FT[1:]) ** 2).sum())}
    res["D"][b] = out
    print("D", b, {f: (round(v["coherence"], 2), round(v["z"], 1)) for f, v in out.items()}, flush=True)

    # E population jump
    Tn = Xs - Xs.mean(0)
    jumps = {}
    for v in range(0, ns - 2):
        jumps[v + 1] = float(np.linalg.norm(Xs[v + 2] - Xs[v]))  # same-parity step centred at v+1
    top = sorted(jumps, key=lambda z: -jumps[z])[:4]
    res["E"][b] = {"same_parity_step_by_centre": jumps, "largest_at": top}
    print("E", b, "largest same-parity population steps centred at", top, "| at base:", round(jumps.get(b, 0), 2), "median:", round(float(np.median(list(jumps.values()))), 2), flush=True)

    # F shapes
    fam = {}
    par = (sv % 2 == 0).astype(float)
    z = (sv - sv.mean()) / sv.std()
    fams = {
        "constant": np.c_[np.ones(ns)],
        "cubic in s": np.c_[np.ones(ns), z, z**2, z**3],
        "parity": np.c_[np.ones(ns), par],
        "cubic + parity": np.c_[np.ones(ns), z, z**2, z**3, par],
        "cubic + parity x cubic": np.c_[np.ones(ns), z, z**2, z**3, par, par * z, par * z**2, par * z**3],
        "linear + cos,sin at period base": np.c_[np.ones(ns), z, np.cos(2 * np.pi * sv / b), np.sin(2 * np.pi * sv / b)],
        "linear + periods base, base/2": np.c_[np.ones(ns), z, np.cos(2 * np.pi * sv / b), np.sin(2 * np.pi * sv / b), np.cos(4 * np.pi * sv / b), np.sin(4 * np.pi * sv / b)],
    }
    for nm, B_ in fams.items():
        # leave-one-sum-out
        pred = np.zeros_like(Y)
        for i in range(ns):
            m = np.arange(ns) != i
            pred[i] = B_[i] @ np.linalg.lstsq(B_[m], Y[m], rcond=None)[0]
        fam[nm] = float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())
    res["F"][b] = fam
    print("F", b, "leave-one-sum-out R2:", {k: round(v, 3) for k, v in fam.items()}, flush=True)
save_json("v01_clock_check", res)
print("DONE")
