"""T08 (stored + weights): a compact formula for layer 1 neurons.

Replace every neuron's activity with a small formula of the sum s and the digits' parities, fitted on half the prompts,
then run the readout on the other half. Candidates, per neuron and base, at digit k:
  sum table           : one value per s (2b-1 numbers) - the reference
  bump                : const + amp * Gaussian(s; centre, width)
  bump + parity gate  : bump + separate values for (odd,odd) and (even,even) pairs
  two bumps + parity  : for neurons with two firing zones
  parity gate only    : const + (odd,odd) and (even,even) values
Reports digit accuracy and parameter counts.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; a, c_ = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    return dict(a=a, b=c_, c=c, s=a + c_ + c, post=t.get("mlp1", rows), resid=t.get("resid1", rows), idx=t.meta["idx"][rows])
def score(d, Xhat, te):
    r = torch.as_tensor(d["resid"][te], device=DEVICE) + (torch.as_tensor(Xhat[te], device=DEVICE, dtype=torch.float32) - torch.as_tensor(d["post"][te], device=DEVICE)) @ Wo
    return float((model.unembed(r).argmax(-1).cpu().numpy() == tok[d["s"][te] % d["base"]]).mean())
def fit_bumps(y, s, nb):
    """Best 0/1/2 Gaussian bumps over a grid of centres and widths, least squares on the rest."""
    best = (np.inf, None)
    cs = np.arange(0, s.max() + 1, 1.0); ws = (0.7, 1.2, 2.0, 3.5, 6.0)
    grid = [(c, w) for c in cs for w in ws]
    def design(params):
        return np.column_stack([np.ones(len(s))] + [np.exp(-0.5 * ((s - c) / w) ** 2) for c, w in params])
    if nb == 1:
        for p in grid:
            X = design([p]); co, *_ = np.linalg.lstsq(X, y, rcond=None); e = ((y - X @ co) ** 2).sum()
            if e < best[0]: best = (e, [p])
    else:
        # greedy: best single, then best second given the first
        one = fit_bumps(y, s, 1)[1]
        for p in grid:
            X = design(one + [p]); co, *_ = np.linalg.lstsq(X, y, rcond=None); e = ((y - X @ co) ** 2).sum()
            if e < best[0]: best = (e, one + [p])
    return best[0], best[1]
for b in (8, 10, 16):
    for k in (0, 1):
        d = load(b, k); d["base"] = b; tr = d["idx"] % 2 == 0; te = ~tr; s = d["s"]; X = d["post"]
        oo = ((d["a"] % 2 == 1) & (d["b"] % 2 == 1)).astype(float); ee = ((d["a"] % 2 == 0) & (d["b"] % 2 == 0)).astype(float)
        out = {"original": score(d, X, te)}
        # sum table
        T = np.zeros_like(X)
        for v in np.unique(s): T[s == v] = X[tr & (s == v)].mean(0) if (tr & (s == v)).any() else X[tr].mean(0)
        out["sum table"] = score(d, T, te)
        # parametric fits per neuron, on per-s means of the train half (fast), then evaluated per row
        sv = np.arange(s.max() + 1)
        Ms = np.stack([X[tr & (s == v)].mean(0) if (tr & (s == v)).any() else X[tr].mean(0) for v in sv])
        preds = {nm: np.zeros_like(X) for nm in ("bump", "bump + parity gate", "two bumps + parity gate", "parity gate only")}
        for n in range(256):
            y_tr = X[tr, n]
            # parity gate only
            G = np.c_[np.ones(tr.sum()), oo[tr], ee[tr]]; cg = np.linalg.lstsq(G, y_tr, rcond=None)[0]
            preds["parity gate only"][:, n] = np.c_[np.ones(len(s)), oo, ee] @ cg
            for nm, nb in (("bump", 1), ("two bumps + parity gate", 2)):
                _, params = fit_bumps(Ms[:, n], sv.astype(float), nb)
                feats = lambda ss: np.column_stack([np.exp(-0.5 * ((ss - c) / w) ** 2) for c, w in params])
                if nm == "bump":
                    Xd = np.c_[np.ones(len(s)), feats(s.astype(float))]
                    co = np.linalg.lstsq(Xd[tr], y_tr, rcond=None)[0]; preds["bump"][:, n] = Xd @ co
                    Xd2 = np.c_[Xd, oo, ee]; co2 = np.linalg.lstsq(Xd2[tr], y_tr, rcond=None)[0]; preds["bump + parity gate"][:, n] = Xd2 @ co2
                else:
                    Xd = np.c_[np.ones(len(s)), feats(s.astype(float)), oo, ee]
                    co = np.linalg.lstsq(Xd[tr], y_tr, rcond=None)[0]; preds[nm][:, n] = Xd @ co
        for nm, P in preds.items(): out[nm] = score(d, P, te)
        out["params per neuron"] = {"sum table": int(len(np.unique(s))), "bump": 4, "bump + parity gate": 6, "two bumps + parity gate": 9, "parity gate only": 3}
        res[f"{b}/k{k}"] = out
        print(f"b{b} digit {k}:", {nm: round(v, 3) for nm, v in out.items() if nm != "params per neuron"}, flush=True)
    save_json("t08_compact_cells", res)
print("DONE")
