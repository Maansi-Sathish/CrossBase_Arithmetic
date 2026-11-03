"""T02 (stored + weights): are layer 1 neurons "place cells" over the sum s, writing that sum's output digit?

A. Per neuron: firing by s (digit 0). Preferred s. Does one bump plus a parity term describe it?
B. Does each neuron write o(preferred s) = preferred s mod base? Chance is 1/base.
C. For each s, which neurons push the correct token most? Are they tuned near that s?
D. Wrapping and bases: for s in 10..15 (wraps in decimal, not in hex), are the cells that do the work in one base silent in the other?
   For s < 8 (no wrap in 8, 10, 16), are the same cells shared?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]
W_tok = (Wo @ model.WE[torch.as_tensor(tok, device=DEVICE)].T).cpu().numpy()
def tune(b):
    t = Table(b); rows = t.select(QUERY, 0); keep = t.meta["correct"][rows] == 1; rows = rows[keep]
    s = t.meta["a"][rows] % b + t.meta["b"][rows] % b; X = t.get("mlp1", rows)
    T = np.stack([X[s == v].mean(0) for v in range(2 * b - 1)])
    return T, X.std(0)
TT = {}
for b in (8, 10, 16):
    T, sdv = tune(b); TT[b] = T; ns = 2 * b - 1; sv = np.arange(ns)
    wr = W_tok[:, :b] - W_tok[:, :b].mean(1, keepdims=True)
    imp = sdv * np.abs(wr).max(1)
    order = np.argsort(-imp); top = order[:40]
    # A: bump + parity fit
    par = (sv % 2 == 0).astype(float)
    r2s, pref = np.zeros(256), np.zeros(256, int)
    for n in range(256):
        y = T[:, n]; tot = ((y - y.mean()) ** 2).sum() + 1e-12; best = -9
        for c in np.arange(0, ns - 0.5, 0.5):
            for w in (1, 1.5, 2, 3, 4, 6):
                g = np.exp(-0.5 * ((sv - c) / w) ** 2)
                X = np.c_[np.ones(ns), g, par]
                co = np.linalg.lstsq(X, y, rcond=None)[0]
                r = 1 - ((y - X @ co) ** 2).sum() / tot
                if r > best and co[1] > 0: best, bc = r, c
        r2s[n] = best; pref[n] = int(round(bc)) if best > -9 else int(np.argmax(y))
    w_ = imp / imp.sum()
    res["A"][b] = {"importance_weighted_bump_r2": float((r2s * w_).sum()), "share_importance_bump_r2_gt_0.8": float(w_[r2s > .8].sum()), "top40_bump_r2_median": float(np.median(r2s[top])), "top40_preferred_s": pref[top].tolist()}
    # B: write match
    wtop = wr.argmax(1)
    match_wrapped = (wtop == pref % b); match_unwrapped = (wtop == np.minimum(pref, 15)) & (pref < 16)
    res["B"][b] = {"top40_writes_pref_mod_base": float(match_wrapped[top].mean()), "top40_writes_pref_mod_base_or_neighbour": float(((np.abs(wtop - pref % b) <= 1) | (np.abs(wtop - pref % b) == b - 1))[top].mean()), "chance": 1 / b, "chance_with_neighbours": 3 / b}
    # C: who pushes the correct token at each s?
    near, gaps = [], []
    for v in range(ns):
        contrib = (T[v] - T.mean(0)) * wr[:, v % b]
        best = np.argsort(-contrib)[:5]
        gaps.append(float(np.median(np.abs(pref[best] - v))))
        near.append(float(np.mean(np.abs(pref[best] - v) <= 2)))
    res["C"][b] = {"median_distance_pref_to_s_top5": gaps, "share_top5_within_2": near}
    print(f"A/B/C b{b}: bump R2 weighted {res['A'][b]['importance_weighted_bump_r2']:.2f} (top40 median {res['A'][b]['top40_bump_r2_median']:.2f}); top40 write o(pref) {res['B'][b]['top40_writes_pref_mod_base']:.2f} (chance {1 / b:.2f}), within one {res['B'][b]['top40_writes_pref_mod_base_or_neighbour']:.2f} (chance {3 / b:.2f}); top-5 contributors tuned within 2 of s: {np.round(near, 1).tolist()}", flush=True)
    TT[b] = (T, pref, imp, wr)
# D gating
T10, p10, i10, wr10 = TT[10]; T16, p16, i16, wr16 = TT[16]; T8, p8, i8, wr8 = TT[8]
def workers(T, wr, b, svals, n=10):
    out = set()
    for v in svals:
        contrib = (T[v] - T.mean(0)) * wr[:, v % b]
        out |= set(np.argsort(-contrib)[:n].tolist())
    return sorted(out)
for name, (x, Tx, wx), (y, Ty, wy), svals in [("s 10-15: decimal (wraps) vs hex (no wrap)", (10, T10, wr10), (16, T16, wr16), range(10, 16)), ("s 8-9: octal (wraps) vs decimal (no wrap)", (8, T8, wr8), (10, T10, wr10), range(8, 10)), ("s 0-7: no wrap in 8, 10, 16", (10, T10, wr10), (16, T16, wr16), range(0, 8))]:
    wx_, wy_ = workers(Tx, wx, x, svals), workers(Ty, wy, y, svals)
    shared = set(wx_) & set(wy_)
    fx = np.array([(Tx[list(svals), n] - Tx[:, n].mean()).mean() for n in wx_]); fx_in_y = np.array([(Ty[list(svals), n] - Ty[:, n].mean()).mean() for n in wx_])
    fy = np.array([(Ty[list(svals), n] - Ty[:, n].mean()).mean() for n in wy_]); fy_in_x = np.array([(Tx[list(svals), n] - Tx[:, n].mean()).mean() for n in wy_])
    res["D"][name] = {"workers_x": len(wx_), "workers_y": len(wy_), "shared": len(shared), "x_workers_relative_firing_in_y": float(np.median(fx_in_y / np.where(np.abs(fx) > 1e-3, fx, np.nan))), "y_workers_relative_firing_in_x": float(np.median(fy_in_x / np.where(np.abs(fy) > 1e-3, fy, np.nan)))}
    print(f"D {name}: workers {len(wx_)}/{len(wy_)}, shared {len(shared)}; {x}'s workers fire in {y} at {res['D'][name]['x_workers_relative_firing_in_y']:.2f} of their level, {y}'s workers fire in {x} at {res['D'][name]['y_workers_relative_firing_in_x']:.2f}", flush=True)
save_json("t02_place_cells", res); print("DONE")
