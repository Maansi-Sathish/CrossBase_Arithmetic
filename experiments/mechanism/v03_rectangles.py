"""V03 (stored + weights): are layer 1 neurons AND gates on digit sets, and does the base gate them?

GELU applied to f(a)+f(b) acts as AND: it fires only if both digits fall in a set S where f is high.
A. Per active neuron at digit 0: fit a rectangle (a in Sa, b in Sb), from thresholds on the marginal pre-activation. R^2 of the rectangle vs the
   real table, against the additive fit. Are Sa and Sb the same set? What kinds of sets (odd, even, high, low, other)?
B. Voting: for each digit pair, how many rectangles fire? Does replacing neurons with binary rectangles still give right answers?
C. Base gating and wrap: for conflict pairs, which neurons drive (own rule - other rule)? Show their sets, firing in each base, and what they write.
   Does each base's gated-on set write that base's wrapped digit?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); tt = torch.as_tensor(tok, device=DEVICE)
Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]; W_tok = (Wo @ model.WE[tt].T).cpu().numpy()
def load(b):
    t = Table(b); rows = t.select(QUERY, 0); keep = t.meta["correct"][rows] == 1; rows = rows[keep]
    a, c = t.meta["a"][rows] % b, t.meta["b"][rows] % b
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    return dict(a=a, b=c, pre=pre.cpu().numpy(), post=gelu_new(pre).cpu().numpy(), mid=mid.cpu().numpy())
D = {b: load(b) for b in BASES}
def tab(d, b, f):
    T = np.zeros((b, b, 256))
    for i in range(b):
        for j in range(b):
            m = (d["a"] == i) & (d["b"] == j); T[i, j] = d[f][m].mean(0) if m.any() else d[f].mean(0)
    return T
def classify(S, b):
    S = set(S)
    if not S or len(S) == b: return "all/none"
    odd, even = {x for x in range(b) if x % 2}, {x for x in range(b) if not x % 2}
    j = lambda x, y: len(x & y) / len(x | y)
    if j(S, odd) >= .8: return "odd"
    if j(S, even) >= .8: return "even"
    if j(S, set(range(b - len(S), b))) >= .8: return "high"
    if j(S, set(range(len(S)))) >= .8: return "low"
    return "other"
T = {b: tab(D[b], b, "post") for b in BASES}; P = {b: tab(D[b], b, "pre") for b in BASES}
sets = {}
for b in BASES:
    t, p = T[b], P[b]; v = t.var((0, 1)); act = np.where(v > 0.003)[0]
    fa, fb = p.mean(1), p.mean(0)
    rect = {}; add_r2, rect_r2, same, kinds, width = [], [], [], {}, []
    for n in act:
        tc = t[:, :, n]; tot = ((tc - tc.mean()) ** 2).sum()
        add = tc.mean(1, keepdims=True) + tc.mean(0, keepdims=True) - tc.mean()
        add_r2.append(1 - ((tc - add) ** 2).sum() / tot)
        # threshold each marginal at the midpoint giving the best rectangle
        best = (-9, None, None)
        for qa in np.quantile(fa[:, n], np.linspace(.1, .9, 9)):
            for qb in np.quantile(fb[:, n], np.linspace(.1, .9, 9)):
                Sa, Sb = fa[:, n] > qa, fb[:, n] > qb
                m = np.outer(Sa, Sb)
                if m.sum() == 0 or m.all(): continue
                pred = np.where(m, tc[m].mean(), tc[~m].mean())
                r = 1 - ((tc - pred) ** 2).sum() / tot
                if r > best[0]: best = (r, np.where(Sa)[0].tolist(), np.where(Sb)[0].tolist())
        rect_r2.append(best[0]); rect[int(n)] = (best[1], best[2], best[0])
        ja = set(best[1]); jb = set(best[2]); same.append(len(ja & jb) / max(len(ja | jb), 1))
        k = classify(best[1], b); kinds[k] = kinds.get(k, 0) + 1; width.append(len(best[1]) / b)
    sets[b] = rect
    w = v[act] / v[act].sum()
    res["A"][b] = {"n_active": len(act), "additive_r2_weighted": float((np.array(add_r2) * w).sum()), "rectangle_r2_weighted": float((np.array(rect_r2) * w).sum()), "rectangle_r2_median": float(np.median(rect_r2)), "Sa_Sb_overlap_mean": float(np.mean(same)), "set_kinds": kinds, "set_width_mean": float(np.mean(width)), "n_rect_r2_gt_07": int((np.array(rect_r2) > .7).sum())}
    print("A", b, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["A"][b].items()}, flush=True)
    # B rectangles as replacement
    R = np.zeros_like(t); mean_q = D[b]["post"].mean(0)
    R[:, :] = mean_q
    for n, (Sa, Sb, _) in rect.items():
        m = np.zeros((b, b), bool); m[np.ix_(Sa, Sb)] = True
        R[:, :, n] = np.where(m, t[:, :, n][m].mean(), t[:, :, n][~m].mean())
    midm = D[b]["mid"].mean(0)
    lg = model.unembed(torch.as_tensor(midm, device=DEVICE, dtype=torch.float32) + torch.as_tensor(R.reshape(b * b, -1), device=DEVICE, dtype=torch.float32) @ Wo + bo).cpu().numpy()[:, tok]
    target = np.add.outer(np.arange(b), np.arange(b)).ravel() % b
    acc = float((lg.argmax(1) == target).mean())
    # how many fire per pair
    counts = np.zeros((b, b))
    for n, (Sa, Sb, _) in rect.items():
        counts[np.ix_(Sa, Sb)] += 1
    # distinct fire-patterns over pairs with different outputs, as identifiability
    res["B"][b] = {"digit0_acc_all_binary_rectangles": acc, "mean_rectangles_per_pair": float(counts.mean())}
    print("B", b, res["B"][b], flush=True)
# C
for x, y in [(8, 10), (10, 16), (8, 16)]:
    lim = min(x, y); Tx, Ty = T[x][:lim, :lim], T[y][:lim, :lim]
    S = np.add.outer(np.arange(lim), np.arange(lim)); conf = S >= lim
    wx = W_tok[:, tok[np.arange(0)] if False else np.arange(0)] if False else None
    gain = np.zeros(256); 
    for i, j in zip(*np.where(conf)):
        ox, oy = S[i, j] % x, S[i, j] % y
        gain += (Tx[i, j] - Ty[i, j]) * (W_tok[:, ox] - W_tok[:, oy])
    gain /= conf.sum()
    order = np.argsort(-np.abs(gain)); total = np.abs(gain).sum()
    tops = []
    for n in order[:8]:
        cellsx = Tx[:, :, n]; cellsy = Ty[:, :, n]
        wtop = "".join(DIGITS[i] for i in np.argsort(-W_tok[n, :16])[:5]); wbot = "".join(DIGITS[i] for i in np.argsort(W_tok[n, :16])[:5])
        tops.append({"neuron": int(n), "gain": float(gain[n]), "mean_fire_conf_x": float(cellsx[conf].mean()), "mean_fire_conf_y": float(cellsy[conf].mean()), "mean_fire_nonconf_x": float(cellsx[~conf].mean()), "mean_fire_nonconf_y": float(cellsy[~conf].mean()), "set_x": sets[x].get(int(n), [None])[0], "set_y": sets[y].get(int(n), [None])[0], "writes": wtop, "suppresses": wbot})
    # is the difference concentrated on conflict cells? Does firing separate cleanly by base?
    dT = Tx - Ty
    res["C"][f"{x}v{y}"] = {"neurons": tops, "share_change_on_conflict_cells": float((dT[conf] ** 2).sum() / (dT**2).sum()), "conflict_cell_share": float(conf.mean()), "top8_gain_share": float(np.abs(gain[order[:8]]).sum() / total), "top32_gain_share": float(np.abs(gain[order[:32]]).sum() / total)}
    print(f"C {x} vs {y}: {res['C'][f'{x}v{y}']['share_change_on_conflict_cells']:.2f} of the change is on conflict pairs (which are {conf.mean():.2f} of pairs); top8 gain share {res['C'][f'{x}v{y}']['top8_gain_share']:.2f}", flush=True)
    for t_ in tops[:6]:
        print("   n%3d gain %+.2f fire on conflict %+.2f/%+.2f, on non-conflict %+.2f/%+.2f, S(%d)=%s S(%d)=%s writes %s suppresses %s" % (t_["neuron"], t_["gain"], t_["mean_fire_conf_x"], t_["mean_fire_conf_y"], t_["mean_fire_nonconf_x"], t_["mean_fire_nonconf_y"], x, t_["set_x"], y, t_["set_y"], t_["writes"], t_["suppresses"]), flush=True)
save_json("v03_rectangles", res); print("DONE")
