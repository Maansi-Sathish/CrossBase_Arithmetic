"""T12 (stored + weights). Q2: how does the base gate a cell?  Digit 0, pairs (a,b) that exist in both bases.
Per neuron: pre_y(a,b) ~ g * pre_x(a,b) + o  (y, x = two bases). Compare: offset only (g=1), gain only (o=0), both.
Then for cells that work for wrapping sums in one base: their offset into the other base, and how much of the drop in
activity is explained by the offset alone (GELU(pre_x + o) vs actual)."""
import itertools, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"].cpu().numpy(); res = {}
WEt = model.WE[torch.as_tensor(tok, device=DEVICE)].cpu().numpy()
def load(b):
    t = Table(b); rows = t.select(QUERY, 0); rows = rows[t.meta["correct"][rows] == 1]
    a, bb = t.meta["a"][rows] % b, t.meta["b"][rows] % b
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = (model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]).cpu().numpy()
    return a, bb, pre
D = {b: load(b) for b in (8, 10, 16)}
def pair_means(b, lim):
    a, bb, pre = D[b]; return {(i, j): pre[(a == i) & (bb == j)].mean(0) for i in range(lim) for j in range(lim) if ((a == i) & (bb == j)).any()}
for x, y in (("10", "16"), ("16", "10"), ("8", "10"), ("10", "8")):
    x, y = int(x), int(y); lim = min(x, y); px, py = pair_means(x, lim), pair_means(y, lim); keys = [k for k in px if k in py]
    Px = np.stack([px[k] for k in keys]); Py = np.stack([py[k] for k in keys]); S = np.array([sum(k) for k in keys])
    tot = ((Py - Py.mean(0)) ** 2).sum(0) + 1e-9
    r_off = 1 - (((Py - Px) - (Py - Px).mean(0)) ** 2).sum(0) / tot
    g_only = (Px * Py).sum(0) / ((Px ** 2).sum(0) + 1e-9); r_gain = 1 - ((Py - g_only * Px) ** 2).sum(0) / tot
    both = [np.polyfit(Px[:, n], Py[:, n], 1) for n in range(256)]; gain = np.array([q[0] for q in both]); off = np.array([q[1] for q in both])
    r_both = 1 - ((Py - (gain * Px + off)) ** 2).sum(0) / tot
    post_x = gelu_new(torch.as_tensor(Px)).numpy(); post_y = gelu_new(torch.as_tensor(Py)).numpy()
    imp = np.abs(post_x - post_x.mean(0)).mean(0) * np.linalg.norm(Wo, axis=1); w = imp / imp.sum()
    # cells that serve wrapping sums in x (sums >= x, valid here only if x < y): top-5 contributors to the right token per sum
    lo_, hi_ = min(x, y), max(x, y); wrap = (S >= lo_) & (S < hi_)  # sums where the two bases disagree
    cells = set()
    for v in np.unique(S[wrap]):
        out_tok = v % x
        contrib = post_x[S == v].mean(0) * (Wo @ (WEt[out_tok] - WEt[:max(x, y)].mean(0)))
        cells |= set(np.argsort(-contrib)[:5].tolist())
    cells = np.array(sorted(cells)); m = wrap
    pred_off = gelu_new(torch.as_tensor(Px[:, cells] + (Py - Px)[:, cells].mean(0))).numpy()
    level_actual = post_y[m][:, cells].mean() / post_x[m][:, cells].mean(); level_offset_only = pred_off[m].mean() / post_x[m][:, cells].mean()
    res[f"{x}->{y}"] = {"weighted R2 offset only": float((w * r_off).sum()), "weighted R2 gain only": float((w * r_gain).sum()), "weighted R2 gain + offset": float((w * r_both).sum()),
                        "weighted median gain": float(np.median(gain[w > np.median(w)])), "n cells": len(cells), "cell mean offset": float((Py - Px)[:, cells].mean()), "other neurons mean offset": float(np.delete(Py - Px, cells, 1).mean()),
                        "cell activity in y / x on those sums": float(level_actual), "predicted by offset alone": float(level_offset_only)}
    print(f"{x}->{y}", {k: round(v, 3) for k, v in res[f'{x}->{y}'].items()}, flush=True)
save_json("t12_gating", res); print("DONE")
