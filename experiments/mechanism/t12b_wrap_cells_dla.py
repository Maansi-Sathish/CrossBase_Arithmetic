"""T12b: in hex prompts with s in 10..15, do decimal's wrap cells still push toward the decimal answer (s-10)?
Direct logit contribution (through ln_f scale) of those cells to token s-10 and to the hex answer, in decimal vs hex."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]
U = model.WE[torch.as_tensor(tok, device=DEVICE)].T * sd["ln_f.weight"][:, None]
def load(b):
    t = Table(b); rows = t.select(QUERY, 0); rows = rows[t.meta["correct"][rows] == 1]
    s = t.meta["a"][rows] % b + t.meta["b"][rows] % b; R = torch.as_tensor(t.get("resid1", rows), device=DEVICE)
    scale = (R - R.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
    return s, torch.as_tensor(t.get("mlp1", rows), device=DEVICE), scale
s10, X10, sc10 = load(10); s16, X16, sc16 = load(16)
# decimal wrap cells: top-10 per sum by contribution to the decimal answer on decimal prompts (s = 10..15)
cells = set(); per = {}
for v in range(10, 16):
    m = torch.as_tensor(s10 == v, device=DEVICE)
    c = ((X10[m] - X10.mean(0)) / sc10[m] * (Wo @ U[:, v - 10])).mean(0)
    cells |= set(torch.argsort(-c)[:10].tolist())
cells = torch.tensor(sorted(cells), device=DEVICE); res = {"n_cells": len(cells)}
rows = {}
for v in range(10, 16):
    out = {}
    for nm, s_, X, sc in (("decimal", s10, X10, sc10), ("hex", s16, X16, sc16)):
        m = torch.as_tensor(s_ == v, device=DEVICE); base_mean = X.mean(0)
        contrib = ((X[m][:, cells] - base_mean[cells]) / sc[m]) @ (Wo[cells] @ U)  # [n, 16 tokens]
        out[nm] = {"to decimal answer": float(contrib[:, v - 10].mean()), "to hex answer": float(contrib[:, v].mean())}
    rows[v] = out
res["by_sum"] = rows
print("decimal wrap cells:", len(cells))
for v, o in rows.items():
    print(f" s={v}: decimal prompts -> push decimal answer {o['decimal']['to decimal answer']:+.2f}, hex answer {o['decimal']['to hex answer']:+.2f} | hex prompts -> decimal answer {o['hex']['to decimal answer']:+.2f}, hex answer {o['hex']['to hex answer']:+.2f}")
save_json("t12b_wrap_cells_dla", res); print("DONE")
