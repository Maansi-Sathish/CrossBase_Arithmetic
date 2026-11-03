"""T11b: compact digit-1 formula with a graded carry. Per neuron: two bumps over s + carry-aware parity gates + terms in the
previous column's sum p0 = a0+b0 (linear, quadratic, each crossed with the carry bit)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; res = {}
for b in (8, 10, 16):
    t = Table(b); rows = t.select(QUERY, 1); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    a, bb = (A // b) % b, (B // b) % b; p0 = (A % b) + (B % b); c = (p0 >= b).astype(int); s = a + bb + c
    X = t.get("mlp1", rows); R = t.get("resid1", rows); tr = t.meta["idx"][rows] % 2 == 0; te = ~tr
    oo = ((a % 2 == 1) & (bb % 2 == 1)).astype(float); ee = ((a % 2 == 0) & (bb % 2 == 0)).astype(float); cf = c.astype(float)
    z = (p0 - b) / b
    gates = np.c_[oo, ee, cf, oo * cf, ee * cf]; graded = np.c_[z, z**2, z * cf, z**2 * cf]
    sv = np.arange(s.max() + 1); grid = [(cc, w) for cc in sv.astype(float) for w in (0.7, 1.2, 2.0, 3.5, 6.0)]
    def score(P):
        r = torch.as_tensor(R[te], device=DEVICE) + (torch.as_tensor(P[te], device=DEVICE, dtype=torch.float32) - torch.as_tensor(X[te], device=DEVICE)) @ Wo
        return float((model.unembed(r).argmax(-1).cpu().numpy() == tok[s[te] % b]).mean())
    Ms = np.stack([X[tr & (s == v)].mean(0) if (tr & (s == v)).any() else X[tr].mean(0) for v in sv])
    P1 = np.zeros_like(X); P2 = np.zeros_like(X)
    for n in range(256):
        ch = []
        for _ in range(2):
            best = (np.inf, None)
            for g in grid:
                D = np.column_stack([np.ones(len(sv))] + [np.exp(-0.5 * ((sv - c_) / w_) ** 2) for c_, w_ in ch + [g]])
                e = ((Ms[:, n] - D @ np.linalg.lstsq(D, Ms[:, n], rcond=None)[0]) ** 2).sum()
                if e < best[0]: best = (e, g)
            ch.append(best[1])
        Bm = np.column_stack([np.ones(len(s))] + [np.exp(-0.5 * ((s - c_) / w_) ** 2) for c_, w_ in ch])
        for P, D in ((P1, np.c_[Bm, gates]), (P2, np.c_[Bm, gates, graded])):
            P[:, n] = D @ np.linalg.lstsq(D[tr], X[tr, n], rcond=None)[0]
    res[b] = {"two bumps + carry-aware gates (8 numbers)": score(P1), "+ graded previous-column sum (12 numbers)": score(P2)}
    print(b, {k: round(v, 3) for k, v in res[b].items()}, flush=True)
save_json("t11b_digit1_compact", res); print("DONE")
