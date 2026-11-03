"""T18b: a compact per-neuron formula over the digits:  neuron_n = GELU( mu_n + f_n(a_k) + f'_n(b_k) + c * C_n ),
with f, f' each a small smooth code (polynomial in the digit value + parity), fitted to pre-activations on half, scored by readout.
Digits 0 (c = 0 always) and 1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; res = {}
for b in (8, 10, 16):
    for k in (0, 1):
        t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
        keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
        M = b**k; a, bb = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(float); s = a + bb + c.astype(int)
        R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
        mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
        P = (model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]).cpu().numpy()
        X = t.get("mlp1", rows); R = t.get("resid1", rows); tr = t.meta["idx"][rows] % 2 == 0; te = ~tr
        def score(D):
            co = np.linalg.lstsq(D[tr], P[tr], rcond=None)[0]; post = gelu_new(torch.as_tensor((D @ co)[te], device=DEVICE, dtype=torch.float32))
            r = torch.as_tensor(R[te], device=DEVICE) + (post - torch.as_tensor(X[te], device=DEVICE)) @ Wo
            return float((model.unembed(r).argmax(-1).cpu().numpy() == tok[s[te] % b]).mean())
        def code(v, deg):
            z = (v - (b - 1) / 2) / ((b - 1) / 2); return np.column_stack([z**d for d in range(1, deg + 1)] + [v % 2])
        one = np.ones((len(P), 1)); cc = c[:, None]
        out = {"full digit tables A[a] + B[b] + c*C": score(np.c_[one, np.eye(b)[a], np.eye(b)[bb][:, 1:], cc])}
        for deg in (2, 3, 4, 6):
            out[f"polynomial degree {deg} + parity, per operand ({2 * (deg + 1) + 2} numbers)"] = score(np.c_[one, code(a, deg), code(bb, deg), cc])
        out["degree 4 + parity, same code for a and b (7 numbers)"] = score(np.c_[one, code(a, 4) + code(bb, 4), cc])
        res[f"{b}/k{k}"] = out; print(f"b{b} digit {k}:", {kk: round(v, 3) for kk, v in out.items()}, flush=True)
save_json("t18b_digit_formula", res); print("DONE")
