"""T18 (stored + weights): a formula for digit-1 neurons over two sums.
pre_n ~ u_n[s1] + v_n[p0]  with s1 = a1+b1 (current pair sum) and p0 = a0+b0 (previous pair sum); neuron = GELU(pre).
Variants: tables for u and v; smooth u, v (cubic + parity, 5 numbers each); smooth u + step-plus-slope v (carry bit + graded slope).
Fitted on pre-activations (half the prompts), scored by readout on the other half."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; res = {}
for b in (8, 10, 16):
    t = Table(b); rows = t.select(QUERY, 1); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    s1 = (A // b) % b + (B // b) % b; p0 = A % b + B % b; c = (p0 >= b).astype(int); s = s1 + c
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    P = (model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]).cpu().numpy()
    X = t.get("mlp1", rows); R = t.get("resid1", rows); tr = t.meta["idx"][rows] % 2 == 0; te = ~tr
    def score(pre_hat):
        post = gelu_new(torch.as_tensor(pre_hat[te], device=DEVICE, dtype=torch.float32))
        r = torch.as_tensor(R[te], device=DEVICE) + (post - torch.as_tensor(X[te], device=DEVICE)) @ Wo
        return float((model.unembed(r).argmax(-1).cpu().numpy() == tok[s[te] % b]).mean())
    def fit(D):
        co = np.linalg.lstsq(D[tr], P[tr], rcond=None)[0]; return D @ co
    oh = lambda v, n: np.eye(n)[v]
    z1 = (s1 - s1.mean()) / s1.std(); z0 = (p0 - p0.mean()) / p0.std()
    smooth = lambda z, v: np.c_[z, z**2, z**3, v % 2]
    out = {"exact pre-activation": score(P),
           "table u[s1] + table v[p0]": score(fit(np.c_[np.ones(len(P)), oh(s1, 2 * b - 1), oh(p0, 2 * b - 1)])),
           "table u[s1] + carry bit only": score(fit(np.c_[np.ones(len(P)), oh(s1, 2 * b - 1), c])),
           "smooth u + smooth v (9 numbers)": score(fit(np.c_[np.ones(len(P)), smooth(z1, s1), smooth(z0, p0)])),
           "smooth u + carry step + graded slope (7 numbers)": score(fit(np.c_[np.ones(len(P)), smooth(z1, s1), c, z0])),
           "smooth u + carry step + slope + slope x carry (8 numbers)": score(fit(np.c_[np.ones(len(P)), smooth(z1, s1), c, z0, z0 * c])),
           "table u[s1] + step + slope + slope x carry": score(fit(np.c_[np.ones(len(P)), oh(s1, 2 * b - 1), c, z0, z0 * c]))}
    # digit 0 for reference: same idea with only s0
    res[b] = out; print(b, {k: round(v, 3) for k, v in out.items()}, flush=True)
save_json("t18_digit1_formula", res); print("DONE")
