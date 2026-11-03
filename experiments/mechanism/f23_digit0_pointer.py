"""F23 (forward passes): is digit 0 also a pointer, hidden by saturation? Larger shifts of the '+' and '=' position tags."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {}; tok = np.array([model.vocab[c] for c in DIGITS])
for b in BASES:
    rng = random.Random(b); P = []
    while len(P) < 500:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) < 3 or len(sb) < 3: continue
        ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_))); c = len(ctx) + 1
        P.append((f"{ctx}\n{sa}+{sb}=", c + len(sa), c + len(sa) + 1 + len(sb), a, c_))
    out = {}
    for tgt in ("plus", "eq"):
        for d in (-8, -5, -3, -2, -1, 0, 1, 2, 3, 5):
            ma = np.zeros(3); mb = np.zeros(3); ok = 0
            for i in range(0, 500, 125):
                ch = P[i:i + 125]; seqs = [[model.vocab[x] for x in p[0]] for p in ch]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
                for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
                x = x.to(DEVICE); pl = torch.tensor([p[1] for p in ch], device=DEVICE); eq = torch.tensor([p[2] for p in ch], device=DEVICE); J = torch.arange(len(ch), device=DEVICE); idx = torch.arange(n, device=DEVICE)[None]
                m = idx == (pl if tgt == "plus" else eq)[:, None]; pos = idx.expand(x.shape); delta = (PE[(pos + d).clamp(0, 255)] - PE[pos]) * m[..., None]
                lg, c = model.forward(x, hooks={"resid_pre0": lambda t, delta=delta: t + delta}, capture={"pattern1"}); pat = c["pattern1"]
                for o in range(3):
                    ma[o] += pat[J, 1, eq, pl - 1 - o].sum().item(); mb[o] += pat[J, 0, eq, eq - 1 - o].sum().item()
                A = np.array([p[3] for p in ch]); B = np.array([p[4] for p in ch]); ok += (lg[J, eq].argmax(-1).cpu().numpy() == tok[(A + B) % b]).sum()
            out[f"{tgt}/{d}"] = {"a_mass_last3": (ma / 500).round(3).tolist(), "b_mass_last3": (mb / 500).round(3).tolist(), "output_correct": float(ok / 500)}
    res[b] = out
    print("b", b, "plus shift: a-reader mass on a0,a1,a2 and output:", {d: (out[f"plus/{d}"]["a_mass_last3"], round(out[f"plus/{d}"]["output_correct"], 2)) for d in (-8, -5, -3, -1, 0, 1, 3, 5)}, flush=True)
    print("     eq shift: b-reader mass on b0,b1,b2 and output:", {d: (out[f"eq/{d}"]["b_mass_last3"], round(out[f"eq/{d}"]["output_correct"], 2)) for d in (-8, -5, -3, -1, 0, 1, 3, 5)}, flush=True)
save_json("f23_digit0_pointer", res); print("DONE")
