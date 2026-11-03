"""F05 (forward passes): which rule produces the carry? Random previous digits and written digit, including impossible
combinations. Candidate rules are scored on trials where the two carry values give different next digits."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}
def predict(strs, bs=512):
    out = []
    for i in range(0, len(strs), bs):
        seqs = [[model.vocab[c] for c in s] for s in strs[i:i + bs]]
        L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
        lg, _ = model.forward(x.to(DEVICE)); out.append(lg[torch.arange(len(seqs), device=DEVICE), q].argmax(-1).cpu().numpy())
    return np.concatenate(out)
RULES = {"written < a+b": lambda a, b, w, B: int(w < a + b), "operands only: a+b >= base": lambda a, b, w, B: int(a + b >= B),
         "written < a": lambda a, b, w, B: int(w < a), "written < b": lambda a, b, w, B: int(w < b), "written < max(a,b)": lambda a, b, w, B: int(w < max(a, b)), "written < min(a,b)": lambda a, b, w, B: int(w < min(a, b)), "no carry": lambda a, b, w, B: 0, "always carry": lambda a, b, w, B: 1}
for base in BASES:
    rng = random.Random(base * 31); res[base] = {}
    for mode in ("consistent", "random"):
        S, rec = [], []
        while len(S) < 4000:
            hi_a, hi_b = rng.randint(1, 999 // base), rng.randint(1, 999 // base)
            a0, b0 = rng.randrange(base), rng.randrange(base)
            A, B = hi_a * base + a0, hi_b * base + b0
            w = (a0 + b0) % base if mode == "consistent" else rng.randrange(base)
            S.append("\n".join(render(*s, base) for s in sample_shots(rng, (A, B))) + f"\n{to_base(A, base)}+{to_base(B, base)}={DIGITS[w]}")
            rec.append((a0, b0, w, hi_a % base + hi_b % base))
        pr = predict(S)
        out = {}
        for nm, fn in RULES.items():
            hit = [model.vocab[DIGITS[(s + fn(a, b, w, base)) % base]] == p for (a, b, w, s), p in zip(rec, pr)]
            out[nm] = float(np.mean(hit))
        # was the output either "next digit with carry" or "without"?
        two = [p in (model.vocab[DIGITS[s % base]], model.vocab[DIGITS[(s + 1) % base]]) for (a, b, w, s), p in zip(rec, pr)]
        out["output is one of the two carry options"] = float(np.mean(two))
        res[base][mode] = out
        print(base, mode, {k: round(v, 3) for k, v in out.items()}, flush=True)
save_json("f05_carry_rule", res)
print("DONE")
