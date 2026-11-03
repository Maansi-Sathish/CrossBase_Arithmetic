"""F22 (forward passes): binary mode. Is it set by which digits appear, or by how long the numbers are?

Random digit strings in context (not real sums), with controlled alphabet and length. Test line: short, digits 0/1, units 1+1.
Rule 2 writes 0, any larger base writes 2.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, Model, save_json
torch.set_grad_enabled(False)
model = Model(); res = {}
def rate(S, bs=256):
    pr = []
    for i in range(0, len(S), bs):
        seqs = [[model.vocab[c] for c in s] for s in S[i:i + bs]]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); lg, c = model.forward(x.to(DEVICE)); pr.append(lg[torch.arange(len(seqs), device=DEVICE), q].argmax(-1).cpu().numpy())
    pr = np.concatenate(pr); return float((pr == model.vocab["0"]).mean()), float((pr == model.vocab["2"]).mean())
def test(rng):
    return "1" + "".join(rng.choice("01") for _ in range(rng.randint(0, 2))) + "1+1" + "".join(rng.choice("01") for _ in range(rng.randint(0, 2))) + "1="
for alpha in ("01", "012", "0123", "01234567"):
    for L in (1, 2, 3, 4, 5, 6, 7, 8, 10):
        rng = random.Random(L * 13 + len(alpha)); S = []
        for _ in range(500):
            lines = []
            for _ in range(5):
                num = lambda n: rng.choice(alpha[1:]) + "".join(rng.choice(alpha) for _ in range(n - 1))
                lines.append(f"{num(L)}+{num(L)}={num(L + rng.randint(0, 1))}")
            S.append("\n".join(lines) + "\n" + test(rng))
        res[f"{alpha}/L{L}"] = rate(S)
    print(f"alphabet {alpha}: (rule2, other) by number length", {L: tuple(np.round(res[f'{alpha}/L{L}'], 2).tolist()) for L in (1, 2, 3, 4, 5, 6, 7, 8, 10)}, flush=True)
save_json("f22_binary_length", res); print("DONE")
