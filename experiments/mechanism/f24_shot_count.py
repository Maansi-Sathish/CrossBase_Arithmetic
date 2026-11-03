"""F24 (forward passes): why is exact accuracy 0 with fewer than 5 examples? Digits, stopping, and how the model knows it is on the last line."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, Model, Problem, greedy, sample_shots, save_json
torch.set_grad_enabled(False)
model = Model(); res = {}
for b in BASES:
    res[b] = {}
    for n in (0, 1, 2, 3, 4, 5, 6, 7, 8):
        rng = random.Random(b * 10 + n); P = []
        tries = 0
        while len(P) < 1500 and tries < 30000:
            tries += 1
            a, c = rng.randint(0, 999), rng.randint(0, 999); pp = Problem(a, c, b, sample_shots(rng, (a, c), k=n))
            if len(pp.prompt) + 16 <= 256: P.append(pp)
        if len(P) < 100: continue
        out = greedy(model, P, max_new=14, bs=256)
        exact = sum(o == p.answer for o, p in zip(out, P)) / len(P)
        digits = sum(o[:len(p.answer)] == p.answer for o, p in zip(out, P)) / len(P)
        nl = sum(o[:len(p.answer)] == p.answer and o[len(p.answer):len(p.answer) + 1] == "\n" for o, p in zip(out, P)) / len(P)
        res[b][n] = {"exact": exact, "digits_right": digits, "digits_right_then_newline": nl, "example": repr(out[0])}
    print(b, {n: (round(v["exact"], 2), round(v["digits_right"], 2), round(v["digits_right_then_newline"], 2)) for n, v in res[b].items()}, "ex 4 shots:", res[b][4]["example"], "6 shots:", res[b].get(6, {}).get("example"), flush=True)
save_json("f24_shot_count", res); print("DONE")
