"""F01 (forward passes): accuracy on held-out pairs vs training pairs, using the real validation split.

A pair absent from a base's validation split belongs to its training pool, so only the validation file is needed.
Scores the whole output, including stopping, plus the team's first-run-of-digits rule.
"""
import random, re, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pyarrow.parquet as pq
from core import BASES, DATASET_ID, DEVICE, Model, Problem, greedy, save_json, wilson
from huggingface_hub import hf_hub_download

print("device", DEVICE, flush=True)
model = Model()
val = pq.read_table(hf_hub_download(DATASET_ID, "data/validation-00000-of-00001.parquet", repo_type="dataset"), columns=["base", "a", "b"]).to_pandas()
print("val rows", len(val), val.groupby("base").size().to_dict(), "max operand", int(max(val.a.max(), val.b.max())), flush=True)
res = {}
N = 20000
for b in BASES:
    v = val[val.base == b]
    codes = v.a.values.astype(np.int64) * 1000 + v.b.values
    in_val = np.zeros(1000 * 1000, bool); in_val[codes] = True
    train_codes = np.where(~in_val)[0]
    rng = np.random.default_rng(b); py = random.Random(b)
    r = {"n_val_pool": int(in_val.sum()), "n_train_pool": len(train_codes)}
    for name, pool in (("heldout", codes), ("train", train_codes)):
        probs = []
        for c in rng.choice(pool, N, replace=False).tolist():
            x, y = c // 1000, c % 1000
            shots = []
            while len(shots) < 5:
                t = int(train_codes[py.randrange(len(train_codes))]); pr = (t // 1000, t % 1000)
                if pr != (x, y) and pr not in shots:
                    shots.append(pr)
            probs.append(Problem(x, y, b, shots))
        out = greedy(model, probs, bs=512)
        full = sum(o == p.answer for o, p in zip(out, probs))
        first = 0
        for o, p in zip(out, probs):
            m = re.match(r"\s*([0-9A-F]+)", o)
            first += bool(m and m.group(1) == p.answer)
        r[name] = {"full": wilson(full, N), "first_run_only": wilson(first, N)}
    res[b] = r
    print(b, {k: (np.round(v["full"], 4).tolist(), round(v["first_run_only"][0], 4)) if isinstance(v, dict) else v for k, v in r.items()}, flush=True)
save_json("f01_heldout", res)
print("DONE")
