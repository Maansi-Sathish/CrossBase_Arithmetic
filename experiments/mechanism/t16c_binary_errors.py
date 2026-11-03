"""T16c: where do binary's errors actually occur? Greedy generation on 3000 random problems; for each wrong answer, the first
wrong position, classified: a digit (and where it sits relative to the operands' lengths), a missing stop, or an early stop.
Then for wrong digits: the true carry-in, and whether the model's digit equals the answer with the carry flipped."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from core import Model, greedy, sample_problems, save_json, schoolbook, to_base
model = Model(); P = [p for p in sample_problems(3000, 2, seed=91) if len(p.prompt) + 14 <= 256]
out = greedy(model, P, bs=256); rows = []
for o, p in zip(out, P):
    if o == p.answer: continue
    la, lb = len(to_base(p.a, 2)), len(to_base(p.b, 2)); k = next((i for i in range(min(len(o), len(p.answer))) if o[i] != p.answer[i]), min(len(o), len(p.answer)))
    if k == len(p.answer): kind = "did not stop"
    elif k == len(o): kind = "stopped early"
    else:
        sb = schoolbook(p.a, p.b, 2); flipped = (sb["a"][k] + sb["b"][k] + 1 - sb["c_in"][k]) % 2 == int(o[k])
        kind = "wrong digit (carry flipped)" if flipped else "wrong digit (other)"
    rows.append(dict(kind=kind, k=k, la=la, lb=lb, short=min(la, lb), long=max(la, lb), c_in=schoolbook(p.a, p.b, 2)["c_in"][k] if k < 12 else -1,
                     run=len(to_base(p.a + p.b, 2)) - len(to_base(p.a + p.b, 2).rstrip("1")) if False else None))
n = len(P); e = len(rows); kinds = {}
for r in rows: kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
rel = {}
for r in rows:
    if r["kind"].startswith("wrong digit"):
        tag = "k < shorter length" if r["k"] < r["short"] else ("k = shorter length" if r["k"] == r["short"] else "k > shorter length")
        rel[tag] = rel.get(tag, 0) + 1
cin = {}
for r in rows:
    if r["kind"].startswith("wrong digit"): cin[r["c_in"]] = cin.get(r["c_in"], 0) + 1
byk = {}
for r in rows: byk[r["k"]] = byk.get(r["k"], 0) + 1
# carry chains: length of the incoming carry run at the wrong digit
chain = []
for r, p in zip([r for r in rows], [p for o, p in zip(out, P) if o != p.answer]):
    if r["kind"].startswith("wrong digit"):
        sb = schoolbook(p.a, p.b, 2); k = r["k"]; L = 0
        while k - 1 - L >= 0 and sb["c_out"][k - 1 - L] == 1: L += 1
        chain.append(L)
res = {"n": n, "errors": e, "accuracy": 1 - e / n, "kinds": kinds, "wrong digit position vs shorter operand": rel, "carry-in at wrong digit": cin, "first wrong position": byk,
       "incoming carry chain length at wrong digit (median, share >= 3)": (float(np.median(chain)) if chain else None, float(np.mean(np.array(chain) >= 3)) if chain else None)}
# baseline chain distribution for comparison: all digit positions of all problems with carry-in
allc = []
for p in P:
    sb = schoolbook(p.a, p.b, 2)
    for k in range(len(p.answer)):
        if sb["c_in"][k]:
            L = 0
            while k - 1 - L >= 0 and sb["c_out"][k - 1 - L] == 1: L += 1
            allc.append(L)
res["all carry-in positions: chain share >= 3"] = float(np.mean(np.array(allc) >= 3))
print(res); save_json("t16c_binary_errors", res); print("DONE")
