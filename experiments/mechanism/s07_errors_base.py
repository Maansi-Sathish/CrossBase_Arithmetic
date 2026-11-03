"""S07 (bucket files, CPU, no forward passes): the model's own errors, carry chains, and base across the context.

A: exact accuracy by longest carry chain from the 10k stored generations (is a long chain harder?).
B: what the errors look like: first wrong digit index, carry at that digit, length errors.
C: 4-way base probe on stored L0/L1 residuals at each context newline, using prompts whose context so far is
   digit-ambiguous as a separate score. Tests whether newline tokens hold a running base summary.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from core import BASES, BUCKET, DIGITS, save_json, schoolbook, wilson
from probe_lib import Ridge, onehot

res = {"A": {}, "B": {}, "C": {}}
store = {i: {0: [], 1: []} for i in range(5)}
y = {i: [] for i in range(5)}
seen_max = {i: [] for i in range(5)}
idxs = {i: [] for i in range(5)}
for bi, base in enumerate(BASES):
    d = torch.load(BUCKET / f"base{base}.pt", map_location="cpu", weights_only=True, mmap=True)
    chains, kinds = {}, {"first_wrong_k": {}, "carry_in_at_first_wrong": 0, "length_only": 0, "wrong": 0, "off_alphabet": 0}
    for idx, r in enumerate(d["baseline"]):
        m = r["metadata"]
        sb = schoolbook(m["a"], m["b"], base)
        best = cur = 0
        for c in sb["c_out"]:
            cur = cur + 1 if c else 0
            best = max(best, cur)
        out = r["result"]["completion"].replace("<eos>", "")
        ok = out == m["answer"]
        e = chains.setdefault(best, [0, 0])
        e[0] += ok
        e[1] += 1
        if not ok:
            kinds["wrong"] += 1
            kinds["off_alphabet"] += any(ch not in DIGITS[:base] for ch in out)
            k = next((i for i, (x, z) in enumerate(zip(out, m["answer"])) if x != z), None)
            if k is None:
                kinds["length_only"] += 1
            else:
                kinds["first_wrong_k"][k] = kinds["first_wrong_k"].get(k, 0) + 1
                kinds["carry_in_at_first_wrong"] += sb["c_in"][k]
        # newline positions in the context
        g = r["result"]["geometry"]["eq_sign"]
        n = len(r["result"]["output_ids"])
        prompt = r["prompt"]
        pos = [i for i, ch in enumerate(prompt) if ch == "\n"]
        for L in (0, 1):
            full = g[L]["residual"].new_empty(0).set_(g[L]["residual"].untyped_storage(), 0, (n, 64))
            for i, p in enumerate(pos):
                store[i][L].append(full[p].numpy().copy())
        # highest digit visible so far: a surface summary that would explain any base read-out
        mx = 0
        at = 0
        for i, p in enumerate(pos):
            for ch in prompt[at:p]:
                if ch in DIGITS:
                    mx = max(mx, DIGITS.index(ch))
            at = p
            seen_max[i].append(mx)
            y[i].append(bi)
            idxs[i].append(idx)
    res["A"][base] = {k: (*wilson(*chains[k]), chains[k][1]) for k in sorted(chains)}
    res["B"][base] = kinds
    print("A", base, {k: (round(v[0], 3), v[3]) for k, v in res["A"][base].items()}, flush=True)
    print("B", base, kinds, flush=True)
    del d
for i in range(5):
    Y, I, M = np.array(y[i]), np.array(idxs[i]), np.array(seen_max[i])
    te = I % 5 == 0
    row = {"chance": 0.25}
    for L in (0, 1):
        X = np.stack(store[i][L])
        p = Ridge(X[~te])
        p.fit(onehot(Y[~te], 4))
        ok = p.predict(X[te]).argmax(1).cpu().numpy() == Y[te]
        row[f"L{L}"] = float(ok.mean())
        row[f"L{L}_per_base"] = [float(ok[Y[te] == b].mean()) for b in range(4)]
    # the surface rule: base from the largest digit seen so far
    rule = np.select([M <= 1, M <= 7, M <= 9], [0, 1, 2], 3)
    row["max_digit_rule"] = float((rule == Y).mean())
    res["C"][f"newline_{i + 1}"] = row
    print("C", i + 1, {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 3)) for k, v in row.items()}, flush=True)
save_json("s07_errors_base", res)
print("DONE")
