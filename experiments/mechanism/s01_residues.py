"""S01 (stored activations only, CPU): can a number's residues in other bases be read linearly?

n mod M for an M that doesn't follow from the prompt's own digits would need a real conversion.
Probes are linear, tested on integers never seen in training, and compared with the same probe on one-hot
input digits. No model forward passes. The untrained-model control is deliberately not run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from core import BASES, save_json
from probe_lib import A_DIGIT, B_DIGIT, QUERY, Table, classify, input_onehot, integer_split

MODULI = [2, 3, 4, 5, 7, 8, 9, 10, 16]
SITES = {"a_0": (A_DIGIT, 0, ["a"]), "b_0": (B_DIGIT, 0, ["a", "b"]), "=": (QUERY, 0, ["a", "b", "a+b"]), "q1": (QUERY, 1, ["a", "b", "a+b"]), "q2": (QUERY, 2, ["a", "b", "a+b"])}
FEATS = ["resid0", "resid1", "mlp0", "mlp1"]
res = {"moduli": MODULI, "results": {}}
for base in BASES:
    t = Table(base)
    res["results"][base] = {}
    for site, (kind, j, targets) in SITES.items():
        rows = t.select(kind, j)
        a, b = t.meta["a"][rows], t.meta["b"][rows]
        X = {f: t.get(f, rows) for f in FEATS}
        X["input_onehot"] = input_onehot(a, b, base)
        res["results"][base][site] = {}
        for tn in targets:
            n = {"a": a, "b": b, "a+b": a + b}[tn]
            out = {}
            for seed in range(3):
                tr, te = integer_split(n, seed=seed)
                for f, x in X.items():
                    for M in MODULI:
                        r = classify(x[tr], (n % M)[tr], x[te], (n % M)[te], M)
                        out.setdefault(f, {}).setdefault(M, []).append((r["acc"], r["chance"], r["n_test"]))
            res["results"][base][site][tn] = {f: {M: {"acc": float(np.mean([v[0] for v in d])), "acc_sd": float(np.std([v[0] for v in d])), "chance": float(np.mean([v[1] for v in d])), "n_test": int(np.mean([v[2] for v in d]))} for M, d in m.items()} for f, m in out.items()}
            s = res["results"][base][site][tn]
            print(f"b{base} {site}/{tn}", {M: (round(max(s[f][M]['acc'] for f in FEATS), 3), round(s['input_onehot'][M]['acc'], 3), round(s['resid1'][M]['chance'], 3)) for M in MODULI}, flush=True)
    save_json("s01_residues", res)
print("DONE")
