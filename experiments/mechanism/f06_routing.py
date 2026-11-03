"""F06 (forward passes): causal routing map. For each base and answer digit k, change ONE input token and find
which component at the query position carries that change to the output.

Variables: a_k, b_k, a_{k-1}, b_{k-1}, a_{k-2}, b_{k-2}, written o_{k-1}, written o_{k-2}.
Components patched at q_k only: L1H0..3 outputs (z1), the layer 0 output residual R (resid_pre1), L0H0..3 outputs (z0).
Metric: of trials where the full corruption changes the prediction, how often does patching ONLY this component do so?
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}
KMAX = {2: 5, 8: 3, 10: 2, 16: 2}
VARS = ["a_k", "b_k", "a_k-1", "b_k-1", "a_k-2", "b_k-2", "o_k-1", "o_k-2"]
COMPS = ["L1H0", "L1H1", "L1H2", "L1H3", "L1all", "R", "L0H0", "L0H1", "L0H2", "L0H3"]


def make_case(rng, base, k):
    """Prompt pieces with operands long enough; returns dict or None."""
    lo = base ** (k + 0) if k else 0
    a, b_ = rng.randint(lo, 999), rng.randint(lo, 999)
    if len(to_base(a, base)) <= k or len(to_base(b_, base)) <= k:
        return None
    return dict(a=a, b=b_, ctx="\n".join(render(*s, base) for s in sample_shots(rng, (a, b_))))


def strings(c, base, k):
    sa, sb = to_base(c["a"], base), to_base(c["b"], base)
    return sa, sb, to_answer(c["a"] + c["b"], base)[:k]


def corrupt(rng, base, sa, sb, ans, var, k):
    """Return (sa, sb, ans) with one token changed, or None if the variable doesn't exist."""
    sa, sb, ans = list(sa), list(sb), list(ans)
    if var.startswith(("a_", "b_")):
        s = sa if var[0] == "a" else sb
        off = {"k": 0, "k-1": 1, "k-2": 2}[var.split("_")[1]]
        j = k - off
        if j < 0 or j >= len(s):
            return None
        i = len(s) - 1 - j
        choices = [d for d in DIGITS[:base] if d != s[i] and not (i == 0 and d == "0" and len(s) > 1)]
        if not choices:
            return None
        s[i] = rng.choice(choices)
    else:
        off = {"o_k-1": 1, "o_k-2": 2}[var]
        j = k - off
        if j < 0 or j >= len(ans):
            return None
        ans[j] = rng.choice([d for d in DIGITS[:base] if d != ans[j]])
    return "".join(sa), "".join(sb), "".join(ans)


def ids(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]
    L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
    for i, s in enumerate(seqs): x[i, :len(s)] = torch.tensor(s)
    return x.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)


for base in BASES:
    res[base] = {}
    for k in range(KMAX[base] + 1):
        rng = random.Random(base * 100 + k)
        table = {}
        for var in VARS:
            C, D = [], []
            tries = 0
            while len(C) < 1200 and tries < 200000:
                tries += 1
                c = make_case(rng, base, k)
                if c is None: continue
                sa, sb, ans = strings(c, base, k)
                r = corrupt(rng, base, sa, sb, ans, var, k)
                if r is None: break
                C.append(f"{c['ctx']}\n{sa}+{sb}={ans}")
                D.append(f"{c['ctx']}\n{r[0]}+{r[1]}={r[2]}")
            if len(C) < 100:
                continue
            counts = {x: 0 for x in COMPS}; changed = 0; total = 0
            for i in range(0, len(C), 300):
                xc, q = ids(C[i:i + 300]); xd, _ = ids(D[i:i + 300])
                r = torch.arange(len(xc), device=DEVICE)
                clean_lg, cc = model.forward(xc, capture={"z1", "z0", "resid_pre1"})
                corr_lg, cd = model.forward(xd, capture={"z1", "z0", "resid_pre1"})
                p0 = clean_lg[r, q].argmax(-1); p1 = corr_lg[r, q].argmax(-1)
                ch = p0 != p1
                changed += ch.sum().item(); total += len(ch)
                for comp in COMPS:
                    if comp == "R":
                        def hook(t, s=cd["resid_pre1"]):
                            t = t.clone(); t[r, q] = s[r, q]; return t
                        hooks = {"resid_pre1": hook}
                    elif comp == "L1all":
                        def hook(t, s=cd["z1"]):
                            t = t.clone(); t[r, q] = s[r, q]; return t
                        hooks = {"z1": hook}
                    else:
                        L, h = int(comp[1]), int(comp[3])
                        def hook(t, s=cd[f"z{L}"], h=h):
                            t = t.clone(); t[r, q, h] = s[r, q, h]; return t
                        hooks = {f"z{L}": hook}
                    lg, _ = model.forward(xc, hooks=hooks)
                    pp = lg[r, q].argmax(-1)
                    counts[comp] += ((pp != p0) & ch).sum().item()
            table[var] = {"change_rate": changed / total, "n": total, **{x: counts[x] / max(changed, 1) for x in COMPS}}
        res[base][k] = table
        print(f"\nBASE {base} digit {k}")
        print("  var     chg   " + " ".join(f"{x:>5}" for x in COMPS))
        for var, t in table.items():
            print(f"  {var:7s} {t['change_rate']:.2f}  " + " ".join(f"{t[x]:5.2f}" for x in COMPS), flush=True)
    save_json("f06_routing", res)
print("DONE")
