"""T10 (forward passes): the route of the stop decision.

Pairs with equal operand lengths: clean has no final carry (model should stop after the last digit); corrupted changes only a's
top digit so the sum carries out (model should write '1'). Same written prefix length L; the written top digit differs.
At the query after L digits, patch one component from corrupted into clean (and back), and count how often the decision flips.
Components: L1H0..H3 output, layer 0 output at the query (R), L0 heads at the query, the written top digit token's layer 0 output,
and layer 0 output at a's top digit.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}
for b in (8, 10, 16):  # binary: equal-length operands always carry out, so no clean pairs exist
    rng = random.Random(b * 13); C, Dd, info = [], [], []
    tries = 0
    while len(C) < 500 and tries < 200000:
        tries += 1
        a, c_ = rng.randint(b, 999), rng.randint(b, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) != len(sb): continue
        L = len(sa); top = b ** (L - 1)
        if a + c_ >= b**L: continue  # clean must not carry out
        # raise a's top digit so the corrupted sum carries out, keeping a's length
        need = b**L - (a + c_); add = -(-need // top) * top
        a2 = a + add
        if a2 >= b**L: continue
        ans, ans2 = to_answer(a + c_, b), to_answer(a2 + c_, b)
        if len(ans) != L or len(ans2) != L + 1: continue
        ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_)))
        st = len(ctx) + 1
        C.append(f"{ctx}\n{sa}+{sb}={ans}"); Dd.append(f"{ctx}\n{to_base(a2, b)}+{sb}={ans2[:L]}"); info.append((st, st + L, st + 2 * L + 1))
    seqs = [[model.vocab[x] for x in s] for s in C]; seqd = [[model.vocab[x] for x in s] for s in Dd]; n = max(map(len, seqs + seqd))
    Xc = torch.full((len(C), n), model.pad, dtype=torch.long); Xd = Xc.clone()
    for j, (s1, s2) in enumerate(zip(seqs, seqd)): Xc[j, :len(s1)] = torch.tensor(s1); Xd[j, :len(s2)] = torch.tensor(s2)
    Xc, Xd = Xc.to(DEVICE), Xd.to(DEVICE); q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); J = torch.arange(len(C), device=DEVICE)
    st, pl, eq = [torch.tensor([x[i] for x in info], device=DEVICE) for i in range(3)]
    names = {"z1", "z0", "resid_pre1"}
    lc, cc = model.forward(Xc, capture=names); ld, cd = model.forward(Xd, capture=names)
    stop_c = lc[J, q].argmax(-1) == model.eos; stop_d = ld[J, q].argmax(-1) == model.eos
    valid = stop_c & ~stop_d
    out = {"clean_stops": float(stop_c.float().mean()), "corrupted_continues": float((~stop_d).float().mean()), "n_valid": int(valid.sum())}
    def run(X, src, dirn):
        rows = {}
        comps = [(f"L1H{h}", "z1", h, "q") for h in range(4)] + [("all L1 heads", "z1", None, "q"), ("layer 0 output at query", "resid_pre1", None, "q")] + [(f"L0H{h} at query", "z0", h, "q") for h in range(4)] + [("layer 0 output at a's top digit", "resid_pre1", None, "atop"), ("layer 0 output at all of a's digits", "resid_pre1", None, "aall")]
        for nm, hook, h, where in comps:
            pos = {"q": q, "atop": st, "aall": None}[where]
            def fn(t, s=src[hook], h=h, pos=pos):
                t = t.clone()
                if pos is None:
                    m = (torch.arange(t.shape[1], device=DEVICE)[None] >= st[:, None]) & (torch.arange(t.shape[1], device=DEVICE)[None] < pl[:, None]); t[m] = s[m]
                elif h is None: t[J, pos] = s[J, pos]
                else: t[J, pos, h] = s[J, pos, h]
                return t
            lg, _ = model.forward(X, hooks={hook: fn}); p = lg[J, q].argmax(-1) == model.eos
            flipped = (~p if dirn == "clean_gets_corrupted" else p) & valid
            rows[nm] = float(flipped.sum() / valid.sum().clamp_min(1))
        return rows
    out["clean_gets_corrupted (stops -> continues)"] = run(Xc, cd, "clean_gets_corrupted")
    out["corrupted_gets_clean (continues -> stops)"] = run(Xd, cc, "corrupted_gets_clean")
    res[b] = out
    print(f"base {b}: clean stops {out['clean_stops']:.2f}, corrupted continues {out['corrupted_continues']:.2f}, pairs {out['n_valid']}")
    for k in ("clean_gets_corrupted (stops -> continues)", "corrupted_gets_clean (continues -> stops)"):
        print("  ", k, {a: round(v, 2) for a, v in out[k].items()}, flush=True)
save_json("t10_stop_route", res); print("DONE")
