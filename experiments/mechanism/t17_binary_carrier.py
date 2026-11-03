"""T17 (forward passes): which part of the test line carries the binary rule?
Test lines with digits 0/1 only whose last digits are 1 and 1 (binary writes 0, octal writes 2), under binary vs octal contexts.
Patch layer 0's output (resid_pre1) from one run into the other on one part of the test line at a time: a's digits, b's digits,
'+', '='. Then split what is patched into (i) position embedding only, (ii) layer 0 heads + MLP only (no embedding change)."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, Model, render, sample_shots, save_json, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {}
ZERO, TWO = model.vocab["0"], model.vocab["2"]
def ids(S):
    seqs = [[model.vocab[x] for x in s] for s in S]; n = max(map(len, seqs)); X = torch.full((len(S), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    return X.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
for lab, (lo, hi) in (("short operands (2-3 digits)", (2, 7)), ("medium operands (4-6 digits)", (9, 63))):
    rng = random.Random(lo * 7); Sb, So, parts = [], [], []
    while len(Sb) < 400:
        a, c_ = rng.randint(lo, hi), rng.randint(lo, hi); sa, sb = to_base(a, 2), to_base(c_, 2)
        if sa[-1] != "1" or sb[-1] != "1": continue
        test = f"{sa}+{sb}="; cb = "\n".join(render(*x, 2) for x in sample_shots(rng, (a, c_))); co = "\n".join(render(*x, 8) for x in sample_shots(rng, (a, c_)))
        if len(cb) + len(test) > 250: continue
        Sb.append(cb + "\n" + test); So.append(co + "\n" + test); parts.append((len(sa), len(sb)))
    Xb, qb = ids(Sb); Xo, qo = ids(So); J = torch.arange(len(Sb), device=DEVICE); la = torch.tensor([p[0] for p in parts], device=DEVICE); lb = torch.tensor([p[1] for p in parts], device=DEVICE)
    T = la + lb + 2  # test line length
    _, cb_ = model.forward(Xb, capture={"resid_pre1"}); _, co_ = model.forward(Xo, capture={"resid_pre1"})
    def offsets(part):  # offsets from the start of the test line
        if part == "a digits": return [(o, o < la) for o in range(int(la.max()))]
        if part == "b digits": return [(la + 1 + o, o < lb) for o in range(int(lb.max()))]
        if part == "+": return [(la, torch.ones_like(la, dtype=torch.bool))]
        if part == "=": return [(T - 1, torch.ones_like(la, dtype=torch.bool))]
        if part == "whole test line": return [(o, o < T) for o in range(int(T.max()))]
    def patch(part, src_c, src_q, dst_q, mode):
        def f(t):
            t = t.clone()
            for off, m in offsets(part):
                off = off if torch.is_tensor(off) else torch.full_like(la, off)
                jd = J[m]; pd = (dst_q - T + 1 + off)[m]; ps = (src_q - T + 1 + off)[m]
                if mode == "all": t[jd, pd] = src_c["resid_pre1"][jd, ps]
                elif mode == "position only": t[jd, pd] = t[jd, pd] + PE[ps] - PE[pd]
                elif mode == "layer 0 computation only": t[jd, pd] = src_c["resid_pre1"][jd, ps] - PE[ps] + PE[pd]
            return t
        return f
    def rule(X, q, hooks=None):
        lg, _ = model.forward(X, hooks=hooks); p = lg[J, q].argmax(-1); return (round(float((p == ZERO).float().mean()), 2), round(float((p == TWO).float().mean()), 2))
    out = {"binary context": rule(Xb, qb), "octal context": rule(Xo, qo)}
    for part in ("whole test line", "a digits", "b digits", "+", "="):
        for mode in ("all", "position only", "layer 0 computation only"):
            out[f"octal gets binary [{part}] ({mode})"] = rule(Xo, qo, {"resid_pre1": patch(part, cb_, qb, qo, mode)})
            out[f"binary gets octal [{part}] ({mode})"] = rule(Xb, qb, {"resid_pre1": patch(part, co_, qo, qb, mode)})
    res[lab] = out; print(lab); [print("  ", k, v) for k, v in out.items()]
save_json("t17_binary_carrier", res); print("DONE")
