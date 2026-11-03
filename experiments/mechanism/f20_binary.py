"""F20 (forward passes + stored): the remaining binary questions.

A. Digit 1: is carry just a_0 AND b_0, with no need for the written digit? Additive theory variants, and rules on random inputs.
B. Binary vs octal, corrected: short test lines, so operand length isn't out of range. How many tier-2 digits flip the rule?
C. Errors at the end of the shorter operand: at that digit, does attention miss, and does the error match a specific misreading?
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, render, sample_shots, save_json, schoolbook, to_answer, to_base
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]
def ids(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
    return x.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
def pred(strs, bs=256):
    out = []
    for i in range(0, len(strs), bs):
        x, q = ids(strs[i:i + bs]); lg, _ = model.forward(x); out.append(lg[torch.arange(len(x), device=DEVICE), q].cpu().numpy())
    return np.concatenate(out)
# A rules
for k in (1, 2, 3):
    rng = random.Random(k); S, rec = [], []
    while len(S) < 4000:
        hi = rng.randint(1, 60); hj = rng.randint(1, 60); ap, bp, w = rng.randrange(2), rng.randrange(2), rng.randrange(2); low = rng.randrange(2**k) if k > 1 else 0
        A = (hi << k) + (ap << (k - 1)) + (rng.randrange(2 ** (k - 1)) if k > 1 else 0); B = (hj << k) + (bp << (k - 1)) + (rng.randrange(2 ** (k - 1)) if k > 1 else 0)
        pref = to_base(A + B, 2)[::-1][:k - 1] + str(w)
        S.append("\n".join(render(*s, 2) for s in sample_shots(rng, (A, B))) + f"\n{to_base(A, 2)}+{to_base(B, 2)}={pref}")
        cin = ((A % 2 ** (k - 1)) + (B % 2 ** (k - 1))) >= 2 ** (k - 1) if k > 1 else False
        rec.append((ap, bp, w, (A >> k & 1) + (B >> k & 1), int(cin)))
    L = pred(S); rec = np.array(rec); ap, bp, w, cur, cin = rec.T; pr = L.argmax(1); J = np.arange(len(L))
    ch1 = pr == tok[(cur + 1) % 2]; ev = L[J, tok[(cur + 1) % 2]] - L[J, tok[cur % 2]]
    rules = {"a AND b (operands only)": (ap + bp) >= 2, "a+b-w > 0.5": (ap + bp - w) > .5, "a OR b": (ap + bp) >= 1, "written=0": w == 0, "true carry from real lower digits": ((ap + bp + cin) >= 2)}
    X = np.c_[ap, bp, w, np.ones(len(ap))].astype(float); co = np.linalg.lstsq(X, ev, rcond=None)[0]
    tab = {}
    for x, y, z in [(p, q2, r) for p in (0, 1) for q2 in (0, 1) for r in (0, 1)]:
        m = (ap == x) & (bp == y) & (w == z); tab[f"a{x} b{y} w{z}"] = float(ch1[m].mean())
    res["A"][f"rules_k{k}"] = {"rules": {a2: float((v == ch1).mean()) for a2, v in rules.items()}, "weights": co[:3].tolist(), "carry_rate_by_cell": tab}
    print(f"A k{k} weights a,b,w {np.round(co[:3], 2).tolist()}; rule match {({a2: round(float((v == ch1).mean()), 3) for a2, v in rules.items()})}\n   carry rate by (a,b,w): {({a2: round(v, 2) for a2, v in tab.items()})}", flush=True)
def load(k):
    t = Table(2); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= 2**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = 2**k; R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]; pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    pm = 2 ** (k - 1) if k else 1
    return dict(a=(A // M) % 2, b=(B // M) % 2, ap=(A // pm) % 2, bp=(B // pm) % 2, w=((A + B) // pm) % 2, c=((A % M + B % M) >= M).astype(int), pre=pre.cpu().numpy(), mid=mid.cpu().numpy(), idx=t.meta["idx"][rows])
for k in (1, 2, 3):
    d = load(k); tr = d["idx"] % 2 == 0; te = ~tr; i = np.ones(len(d["a"]))
    v = {"a,b,c": [i, d["a"], d["b"], d["c"]], "a,b,prev a,prev b": [i, d["a"], d["b"], d["ap"], d["bp"]], "a,b,prev a,prev b,written": [i, d["a"], d["b"], d["ap"], d["bp"], d["w"]], "a,b,written": [i, d["a"], d["b"], d["w"]], "a,b,c + prev": [i, d["a"], d["b"], d["c"], d["ap"], d["bp"]]}
    for nm, cols in v.items():
        X = np.column_stack(cols).astype(np.float32); co = np.linalg.lstsq(X[tr], d["pre"][tr], rcond=None)[0]; p = X[te] @ co
        lg = model.unembed(torch.as_tensor(d["mid"][tr].mean(0), device=DEVICE) + gelu_new(torch.as_tensor(p, device=DEVICE)) @ Wo + bo).argmax(-1).cpu().numpy()
        res["A"][f"theory_k{k}/{nm}"] = float((lg == tok[(d["a"][te] + d["b"][te] + d["c"][te]) % 2]).mean())
    print(f"A theory k{k}:", {nm: round(res['A'][f'theory_k{k}/{nm}'], 3) for nm in v}, flush=True)
# B
def short(rng):
    a = "1" + "".join(rng.choice("01") for _ in range(rng.randint(0, 2))) + "1"; b = "1" + "".join(rng.choice("01") for _ in range(rng.randint(0, 2))) + "1"; return a, b
def rate(S): 
    pr = pred(S).argmax(1); return float((pr == model.vocab["0"]).mean()), float((pr == model.vocab["2"]).mean())
for B in range(2, 9):
    rng = random.Random(B); S = []
    for _ in range(800):
        a, b = short(rng); S.append("\n".join(render(*s, B) for s in sample_shots(rng, (-1, -1))) + f"\n{a}+{b}=")
    res["B"][f"context base {B}"] = rate(S); print("B context base", B, "(rule 2, rule >=3):", np.round(res["B"][f"context base {B}"], 3).tolist(), flush=True)
for d in "27":
    for n in (0, 1, 2, 4, 8, 16, 32):
        rng = random.Random(n * 7 + ord(d)); S = []
        for _ in range(600):
            a, b = short(rng); ctx = list("\n".join(render(*s, 2) for s in sample_shots(rng, (-1, -1)))); pos = [j for j, ch in enumerate(ctx) if ch in "01"]
            for j in rng.sample(pos, min(n, len(pos))): ctx[j] = d
            S.append("".join(ctx) + f"\n{a}+{b}=")
        res["B"][f"binary ctx, {n} tokens -> {d}"] = rate(S)
    print(f"B binary context, n tokens changed to {d} (rule 2, rule>=3):", {n: np.round(res['B'][f'binary ctx, {n} tokens -> {d}'], 2).tolist() for n in (0, 1, 2, 4, 8, 16, 32)}, flush=True)
# C errors
rng = random.Random(55); P = []
while len(P) < 5000:
    a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, 2), to_base(c_, 2)
    if abs(len(sa) - len(sb)) < 5: continue
    P.append((a, c_, sa, sb, "\n".join(render(*s, 2) for s in sample_shots(rng, (a, c_)))))
stat = {}
for i in range(0, len(P), 128):
    ch = P[i:i + 128]; seqs = []
    for a, c_, sa, sb, ctx in ch: seqs.append(f"{ctx}\n{sa}+{sb}={to_answer(a + c_, 2)}")
    x, _ = ids(seqs); lg, c = model.forward(x, capture={"pattern1"}); pat = c["pattern1"].cpu().numpy(); pr = lg.argmax(-1).cpu().numpy()
    for j, (a, c_, sa, sb, ctx) in enumerate(ch):
        c0 = len(ctx) + 1; pl = c0 + len(sa); eq = pl + 1 + len(sb); ans = to_answer(a + c_, 2); sb_short = len(sb) <= len(sa); sl = min(len(sa), len(sb)); head = 0 if sb_short else 1; last = (eq if sb_short else pl) - 1
        sc = schoolbook(a, c_, 2)
        for k in (sl - 1, sl, sl + 1):
            if k >= len(ans) or k < 0: continue
            q = eq + k; wrong = pr[j, q] != model.vocab[ans[k]]
            target = (eq - 1 - k if sb_short else pl - 1 - k)
            key = (k - sl, bool(wrong))
            e = stat.setdefault(key, {"n": 0, "mass_target": 0., "mass_plus": 0., "mass_last_digit": 0., "alt_carry": 0})
            e["n"] += 1
            e["mass_last_digit"] += pat[j, head, q, last]
            e["mass_plus"] += pat[j, head, q, pl if sb_short else c0 - 1]
            if k < sl: e["mass_target"] += pat[j, head, q, target]
            flip = str((int(ans[k]) + 1) % 2); e["alt_carry"] += pr[j, q] == model.vocab[flip]
res["C"] = {f"k-short={k_}/wrong={w_}": {a2: (v[a2] / v["n"] if a2 != "n" else v["n"]) for a2 in v} for (k_, w_), v in sorted(stat.items())}
for key, v in res["C"].items(): print("C", key, {a2: (round(x, 3) if isinstance(x, float) else x) for a2, x in v.items()}, flush=True)
save_json("f20_binary", res); print("DONE")
