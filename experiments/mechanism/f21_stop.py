"""F21 (stored + forward passes): the stop circuit, and the unseen-magnitude gap.

A. Stop evidence and neuron 177 by state: digit index relative to both operand lengths, carry.
B. Linear account of 177's input: operands exhausted, carry.
C. Does the carry vector control stopping? Add it to stop rows (should write a digit), subtract it from final-carry rows (should stop).
D. What do heads read when an operand has run out? Attention at the query for exhausted operands.
E. Unseen magnitudes in hex and octal: what type of error produces low exact accuracy with high digit accuracy?
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, Problem, gelu_new, greedy, sample_shots, save_json, to_answer, to_base
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]; Wc = sd["h.1.mlp.c_fc.weight"]; bc = sd["h.1.mlp.c_fc.bias"]
L = lambda mid, pre: model.unembed(mid + gelu_new(pre) @ Wo + bo)
for b in BASES:
    t = Table(b); m = t.meta; q = np.where((m["kind"] == QUERY) & (m["correct"] == 1))[0]
    A, B, k = m["a"][q].astype(np.int64), m["b"][q].astype(np.int64), m["j"][q].astype(np.int64)
    la = np.array([len(np.base_repr(int(x), b)) for x in A]); lb = np.array([len(np.base_repr(int(x), b)) for x in B]); M = b**k
    c = ((A % M + B % M) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", q), device=DEVICE); Z = torch.as_tensor(t.get("z1", q), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]; pre = model._ln("h.1.ln_2", mid) @ Wc + bc
    post = gelu_new(pre).cpu().numpy()[:, 177]; lg = model.unembed(mid + gelu_new(pre) @ Wo + bo)
    ev = (lg[:, model.eos] - lg[:, torch.as_tensor(tok, device=DEVICE)].max(1).values).cpu().numpy()
    nl = model.vocab["\n"]
    exh = (k >= la).astype(int) + (k >= lb).astype(int)
    state = {}
    for e_, c_ in [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]:
        s = (exh == e_) & (c == c_)
        if s.sum() >= 5: state[f"exhausted {e_}, carry {c_}"] = {"n": int(s.sum()), "n177": float(post[s].mean()), "stop_evidence": float(ev[s].mean()), "stops": float((ev[s] > 0).mean())}
    res["A"][b] = state
    print(f"A b{b}:", {a2: (v['n'], round(v['n177'], 2), round(v['stop_evidence'], 1), round(v['stops'], 2)) for a2, v in state.items()}, flush=True)
    # B linear account
    p177 = pre.cpu().numpy()[:, 177]; X = np.c_[np.ones(len(k)), exh == 1, exh == 2, c, (exh == 2) * c].astype(float); co = np.linalg.lstsq(X, p177, rcond=None)[0]
    res["B"][b] = {"coefs: const, one exhausted, both exhausted, carry, both x carry": co.tolist(), "r2": float(1 - ((p177 - X @ co) ** 2).sum() / ((p177 - p177.mean()) ** 2).sum())}
    print("B", b, np.round(co, 2).tolist(), "R2", round(res["B"][b]["r2"], 3), flush=True)
    # C carry vector on stop and final-carry rows, learned from digit rows with both digits present
    dig = (exh == 0) & (k >= 1); ds = []; ak, bk = (A // M) % b, (B // M) % b
    for i in range(b):
        for j in range(b):
            m0 = dig & (ak == i) & (bk == j) & (c == 0); m1 = dig & (ak == i) & (bk == j) & (c == 1)
            if m0.sum() >= 2 and m1.sum() >= 2: ds.append(pre[torch.as_tensor(m1, device=DEVICE)].mean(0) - pre[torch.as_tensor(m0, device=DEVICE)].mean(0))
    v = torch.stack(ds).mean(0)
    sr = torch.as_tensor((exh == 2) & (c == 0), device=DEVICE); cr = torch.as_tensor((exh == 2) & (c == 1), device=DEVICE); out = {}
    for al in (.5, 1., 1.5):
        ps = L(mid[sr], pre[sr] + al * v).argmax(-1); pc = L(mid[cr], pre[cr] - al * v).argmax(-1)
        out[al] = {"stop_rows_still_stop": float((ps == model.eos).float().mean()), "stop_rows_write_1": float((ps == model.vocab["1"]).float().mean()), "final_carry_rows_write_1": float((pc == model.vocab["1"]).float().mean()), "final_carry_rows_stop": float((pc == model.eos).float().mean())}
    res["C"][b] = out
    print("C", b, {a2: {k2: round(x, 2) for k2, x in r.items()} for a2, r in out.items()}, flush=True)
# D attention on exhausted operands
for b in (10, 16, 2):
    rng = random.Random(b); tally = {}
    for _ in range(1500):
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b); ans = to_answer(a + c_, b); ctx = "\n".join(render_ for render_ in [])
        from core import render
        ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_)))
        k = rng.randint(0, len(ans)); s = f"{ctx}\n{sa}+{sb}={ans[:k]}"; c0 = len(ctx) + 1; pl = c0 + len(sa); eq = pl + 1 + len(sb); q = len(s) - 1
        x = torch.tensor([[model.vocab[ch] for ch in s]], device=DEVICE); _, cc = model.forward(x, capture={"pattern1"}); P = cc["pattern1"][0].cpu().numpy()
        ex = int(k >= len(sa)) + int(k >= len(sb)); key = (min(k, 3), ex)
        e = tally.setdefault(key, {"n": 0, "digits_a": np.zeros(4), "digits_b": np.zeros(4), "plus": np.zeros(4), "eq": np.zeros(4), "answer": np.zeros(4), "context": np.zeros(4), "newline": np.zeros(4)})
        e["n"] += 1
        for h in range(4):
            row = P[h, q]
            e["digits_a"][h] += row[c0:pl].sum(); e["digits_b"][h] += row[pl + 1:eq].sum(); e["plus"][h] += row[pl]; e["eq"][h] += row[eq]; e["answer"][h] += row[eq + 1:q + 1].sum(); e["context"][h] += row[:c0 - 1].sum(); e["newline"][h] += row[c0 - 1]
    res["D"][b] = {f"k{k}/exhausted{x}": {a2: (v / tally[(k, x)]["n"]).round(2).tolist() if isinstance(v, np.ndarray) else v for a2, v in tally[(k, x)].items()} for k, x in sorted(tally)}
    for key, v in res["D"][b].items():
        print("D", b, key, {a2: x for a2, x in v.items()}, flush=True)
save_json("f21_stop", res)
# E unseen magnitudes
for b, lo, hi, lab in [(16, 1000, 2047, "hex 1000-2047, sum <= FFF"), (8, 1000, 2047, "octal 1000-2047, sum <= 7777"), (16, 256, 999, "hex seen range"), (10, 1000, 2047, "decimal 1000-2047 (new length)")]:
    rng = random.Random(b + lo); P = []
    while len(P) < 3000:
        a, c_ = rng.randint(lo, hi), rng.randint(lo, hi)
        if b in (8, 16) and a + c_ >= 4096: continue
        P.append(Problem(a, c_, b, sample_shots(rng, (a, c_))))
    out = greedy(model, P, bs=256); kinds = {"correct": 0, "digits right then extra output": 0, "too short": 0, "wrong digit": 0}; wk = {}
    for o, p in zip(out, P):
        ans = p.answer
        if o == ans: kinds["correct"] += 1
        elif o.startswith(ans): kinds["digits right then extra output"] += 1
        elif ans.startswith(o): kinds["too short"] += 1
        else:
            kinds["wrong digit"] += 1; k = next(i for i, (x, y) in enumerate(zip(o, ans)) if x != y); wk[k] = wk.get(k, 0) + 1
    ex = [(to_base(p.a, b), to_base(p.b, b), p.answer, o) for o, p in zip(out, P) if o != p.answer][:4]
    res["E"][lab] = {"kinds": {a2: v / len(P) for a2, v in kinds.items()}, "first_wrong_digit_index": {int(a2): v / len(P) for a2, v in sorted(wk.items())}, "examples": ex}
    print("E", lab, {a2: round(v, 3) for a2, v in res["E"][lab]["kinds"].items()}, "first wrong at", {a2: round(v, 3) for a2, v in res["E"][lab]["first_wrong_digit_index"].items()}, "ex", ex[:3], flush=True)
save_json("f21_stop", res); print("DONE")
