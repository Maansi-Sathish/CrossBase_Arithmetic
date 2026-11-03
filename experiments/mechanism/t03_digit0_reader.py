"""T03 (forward passes): how does L1H1 find a_0 (and L1H0 find b_0) at digit 0?

Hypotheses: (R) recency: the most recent digit of the right type wins, using position tags; (P) pointer to pos('+')-1;
(C) content: a position signal computed in layer 0 from the tokens themselves, not from tags.
A. Tag edits on a's digits only (keys), keeping '+' and the query untouched:
   shift all by d (order kept): R and C predict no change; P predicts a move.
   reverse their order: R predicts attention moves to a's first digit; C predicts little change.
   flatten (same tag for all): R predicts spreading; C predicts little change.
B. Key content: mean-ablate each L0 head at a's digits. Which head do keys need?
C. Type: mean-ablate each L0 head at b's digits. Does L1H1 start reading b's digits?
D. Query: mean-ablate each L0 head at '='. Which head does the query need?
Same for L1H0 and b's digits.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {}
tok = np.array([model.vocab[c] for c in DIGITS])
for b in BASES:
    rng = random.Random(b * 3 + 1); P = []
    while len(P) < 600:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) < 3 or len(sb) < 3: continue
        ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_))); st = len(ctx) + 1
        P.append((f"{ctx}\n{sa}+{sb}=", st, st + len(sa), st + len(sa) + 1 + len(sb), a, c_))
    seqs = [[model.vocab[x] for x in p[0]] for p in P]; n = max(map(len, seqs)); X = torch.full((len(P), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    X = X.to(DEVICE); st, pl, eq = [torch.tensor([p[i] for p in P], device=DEVICE) for i in (1, 2, 3)]
    A_ = np.array([p[4] for p in P]); B_ = np.array([p[5] for p in P]); truth = tok[(A_ + B_) % b]
    J = torch.arange(len(P), device=DEVICE); idx = torch.arange(n, device=DEVICE)[None]
    amask = (idx >= st[:, None]) & (idx < pl[:, None]); bmask = (idx > pl[:, None]) & (idx < eq[:, None]); qmask = idx == eq[:, None]
    _, c0 = model.forward(X, capture={"z0"}); zmean = {r: c0["z0"][m].mean(0) for r, m in (("a", amask), ("b", bmask), ("q", qmask))}
    def measure(hooks):
        out = {"H1_a0": 0., "H1_a_first": 0., "H1_on_b": 0., "H0_b0": 0., "H0_on_a": 0., "correct": 0.}
        for i in range(0, len(P), 150):
            sl = slice(i, i + 150); x = X[sl]; jj = torch.arange(x.shape[0], device=DEVICE)
            h = hooks(sl) if hooks else None
            lg, c = model.forward(x, hooks=h, capture={"pattern1"}); pat = c["pattern1"]; e = eq[sl]; p_ = pl[sl]; s_ = st[sl]
            row1, row0 = pat[jj, 1, e], pat[jj, 0, e]
            am, bm = amask[sl], bmask[sl]
            out["H1_a0"] += row1[jj, p_ - 1].sum().item(); out["H1_a_first"] += row1[jj, s_].sum().item(); out["H1_on_b"] += (row1 * bm).sum().item()
            out["H0_b0"] += row0[jj, e - 1].sum().item(); out["H0_on_a"] += (row0 * am).sum().item()
            out["correct"] += (lg[jj, e].argmax(-1).cpu().numpy() == truth[sl]).sum()
        return {k: float(v / len(P)) for k, v in out.items()}
    def tag_edit(kind, d=0, region="a"):
        m = amask if region == "a" else bmask
        lo, hi = (st, pl) if region == "a" else (pl + 1, eq)
        def f(sl):
            pos = idx.expand(X.shape)[sl].clone(); mm = m[sl]
            if kind == "shift": newp = pos + d
            elif kind == "reverse": newp = (lo[sl][:, None] + hi[sl][:, None] - 1 - pos)
            elif kind == "flatten": newp = ((lo[sl] + hi[sl] - 1) // 2)[:, None].expand_as(pos)
            newp = torch.where(mm, newp, pos).clamp(0, 255)
            delta = PE[newp] - PE[pos]
            return {"resid_pre0": lambda t, delta=delta: t + delta}
        return f
    def head_ablate(h, region):
        m = {"a": amask, "b": bmask, "q": qmask}[region]
        def f(sl):
            def fn(t, mm=m[sl]):
                t = t.clone(); t[:, :, h][mm] = zmean[region][h]; return t
            return {"z0": fn}
        return f
    r = {"clean": measure(None)}
    for d in (-8, -4, -2, -1, 1, 2):
        r[f"a tags shift {d:+d}"] = measure(tag_edit("shift", d))
    r["a tags reversed"] = measure(tag_edit("reverse")); r["a tags flattened"] = measure(tag_edit("flatten"))
    r["b tags reversed"] = measure(tag_edit("reverse", region="b")); r["b tags flattened"] = measure(tag_edit("flatten", region="b"))
    for h in range(4):
        r[f"L0H{h} ablated at a digits"] = measure(head_ablate(h, "a"))
        r[f"L0H{h} ablated at b digits"] = measure(head_ablate(h, "b"))
        r[f"L0H{h} ablated at '='"] = measure(head_ablate(h, "q"))
    res[b] = r
    print(f"\nbase {b}: H1 on a0 | H1 on a's first digit | H1 on b | H0 on b0 | H0 on a | correct")
    for k, v in r.items():
        print(f"   {k:28s} {v['H1_a0']:.2f} {v['H1_a_first']:.2f} {v['H1_on_b']:.2f} | {v['H0_b0']:.2f} {v['H0_on_a']:.2f} | {v['correct']:.2f}", flush=True)
    save_json("t03_digit0_reader", res)
print("DONE")
