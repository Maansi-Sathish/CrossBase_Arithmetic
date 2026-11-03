"""T09 (forward passes): why does the digit-0 first-operand reader (L1H1) ignore b's digits?

Candidates, applied to b's digits only, alone and combined:
  tags   : give b's digits the position tags a's digits have (as if b sat where a is)
  plus   : block every layer 0 head at b's digits from attending to '+' (renormalised)
  content: give b's digits the layer 0 head outputs typical of a's digits
Measure L1H1 attention on b's digits and on a_0, and accuracy.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {}; tok = np.array([model.vocab[c] for c in DIGITS])
for b in BASES:
    rng = random.Random(b * 11); P = []
    while len(P) < 600:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) != len(sb) or len(sa) < 2: continue  # equal lengths so b's digits can take a's tags one-to-one
        ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_))); st = len(ctx) + 1
        P.append((f"{ctx}\n{sa}+{sb}=", st, st + len(sa), st + 2 * len(sa) + 1, a, c_))
    seqs = [[model.vocab[x] for x in p[0]] for p in P]; n = max(map(len, seqs)); X = torch.full((len(P), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    X = X.to(DEVICE); st, pl, eq = [torch.tensor([p[i] for p in P], device=DEVICE) for i in (1, 2, 3)]
    J = torch.arange(len(P), device=DEVICE); idx = torch.arange(n, device=DEVICE)[None]
    am = (idx >= st[:, None]) & (idx < pl[:, None]); bm = (idx > pl[:, None]) & (idx < eq[:, None])
    truth = tok[(np.array([p[4] for p in P]) + np.array([p[5] for p in P])) % b]
    _, c0 = model.forward(X, capture={"z0"}); za = c0["z0"][am].mean(0)
    pos = idx.expand(X.shape); shift = (pl + 1 - st)[:, None]
    tag_delta = torch.where(bm[..., None], PE[(pos - shift).clamp_min(0)] - PE[pos], torch.zeros_like(PE[pos]))
    def hooks(tags, plus, content):
        h = {}
        if tags: h["resid_pre0"] = lambda t: t + tag_delta
        if plus:
            def pat(t):
                t = t.clone(); keymask = (idx == pl[:, None])  # [B, n]
                rows = bm[:, None, :, None] & keymask[:, None, None, :]
                t = t.masked_fill(rows, 0.0); return t / t.sum(-1, keepdim=True).clamp_min(1e-9)
            h["pattern0"] = pat
        if content: h["z0"] = lambda t: torch.where(bm[..., None, None], za, t)
        return h
    out = {}
    for tags in (0, 1):
        for plus in (0, 1):
            for content in (0, 1):
                name = "+".join([nm for nm, f in (("tags", tags), ("plus", plus), ("content", content)) if f]) or "clean"
                lg, c = model.forward(X, hooks=hooks(tags, plus, content), capture={"pattern1"}); row = c["pattern1"][J, 1, eq]
                out[name] = {"H1_on_b": float((row * bm).sum(1).mean()), "H1_on_a0": float(row[J, pl - 1].mean()), "correct": float((lg[J, eq].argmax(-1).cpu().numpy() == truth).mean())}
    res[b] = out
    print(f"base {b}: L1H1 on b digits / on a0 / correct", {k: (round(v['H1_on_b'], 2), round(v['H1_on_a0'], 2), round(v['correct'], 2)) for k, v in out.items()}, flush=True)
save_json("t09_operand_identity", res); print("DONE")
