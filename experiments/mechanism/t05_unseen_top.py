"""T05 (forward passes): why does hex give a spurious final '1' on unseen magnitudes?

Hypothesis: digit tables are position-specific. In hex training, digit position 2 only ever held operand digits 0-3 and sum digits 0-7.
So the stop/carry decision at the query after 3 digits sees unfamiliar values. The same values at position 1 were all seen in training.
Prompts have no carry into the top digit, so the right action after the last digit is always to stop.
A. Hex: stop after the 3rd digit, by top digits (a2, b2). Control: 2-digit numbers, stop after the 2nd digit, same digit values.
B. Octal: 4-digit numbers, by top digits (training saw a3 in {0,1}).
C. Which variable predicts the error: operand top digit values or the written top digit?
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}
def run(S):
    out = []
    for i in range(0, len(S), 256):
        seqs = [[model.vocab[c] for c in s] for s in S[i:i + 256]]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); lg, _ = model.forward(x.to(DEVICE)); out.append(lg[torch.arange(len(seqs), device=DEVICE), q].argmax(-1).cpu().numpy())
    return np.concatenate(out)
def grid(base, top_pos, n_per=40, seed=0):
    """Stop rate after writing top_pos+1 digits, with no carry into or out of the top digit."""
    rng = random.Random(seed); tab = np.full((base, base), np.nan); written = {}
    for x in range(1, base):
        for y in range(1, base):
            if x + y >= base: continue
            S = []
            for _ in range(n_per):
                lowa = [rng.randrange(base) for _ in range(top_pos)]; lowb = [rng.randrange(base - lowa[i]) for i in range(top_pos)]  # no carries anywhere
                a = x * base**top_pos + sum(d * base**i for i, d in enumerate(lowa)); b = y * base**top_pos + sum(d * base**i for i, d in enumerate(lowb))
                ans = to_answer(a + b, base)
                S.append("\n".join(render(*s, base) for s in sample_shots(rng, (a, b))) + f"\n{to_base(a, base)}+{to_base(b, base)}={ans}")
            pr = run(S); tab[x, y] = float((pr == model.eos).mean())
    return tab
for base, pos, lab in ((16, 2, "hex, 3-digit numbers (stop after 3rd digit)"), (16, 1, "hex, 2-digit numbers (control)"), (8, 3, "octal, 4-digit numbers"), (8, 2, "octal, 3-digit numbers (control)"), (10, 2, "decimal, 3-digit numbers")):
    t = grid(base, pos, seed=base + pos)
    res[lab] = t.tolist()
    seen_max = {(16, 2): 3, (16, 1): 15, (8, 3): 1, (8, 2): 7, (10, 2): 9}[(base, pos)]
    xs, ys = np.meshgrid(np.arange(base), np.arange(base), indexing="ij"); ok = ~np.isnan(t)
    both_seen = ok & (xs <= seen_max) & (ys <= seen_max); unseen = ok & ((xs > seen_max) | (ys > seen_max))
    wsum = xs + ys
    print(f"{lab}: stop rate where both top digits were seen at this position {np.nanmean(t[both_seen]) if both_seen.any() else float('nan'):.2f}, where at least one wasn't {np.nanmean(t[unseen]) if unseen.any() else float('nan'):.2f}", flush=True)
    if unseen.any():
        by_w = {int(w): round(float(np.nanmean(t[ok & (wsum == w)])), 2) for w in range(2, base)}
        by_max = {int(m): round(float(np.nanmean(t[ok & (np.maximum(xs, ys) == m)])), 2) for m in range(1, base)}
        print(f"   stop rate by top sum digit written: {by_w}\n   by larger top operand digit: {by_max}", flush=True)
        res[lab + " by written"] = by_w; res[lab + " by max operand"] = by_max
save_json("t05_unseen_top", res); print("DONE")
