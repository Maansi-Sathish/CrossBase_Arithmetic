"""T15b: separate length from value on large numbers. Teacher-forced accuracy at each digit position k for operands:
  in range   : both <= 999 (training)
  big, same length: some operand > 999 but its length in base b is no longer than 999's
  longer     : some operand longer than any training number
and, within big numbers, by whether the leading digit value occurred at that position in training.
Plus routing: at digit 1 for longer numbers, where does L1H3 look (target a_1/b_1 vs neighbours)?"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, schoolbook, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); tok = np.array([model.vocab[c] for c in DIGITS]); res = {}
def run(S, cap=None):
    seqs = [[model.vocab[x] for x in s] for s in S]; n = max(map(len, seqs)); X = torch.full((len(S), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); lg, c = model.forward(X.to(DEVICE), capture=cap or set()); J = torch.arange(len(S), device=DEVICE)
    return lg[J, q].argmax(-1).cpu().numpy(), c, q, J
for b in (8, 10, 16):
    Lmax = len(to_base(999, b)); topmax = int(to_base(999, b)[0], 16); rng = random.Random(b * 37); out = {}
    for k in range(4):
        cats = {"in range": [], "big, same length, leading digit seen": [], "big, same length, leading digit unseen": [], "longer": []}; att = {c: [] for c in cats}
        tries = 0
        while min(len(v) for v in cats.values()) < 300 and tries < 200000:
            tries += 1
            a, c_ = rng.randint(0, 4095), rng.randint(0, 4095)
            if a + c_ < b**k or len(to_base(min(a, c_), b)) <= k: continue
            la, lb = len(to_base(a, b)), len(to_base(c_, b))
            if a <= 999 and c_ <= 999: cat = "in range"
            elif max(la, lb) > Lmax: cat = "longer"
            else:
                tops = [int(to_base(v, b)[0], 16) for v in (a, c_) if len(to_base(v, b)) == Lmax]
                cat = "big, same length, leading digit unseen" if any(t > topmax for t in tops) else "big, same length, leading digit seen"
            if len(cats[cat]) >= 300: continue
            ans = to_answer(a + c_, b); s = "\n".join(render(*x, b) for x in sample_shots(rng, (a, c_))) + f"\n{to_base(a, b)}+{to_base(c_, b)}={ans[:k]}"
            if len(s) > 250: continue
            cats[cat].append((s, schoolbook(a, c_, b)["o"][k]))
        row = {}
        for cat, items in cats.items():
            if not items: continue
            pr, _, _, _ = run([x[0] for x in items]); row[cat] = round(float((pr == tok[np.array([x[1] for x in items])]).mean()), 2)
        out[k] = row; print(f"b{b} digit {k}:", row, flush=True)
    res[b] = out
save_json("t15b_large_numbers", res); print("DONE")
