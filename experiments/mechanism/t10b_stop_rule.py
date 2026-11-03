"""T10b (forward passes): does the stop decision follow the carry rule a_top + b_top - written?
Equal-length operands; after writing all L digits, the right action is EOS if no carry out, else '1'.
Regress logit(EOS) - logit('1') on a_top, b_top, written top digit; compare weights with the carry rule (E20)."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}; one = model.vocab["1"]
for b in (8, 10, 16):
    rng = random.Random(b * 17); S, rec = [], []
    while len(S) < 2000:
        a, c_ = rng.randint(b, 999), rng.randint(b, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) != len(sb): continue
        L = len(sa); ans = to_answer(a + c_, b)
        S.append("\n".join(render(*x, b) for x in sample_shots(rng, (a, c_))) + f"\n{sa}+{sb}={ans[:L]}")
        at, bt = a // b**(L - 1), c_ // b**(L - 1); w = int(ans[L - 1], 16); rec.append((at, bt, w, int(len(ans) > L), L))
    out = []
    for i in range(0, len(S), 250):
        seqs = [[model.vocab[x] for x in s] for s in S[i:i + 250]]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); lg, _ = model.forward(x.to(DEVICE)); r = lg[torch.arange(len(seqs), device=DEVICE), q]
        out.append(torch.stack([r[:, model.eos] - r[:, one], (r.argmax(-1) == model.eos).float()], 1).cpu().numpy())
    o = np.concatenate(out); rec = np.array(rec); at, bt, w, carry, L = rec.T; ev = o[:, 0]; stop = o[:, 1] == 1
    X = np.c_[at, bt, w, np.ones(len(at))].astype(float); co = np.linalg.lstsq(X, ev, rcond=None)[0]; r2 = 1 - ((ev - X @ co) ** 2).sum() / ((ev - ev.mean()) ** 2).sum()
    lin = at + bt - w; best = max((float(((lin < th) == stop).mean()), float(th)) for th in np.arange(-b, 2 * b) + .5)
    res[b] = {"weights_atop_btop_written": co[:3].tolist(), "r2": float(r2), "rule_accuracy": best[0], "threshold_over_base": best[1] / b, "model_correct": float((stop == (carry == 0)).mean()), "n": len(S)}
    print(b, {k: (np.round(v, 2).tolist() if isinstance(v, list) else round(v, 3)) for k, v in res[b].items()}, flush=True)
save_json("t10b_stop_rule", res); print("DONE")
