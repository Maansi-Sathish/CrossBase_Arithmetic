"""T14b: stop rate as a function of the final position's tag. 5-shot prompts, every tag shifted by D; also the training
distribution of final positions. And the position code in the residual alone: replace the final position's tag only."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {}
for b in (2, 8, 10, 16):
    rng = random.Random(b * 23); S = []
    while len(S) < 400:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); s = "\n".join(render(*x, b) for x in sample_shots(rng, (a, c_))) + f"\n{to_base(a, b)}+{to_base(c_, b)}={to_answer(a + c_, b)}"
        if len(s) < 240: S.append(s)
    seqs = [[model.vocab[x] for x in s] for s in S]; n = max(map(len, seqs)); X = torch.full((len(S), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    X = X.to(DEVICE); q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); J = torch.arange(len(S), device=DEVICE)
    pos_all, stop_all, pos_one, stop_one = [], [], [], []
    for D in range(-60, 61, 5):
        def f(t, D=D):
            idx = torch.arange(t.shape[1], device=DEVICE); return t + PE[(idx + D).clamp(0, 255)] - PE[idx]
        lg, _ = model.forward(X, hooks={"resid_pre0": f}); st = (lg[J, q].argmax(-1) == model.eos).cpu().numpy()
        pos_all += (q + D).clamp(0, 255).tolist(); stop_all += st.tolist()
        def g(t, D=D):  # only the final position's tag
            t = t.clone(); t[J, q] = t[J, q] + PE[(q + D).clamp(0, 255)] - PE[q]; return t
        lg, _ = model.forward(X, hooks={"resid_pre0": g}); st = (lg[J, q].argmax(-1) == model.eos).cpu().numpy()
        pos_one += (q + D).clamp(0, 255).tolist(); stop_one += st.tolist()
    def curve(p, s):
        p, s = np.array(p), np.array(s); bins = np.arange(0, 260, 10)
        return {int(lo): round(float(s[(p >= lo) & (p < lo + 10)].mean()), 2) for lo in bins if ((p >= lo) & (p < lo + 10)).sum() > 30}
    natural = (q.cpu().numpy())
    res[b] = {"all tags shifted": curve(pos_all, stop_all), "final tag only": curve(pos_one, stop_one), "natural final position 5-95%": [int(np.percentile(natural, 5)), int(np.percentile(natural, 95))]}
    print(f"b{b} natural final position 5-95%: {res[b]['natural final position 5-95%']}\n  stop by final position, all tags shifted: {res[b]['all tags shifted']}\n  final tag only: {res[b]['final tag only']}", flush=True)
save_json("t14b_stop_window", res); print("DONE")
