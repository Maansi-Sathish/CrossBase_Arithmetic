"""T21b: split each step's digit logits (relative to the average decimal digit) into embeddings + layer 0, layer 1 heads, layer 1 MLP.
Also L0H1's attention at '=' on the test line's own tokens."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, Problem, sample_shots, save_json
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {}
tok = torch.tensor([model.vocab[c] for c in DIGITS[:10]] + [model.eos, model.vocab["\n"]], device=DEVICE)
U = model.WE.T * sd["ln_f.weight"][:, None]
p = Problem(8, 7, 10, sample_shots(random.Random(0), (8, 7))); P0 = p.prompt
for lab, s in (("digit 0 (after '=')", P0), ("digit 1 (after '5')", P0 + "5"), ("stop (after '51')", P0 + "51")):
    X = torch.tensor([[model.vocab[x] for x in s]], device=DEVICE); lg, c = model.forward(X, capture={"resid_pre1", "attn_out1", "mlp_out1", "resid_post1", "z1", "pattern0", "pattern1"}); q = X.shape[1] - 1
    R = c["resid_post1"][0, q]; scale = (R - R.mean()).pow(2).mean().add(1e-5).sqrt()
    out = {}
    for nm, v in (("embeddings + layer 0", c["resid_pre1"][0, q]), ("layer 1 heads", c["attn_out1"][0, q]), ("layer 1 MLP", c["mlp_out1"][0, q])):
        l = ((v - v.mean()) / scale) @ U[:, tok]; out[nm] = l.cpu().numpy()
    Wp = sd["h.1.attn.c_proj.weight"]
    for h in range(4):
        v = c["z1"][0, q, h] @ Wp[16 * h:16 * h + 16]; out[f"  L1H{h}"] = (((v - v.mean()) / scale) @ U[:, tok]).cpu().numpy()
    names = list(DIGITS[:10]) + ["EOS", "newline"]
    tgt = {"digit 0 (after '=')": "5", "digit 1 (after '5')": "1", "stop (after '51')": "EOS"}[lab]; j = names.index(tgt)
    res[lab] = {nm: {"target": round(float(v[j] - v[:10].mean()), 2), "best other digit": (names[int(np.argmax(np.delete(v[:10], j if j < 10 else 0)))] if j < 10 else "newline")} for nm, v in out.items()}
    if lab.startswith("stop"):
        res[lab] = {nm: round(float(v[10] - v[11]), 2) for nm, v in out.items()}  # EOS minus newline
    res[lab]["final logit of target"] = round(float(lg[0, q, model.vocab[tgt] if tgt != "EOS" else model.eos]), 1)
    print(lab, res[lab], flush=True)
    if lab.startswith("digit 0"):
        st = s.rfind("\n") + 1; a0 = c["pattern0"][0, 1, q].cpu().numpy()
        res["L0H1 at '=' on test line tokens"] = {f"{ch}@{i}": round(float(a0[st + i]), 3) for i, ch in enumerate(s[st:])}
        print(res["L0H1 at '=' on test line tokens"], flush=True)
save_json("t21b_logit_parts", res); print("DONE")
