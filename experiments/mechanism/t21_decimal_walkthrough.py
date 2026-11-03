"""T21: step-by-step internal values for one decimal prompt, 8 + 7 (answer written reversed: 51).
A. Base reading: L0H1 attention at '=' by context-digit tier.
B. Fetch: L1H1 and L1H0 attention at '='.
C. Neuron inputs: for the neurons that push '5' most, the additive decomposition pre = mu + A[8] + B[7] fitted on decimal digit-0 data,
   their preferred sum (place cell), and their push on the logit of '5'. Then: add the hex base offset to every neuron input -> new answer.
D. Digit 1: carry rule a + b - written, and the neurons that push '1'.
E. Stop: neuron 177 and logit(EOS) vs logit(newline), final position."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, Problem, gelu_new, sample_shots, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; inv = {v: k for k, v in model.vocab.items()}; res = {}
tok = torch.tensor([model.vocab[c] for c in DIGITS], device=DEVICE)
Wfc, bfc, Wo, bo = sd["h.1.mlp.c_fc.weight"], sd["h.1.mlp.c_fc.bias"], sd["h.1.mlp.c_proj.weight"], sd["h.1.mlp.c_proj.bias"]
U = model.WE.T * sd["ln_f.weight"][:, None]
Wo_c = Wo - Wo.mean(1, keepdim=True)  # the final LayerNorm removes the mean across residual dimensions
p = Problem(8, 7, 10, sample_shots(random.Random(0), (8, 7))); base_prompt = p.prompt
def run(s):
    X = torch.tensor([[model.vocab[x] for x in s]], device=DEVICE)
    lg, c = model.forward(X, capture={"pattern0", "pattern1", "resid_mid1", "mlp_post1", "resid_post1"})
    q = X.shape[1] - 1; mid = c["resid_mid1"][0, q]; pre = model._ln("h.1.ln_2", mid) @ Wfc + bfc
    R = c["resid_post1"][0, q]; scale = (R - R.mean()).pow(2).mean().add(1e-5).sqrt()
    return lg[0, q], c, q, mid, pre, scale
# fitted additive tables and tuning, decimal digit 0, from stored data
t = Table(10); rows = t.select(QUERY, 0); rows = rows[t.meta["correct"][rows] == 1]
a_, b_ = t.meta["a"][rows] % 10, t.meta["b"][rows] % 10
R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
mid_t = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
P10 = (model._ln("h.1.ln_2", mid_t) @ Wfc + bfc).cpu().numpy()
D = np.c_[np.ones(len(a_)), np.eye(10)[a_][:, 1:], np.eye(10)[b_][:, 1:]]
co = np.linalg.lstsq(D, P10, rcond=None)[0]; mu = co[0]; A = np.r_[np.zeros((1, 256)), co[1:10]]; B = np.r_[np.zeros((1, 256)), co[10:19]]
# centre the digit tables so mu is the average input
mu = mu + A.mean(0) + B.mean(0); A = A - A.mean(0); B = B - B.mean(0)
s_ = a_ + b_; post10 = gelu_new(torch.as_tensor(P10)).numpy(); tuning = np.stack([post10[s_ == v].mean(0) for v in range(19)])
# hex offset: mean (pre_hex - pre_dec) over pairs present in both, digit 0
t16 = Table(16); r16 = t16.select(QUERY, 0); r16 = r16[t16.meta["correct"][r16] == 1]
a16, b16 = t16.meta["a"][r16] % 16, t16.meta["b"][r16] % 16
mid16 = torch.as_tensor(t16.get("resid0", r16), device=DEVICE) + torch.as_tensor(t16.get("z1", r16), device=DEVICE) @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
P16 = (model._ln("h.1.ln_2", mid16) @ Wfc + bfc).cpu().numpy()
diffs = [P16[(a16 == i) & (b16 == j)].mean(0) - P10[(a_ == i) & (b_ == j)].mean(0) for i in range(10) for j in range(10) if ((a16 == i) & (b16 == j)).any() and ((a_ == i) & (b_ == j)).any()]
hex_offset = torch.as_tensor(np.mean(diffs, 0), device=DEVICE, dtype=torch.float32)
def digit_logits(lg, n=3):
    v = lg[tok]; top = torch.topk(v, n); return [(DIGITS[int(i)], round(float(x), 1)) for x, i in zip(top.values, top.indices)]
# ---- digit 0
lg, c, q, mid, pre, scale = run(base_prompt); line_start = base_prompt.rfind("\n") + 1; ctx = base_prompt[:line_start]
tiers = {"0-1": "01", "2-7": "234567", "8-9": "89", "A-F": "ABCDEF"}
att0 = c["pattern0"][0, 1, q].cpu().numpy()
A_ = {"context digit counts": {k: sum(ch in v for ch in ctx) for k, v in tiers.items()},
      "L0H1 attention at '=' by tier (context)": {k: round(float(sum(att0[i] for i, ch in enumerate(ctx) if ch in v)), 3) for k, v in tiers.items()},
      "L0H1 attention at '=' on the test line": round(float(att0[line_start:].sum()), 3)}
res["A"] = A_; print("A", A_, flush=True)
att1 = c["pattern1"][0, :, q].cpu().numpy()
res["B"] = {"L1H1 on '8'": round(float(att1[1, line_start]), 2), "L1H0 on '7'": round(float(att1[0, line_start + 2]), 2)}; print("B", res["B"], flush=True)
post = gelu_new(pre); push = (post[:, None] * (Wo_c @ U[:, tok[:10]])) / scale  # [256, 10]: push of each neuron on digits 0-9
push_c = push - push.mean(1, keepdim=True)  # relative to the average decimal digit
top = torch.argsort(-push_c[:, 5])[:6].cpu().numpy()
cells = []
for n in top:
    cells.append({"neuron": int(n), "mu": round(float(mu[n]), 2), "A[8]": round(float(A[8, n]), 2), "B[7]": round(float(B[7, n]), 2), "fitted input": round(float(mu[n] + A[8, n] + B[7, n]), 2),
                  "actual input": round(float(pre[n]), 2), "output GELU": round(float(post[n]), 2), "preferred sum": int(tuning[:, n].argmax()), "push on '5' vs average digit": round(float(push_c[n, 5]), 2)})
res["C"] = {"cells": cells, "neurons with output > 0.1": int((post > 0.1).sum()), "total MLP push on '5' vs average digit": round(float(push_c[:, 5].sum()), 2),
            "top 6 neurons' share of that push": round(float(push_c[top, 5].sum() / push_c[:, 5].sum()), 2), "digit logits": digit_logits(lg)}
for x in cells: print("C", x, flush=True)
print("C", {k: v for k, v in res["C"].items() if k != "cells"}, flush=True)
pre_h = pre + hex_offset; post_h = gelu_new(pre_h); lg_h = model.unembed((mid + post_h @ Wo + bo)[None])[0]
res["C"]["with the hex offset added to every neuron input"] = digit_logits(lg_h)
res["C"]["hex offset size vs typical input spread"] = round(float(hex_offset.norm() / torch.as_tensor(P10).std(0).norm()), 2)
print("C hex offset ->", res["C"]["with the hex offset added to every neuron input"], flush=True)
# ---- digit 1
lg1, c1, q1, mid1, pre1, scale1 = run(base_prompt + "5"); att = c1["pattern1"][0, :, q1].cpu().numpy(); L = base_prompt + "5"
post1 = gelu_new(pre1); push1 = (post1[:, None] * (Wo_c @ U[:, tok[:10]])) / scale1; push1c = push1 - push1.mean(1, keepdim=True)
top1 = torch.argsort(-push1c[:, 1])[:5].cpu().numpy()
res["D"] = {"L1H0 on '+'": round(float(att[0, line_start + 1]), 2), "L1H1 on '8'": round(float(att[1, line_start]), 2), "L1H3 on '7'": round(float(att[3, line_start + 2]), 2),
            "a + b - written": 8 + 7 - 5, "threshold": 5.5, "digit logits": digit_logits(lg1),
            "top neurons pushing '1' (neuron, output, push)": [(int(n), round(float(post1[n]), 2), round(float(push1c[n, 1]), 2)) for n in top1], "their share": round(float(push1c[top1, 1].sum() / push1c[:, 1].sum()), 2), "total MLP push on '1' vs average digit": round(float(push1c[:, 1].sum()), 2)}
print("D", res["D"], flush=True)
# ---- stop
lg2, c2, q2, mid2, pre2, scale2 = run(base_prompt + "51"); post2 = gelu_new(pre2)
eos_dir = (U[:, model.eos] - U[:, model.vocab["\n"]])
res["E"] = {"final position": int(q2), "neuron 177 output": round(float(post2[177]), 2), "neuron 177 push on EOS vs newline": round(float(post2[177] * (Wo_c[177] @ eos_dir) / scale2), 2),
            "logit EOS": round(float(lg2[model.eos]), 1), "logit newline": round(float(lg2[model.vocab['\n']]), 1),
            "neuron 177 output at the two earlier steps": (round(float(post[177]), 2), round(float(post1[177]), 2))}
print("E", res["E"], flush=True)
res["prompt"] = base_prompt
save_json("t21_decimal_walkthrough", res); print("DONE")
