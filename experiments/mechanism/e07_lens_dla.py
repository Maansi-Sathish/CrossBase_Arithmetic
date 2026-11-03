"""E7: logit lens and direct logit attribution, for pivot translation and component contributions.

Pivot test (cf. Wendler et al., "Do Llamas Work in English?"): at each intermediate stage, does the
residual stream decode as the answer digit in the prompt's base, or in a DIFFERENT base first? Scored on
conflict trials, where the digits differ between the two bases, so agreement can't explain a hit.

DLA: the final logit is a linear sum over [embeddings, L0 attn, L0 mlp, L1 attn, L1 mlp], up to the final
LayerNorm's scale. We report each one's contribution to (correct digit minus mean digit logit).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, DIGITS, Batch, Model, iter_batches, sample_problems, save_json, schoolbook, to_answer

model = Model()
digit_ids = torch.tensor([model.vocab[c] for c in DIGITS], device=model.device)
STAGES = ["resid_pre0", "resid_mid0", "resid_post0", "resid_mid1", "resid_post1"]
res: dict = {"stages": STAGES, "lens": {}, "dla": {}, "dla_heads": {}}


@torch.no_grad()
def run(base: int, n: int = 8000, kmax: int = 3) -> dict:
    problems = sample_problems(n, base, seed=31)
    # for each k, stage: rates of top token == own digit, foreign digit (on conflict trials), etc.
    lens = {k: {s: {"own": 0, "total": 0, **{f"as_b{t}": 0 for t in BASES if t != base}, **{f"conf_n_b{t}": 0 for t in BASES if t != base}, **{f"conf_own_b{t}": 0 for t in BASES if t != base}} for s in STAGES} for k in range(kmax)}
    dla = np.zeros((kmax, 5))
    dla_heads = np.zeros((kmax, 2, 4))
    dla_cnt = np.zeros(kmax)
    capture = set(STAGES) | {"attn_out0", "mlp_out0", "attn_out1", "mlp_out1", "z0", "z1"}
    for chunk in iter_batches(problems, 256):
        bt = Batch(model, chunk)
        logits, cache = model.forward(bt.ids, capture=capture)
        for k in range(kmax):
            sel = np.where(bt.len_ans > k)[0]
            if len(sel) == 0:
                continue
            q = torch.tensor(bt.p_eq[sel] + k, device=model.device)
            si = torch.tensor(sel, device=model.device)
            own = [p for i, p in enumerate(chunk) if i in set(sel.tolist())]
            own_digit = torch.tensor([model.vocab[to_answer(p.a + p.b, base)[k]] for p in own], device=model.device)
            foreign = {}
            for t in BASES:
                if t == base:
                    continue
                ans = [to_answer(p.a + p.b, t) for p in own]
                foreign[t] = torch.tensor([model.vocab[x[k]] if len(x) > k else -1 for x in ans], device=model.device)
            for s in STAGES:
                lg = model.unembed(cache[s][si, q])
                top = lg.argmax(-1)
                d = lens[k][s]
                d["own"] += (top == own_digit).sum().item()
                d["total"] += len(sel)
                for t, f in foreign.items():
                    d[f"as_b{t}"] += (top == f).sum().item()
                    # conflict trials: digit differs between the two bases
                    c = (f != own_digit) & (f >= 0)
                    d[f"conf_n_b{t}"] += c.sum().item()
                    d[f"conf_own_b{t}"] += ((top == f) & c).sum().item()
            # DLA uses the scale of the final LayerNorm at each sample, frozen, so contributions sum
            final = cache["resid_post1"][si, q]
            scale = model.sd["ln_f.weight"] / (final.std(-1, keepdim=True, unbiased=False) + 1e-5)
            def contrib(v: torch.Tensor) -> torch.Tensor:
                proj = (v * scale) @ model.WE.T  # [n, vocab]
                return proj.gather(1, own_digit[:, None])[:, 0] - proj[:, digit_ids[: base]].mean(1)
            emb = cache["resid_pre0"][si, q]
            parts = [emb, cache["attn_out0"][si, q], cache["mlp_out0"][si, q], cache["attn_out1"][si, q], cache["mlp_out1"][si, q]]
            for pi, v in enumerate(parts):
                dla[k, pi] += contrib(v).sum().item()
            # per-head contribution, through each layer's output projection
            for L in range(2):
                zz = cache[f"z{L}"][si, q]  # [n, h, dh]
                wout = model.sd[f"h.{L}.attn.c_proj.weight"]
                for h in range(4):
                    zh = torch.zeros(len(sel), model.d_model, device=model.device)
                    zh[:, h * model.d_head : (h + 1) * model.d_head] = zz[:, h]
                    dla_heads[k, L, h] += contrib(zh @ wout).sum().item()
            dla_cnt[k] += len(sel)
    return {"lens": lens, "dla": dla / dla_cnt[:, None], "dla_heads": dla_heads / dla_cnt[:, None, None]}


for base in BASES:
    r = run(base)
    res["lens"][base] = r["lens"]
    res["dla"][base] = r["dla"]
    res["dla_heads"][base] = r["dla_heads"]
    print("base", base)
    for k in r["lens"]:
        for s in STAGES:
            d = r["lens"][k][s]
            line = f"  k{k} {s}: own={d['own'] / max(d['total'], 1):.3f}"
            for t in BASES:
                if t != base:
                    line += f" | b{t} on conflict={d[f'conf_own_b{t}'] / max(d[f'conf_n_b{t}'], 1):.3f}"
            print(line)
    print("  DLA [embed, attn0, mlp0, attn1, mlp1] per k\n", np.round(r["dla"], 2))
    print("  DLA heads (k0)\n", np.round(r["dla_heads"][0], 2), flush=True)
save_json("e07_lens_dla", res)
