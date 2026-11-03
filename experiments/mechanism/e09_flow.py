"""E9: information flow, stage by stage. What does each sublayer add, and is it the same in every base?

Walks the forward pass in order: embed -> L0 attn -> L0 mlp -> L1 attn -> L1 mlp. At each stage we probe the
schoolbook variables at q_k. Comparing the input and the output of a sublayer shows what its neurons computed.
Also tests:
  * pooled probes: one probe across all four bases. A pooled probe for raw sum s works only if s is written
    the same way in every base; a pooled probe for the output digit o works only if o is. Where the code
    switches from s to o is where the base-dependent wrap happens.
  * sublayer maps: fit a linear map input -> output for a sublayer in base A and test it in base B. If it
    transfers, the sublayer does the same job across bases; a gap isolates the base-specific part.
  * completeness: how much of the digit logits does a saturated table on (a_k, b_k, c_k, base) explain?
    What's left over is non-local computation, such as a whole-number value.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, DIGITS, Batch, Model, iter_batches, sample_problems, save_json, schoolbook
from probe_lib import Ridge, onehot, r2

model = Model()
STAGES = ["resid_pre0", "attn_out0", "resid_mid0", "mlp_out0", "resid_post0", "attn_out1", "resid_mid1", "mlp_out1", "resid_post1"]
digit_ids = torch.tensor([model.vocab[c] for c in DIGITS], device=model.device)
VARS = ["a", "b", "c_in", "s", "o", "c_out"]
NCLS = {"a": 16, "b": 16, "c_in": 2, "s": 32, "o": 16, "c_out": 2}
res: dict = {"stages": STAGES}


@torch.no_grad()
def collect(base: int, k: int, n: int = 12000) -> dict:
    """Fresh activations at q_k for every stage, teacher-forced, with schoolbook labels."""
    problems = [p for p in sample_problems(n, base, seed=41) if p.a + p.b >= base**k]
    out = {s: [] for s in STAGES}
    logits_all = []
    labels = {v: [] for v in VARS}
    idx = []
    for ci, chunk in enumerate(iter_batches(problems, 256)):
        bt = Batch(model, chunk)
        logits, cache = model.forward(bt.ids, capture=set(STAGES))
        q = torch.tensor(bt.p_eq + k, device=model.device)
        r = torch.arange(len(chunk), device=model.device)
        for s in STAGES:
            out[s].append(cache[s][r, q].cpu().numpy())
        logits_all.append(logits[r, q][:, digit_ids].cpu().numpy())
        for p in chunk:
            sb = schoolbook(p.a, p.b, base)
            for v in VARS:
                labels[v].append(sb[v][k])
            idx.append((p.a, p.b))
    return {
        **{s: np.concatenate(v) for s, v in out.items()},
        "logits": np.concatenate(logits_all),
        **{f"y_{v}": np.array(labels[v]) for v in VARS},
        "pairs": np.array(idx),
    }


data = {(b, k): collect(b, k) for b in BASES for k in [0, 1, 2]}
print("collected", flush=True)


def split(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random(n) < 0.2


# ---------------------------------------------------------------------- A: stage x variable, within base
res["within"] = {}
for b in BASES:
    res["within"][b] = {}
    for k in [0, 1, 2]:
        D = data[(b, k)]
        te = split(len(D["y_a"]))
        tab = {}
        for s in STAGES:
            row = {}
            for v in VARS:
                y = D[f"y_{v}"]
                if len(np.unique(y[~te])) < 2:
                    continue
                p = Ridge(D[s][~te])
                p.fit(onehot(y[~te], NCLS[v]))
                row[v] = float((p.predict(D[s][te]).argmax(1).cpu().numpy() == y[te]).mean())
            tab[s] = row
        res["within"][b][k] = tab
        print(f"within b{b} k{k}")
        for s in STAGES:
            print("   ", s.ljust(12), {v: round(x, 2) for v, x in tab[s].items()}, flush=True)

# ---------------------------------------------------------------------- B: pooled probes
res["pooled"] = {}
for k in [0, 1]:
    res["pooled"][k] = {}
    for s in STAGES:
        X = np.concatenate([data[(b, k)][s] for b in BASES])
        labels = np.concatenate([np.full(len(data[(b, k)][s]), b) for b in BASES])
        te = split(len(X), 1)
        row = {}
        for v in VARS:
            y = np.concatenate([data[(b, k)][f"y_{v}"] for b in BASES])
            p = Ridge(X[~te])
            p.fit(onehot(y[~te], NCLS[v]))
            pred = p.predict(X[te]).argmax(1).cpu().numpy()
            row[v] = {int(b): float((pred[labels[te] == b] == y[te][labels[te] == b]).mean()) for b in BASES}
        res["pooled"][k][s] = row
        print(f"pooled k{k} {s}:", {v: [round(row[v][b], 2) for b in BASES] for v in ("s", "o", "c_in")}, flush=True)

# ---------------------------------------------------------------------- C: sublayer map transfer
res["sublayer_maps"] = {}
pairs = {"L0 mlp": ("resid_mid0", "mlp_out0"), "L1 mlp": ("resid_mid1", "mlp_out1"), "L1 attn": ("resid_post0", "attn_out1"), "L0 attn": ("resid_pre0", "attn_out0")}
for name, (src, dst) in pairs.items():
    for k in [0, 1]:
        mat = np.zeros((4, 4))
        for i, bx in enumerate(BASES):
            Dx = data[(bx, k)]
            te = split(len(Dx[src]))
            p = Ridge(Dx[src][~te])
            p.fit(Dx[dst][~te])
            for j, by in enumerate(BASES):
                Dy = data[(by, k)]
                tey = split(len(Dy[src]))
                mat[i, j] = r2(p.predict(Dy[src][tey]), Dy[dst][tey].astype(np.float64))
        res["sublayer_maps"][f"{name}/k{k}"] = mat
        print(f"sublayer linear map {name} k{k} (train row, test col)\n", np.round(mat, 2), flush=True)

# ---------------------------------------------------------------------- D: completeness of digit-local state
res["completeness"] = {}
for b in BASES:
    res["completeness"][b] = {}
    for k in [0, 1, 2]:
        D = data[(b, k)]
        L = D["logits"][:, :b]
        L = L - L.mean(1, keepdims=True)
        te = split(len(L), 2)
        key = D["y_a"] * 1000 + D["y_b"] * 10 + D["y_c_in"]
        # saturated table on (a_k, b_k, c_k): the strictest possible digit-local model
        uniq = np.unique(key[~te])
        means = {u: L[~te][key[~te] == u].mean(0) for u in uniq}
        pred = np.stack([means.get(x, L[~te].mean(0)) for x in key[te]])
        residual = L[te] - pred
        r2_local = 1 - (residual**2).sum() / ((L[te] - L[te].mean(0)) ** 2).sum()
        # is what is left over explained by anything else? Probe the leftover from higher-digit info.
        a, bb = D["pairs"][:, 0], D["pairs"][:, 1]
        key2 = key * 100 + (a // b ** (k + 1)) % b * 10 + (bb // b ** (k + 1)) % b if b <= 10 else key * 400 + (a // b ** (k + 1)) % b * 20 + (bb // b ** (k + 1)) % b
        residual_tr = L[~te] - np.stack([means[x] for x in key[~te]])
        # how much leftover goes with the integer's size?
        size = np.c_[a / 1000, bb / 1000]
        p = Ridge(size[~te])
        p.fit(residual_tr)
        r2_size = r2(p.predict(size[te]), residual.astype(np.float64))
        res["completeness"][b][k] = {
            "r2_table_akbkck": float(r2_local),
            "r2_leftover_from_size": float(r2_size),
            "argmax_agreement_of_table": float((pred.argmax(1) == L[te].argmax(1)).mean()),
        }
        print("completeness", b, k, res["completeness"][b][k], flush=True)

save_json("e09_flow", res)
