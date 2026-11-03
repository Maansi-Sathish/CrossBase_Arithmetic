"""E3: attention anatomy and digit-embedding geometry.

Part 1 -- attention: at the query for answer digit k, where does each head look? A digit-local
algorithm predicts mass on a_k, b_k (and a_{k-1}, b_{k-1} or the last emitted digit for the carry).
Also records the attention of operand tokens, which shows how alignment is computed.

Part 2 -- embeddings: is there number-line or circular structure in the 16 digit tokens themselves?
Leave-one-digit-out fits with a permutation null, since 16 points in 64 dimensions overfit easily.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, DIGITS, Batch, Model, iter_batches, sample_problems, save_json, to_base

model = Model()
res: dict = {}
SLOTS = [
    "a_k", "b_k", "a_k-1", "b_k-1", "a_older", "b_older", "a_higher", "b_higher",
    "+", "=", "o_k-1", "o_older", "ctx_digit", "ctx_op", "ctx_answer", "ctx_newline",
]


@torch.no_grad()
def attention_slots(base: int, n: int = 6000, kmax: int = 4) -> dict:
    """Mean attention per slot, by query digit k, head and layer."""
    problems = sample_problems(n, base, seed=21)
    tot = np.zeros((kmax, 2, 4, len(SLOTS)))
    cnt = np.zeros(kmax)
    for chunk in iter_batches(problems, 128):
        bt = Batch(model, chunk)
        _, cache = model.forward(bt.ids, capture={"pattern0", "pattern1"})
        pats = [cache["pattern0"].cpu().numpy(), cache["pattern1"].cpu().numpy()]
        for i, p in enumerate(chunk):
            eq, plus, la, lb = bt.p_eq[i], bt.p_plus[i], bt.len_a[i], bt.len_b[i]
            prompt = p.prompt
            line_start = bt.p_astart[i]
            # context labels
            ctx = {"ctx_digit": [], "ctx_op": [], "ctx_answer": [], "ctx_newline": []}
            after_eq = False
            for t, ch in enumerate(prompt[:line_start]):
                if ch == "\n":
                    ctx["ctx_newline"].append(t)
                    after_eq = False
                elif ch in "+=":
                    ctx["ctx_op"].append(t)
                    after_eq = ch == "="
                elif after_eq:
                    ctx["ctx_answer"].append(t)
                else:
                    ctx["ctx_digit"].append(t)
            for k in range(min(kmax, bt.len_ans[i])):
                # require the operand digit k to exist on both sides so a_k/b_k slots are defined
                if k >= la or k >= lb:
                    continue
                q = eq + k
                adig = lambda j: plus - 1 - j  # noqa: E731
                bdig = lambda j: eq - 1 - j  # noqa: E731
                slots: dict[str, list[int]] = {s: [] for s in SLOTS}
                slots["a_k"] = [adig(k)]
                slots["b_k"] = [bdig(k)]
                if k >= 1:
                    slots["a_k-1"] = [adig(k - 1)]
                    slots["b_k-1"] = [bdig(k - 1)]
                    slots["a_older"] = [adig(j) for j in range(k - 1)]
                    slots["b_older"] = [bdig(j) for j in range(k - 1)]
                    slots["o_k-1"] = [eq + k]
                    slots["o_older"] = [eq + j for j in range(1, k)]
                slots["a_higher"] = [adig(j) for j in range(k + 1, la)]
                slots["b_higher"] = [bdig(j) for j in range(k + 1, lb)]
                slots["+"] = [plus]
                slots["="] = [eq]
                slots.update(ctx)
                for L in range(2):
                    for h in range(4):
                        row = pats[L][i, h, q]
                        for si, s in enumerate(SLOTS):
                            if slots[s]:
                                tot[k, L, h, si] += row[slots[s]].sum()
                cnt[k] += 1
    return {"mean": tot / cnt[:, None, None, None], "count": cnt}


res["slots"] = SLOTS
res["attention"] = {}
store = {}
for b in BASES:
    r = attention_slots(b)
    res["attention"][b] = {"mean": r["mean"], "count": r["count"]}
    store[b] = r["mean"]
    print("base", b, "count", r["count"])
    for k in range(3):
        for L in range(2):
            for h in range(4):
                row = r["mean"][k, L, h]
                top = np.argsort(-row)[:4]
                print(f"  k{k} L{L}H{h}: " + ", ".join(f"{SLOTS[t]}={row[t]:.2f}" for t in top))

# Cross-base consistency: correlation of per-head slot profiles between each pair of bases.
res["attention_crossbase_corr"] = {}
for k in range(3):
    m = np.zeros((4, 4))
    for i, x in enumerate(BASES):
        for j, y in enumerate(BASES):
            m[i, j] = np.corrcoef(store[x][k].ravel(), store[y][k].ravel())[0, 1]
    res["attention_crossbase_corr"][k] = m
    print("cross-base attention profile corr, k", k, "\n", np.round(m, 3))


# --------------------------------------------------------------------------------------------- #
# Part 1b: what do operand tokens attend to, at L0? (how does the model align digits?)
@torch.no_grad()
def operand_attention(base: int, n: int = 3000) -> dict:
    """For test-line digit tokens at L0/L1: mass on previous digits of same number, '+', context."""
    problems = sample_problems(n, base, seed=22)
    keys = ["same_number_prev_digits", "first_operand_from_b", "plus", "ctx", "self"]
    tot = {(w, L): np.zeros((4, len(keys))) for w in ("a", "b", "plus", "eq") for L in range(2)}
    cnt = {w: 0 for w in ("a", "b", "plus", "eq")}
    for chunk in iter_batches(problems, 128):
        bt = Batch(model, chunk)
        _, cache = model.forward(bt.ids, capture={"pattern0", "pattern1"})
        pats = [cache["pattern0"].cpu().numpy(), cache["pattern1"].cpu().numpy()]
        for i in range(len(chunk)):
            st, plus, eq = bt.p_astart[i], bt.p_plus[i], bt.p_eq[i]
            for L in range(2):
                for h in range(4):
                    # last digit of a
                    row = pats[L][i, h, plus - 1]
                    tot[("a", L)][h] += [row[st : plus - 1].sum(), 0, 0, row[:st].sum(), row[plus - 1]]
                    # last digit of b
                    row = pats[L][i, h, eq - 1]
                    tot[("b", L)][h] += [row[plus + 1 : eq - 1].sum(), row[st : plus].sum(), row[plus], row[:st].sum(), row[eq - 1]]
                    row = pats[L][i, h, plus]
                    tot[("plus", L)][h] += [0, row[st : plus].sum(), row[plus], row[:st].sum(), 0]
                    row = pats[L][i, h, eq]
                    tot[("eq", L)][h] += [row[plus + 1 : eq].sum(), row[st : plus].sum(), row[plus], row[:st].sum(), row[eq]]
            for w in cnt:
                cnt[w] += 1
    return {"keys": keys, **{f"{w}_L{L}": tot[(w, L)] / cnt[w] for w in cnt for L in range(2)}}


res["operand_attention"] = {b: operand_attention(b) for b in BASES}

# --------------------------------------------------------------------------------------------- #
# Part 2: digit embedding geometry
W = model.WE[[model.vocab[c] for c in DIGITS]].cpu().numpy()  # [16, 64]
Wc = W - W.mean(0)
U, S, Vt = np.linalg.svd(Wc, full_matrices=False)
res["embedding_var_explained"] = (S**2 / (S**2).sum())
res["embedding_pcs"] = U[:, :6] * S[:6]
print("embedding PCA var", np.round(res["embedding_var_explained"][:8], 3))

# number-line structure: does cosine distance grow with numeric distance?
norm = Wc / np.linalg.norm(Wc, axis=1, keepdims=True)
cos = norm @ norm.T
iu = np.triu_indices(16, 1)
dist = np.abs(iu[0] - iu[1]).astype(float)
obs = np.corrcoef(cos[iu], dist)[0, 1]
rng = np.random.default_rng(0)
null = []
for _ in range(5000):
    pm = rng.permutation(16)
    null.append(np.corrcoef(cos[pm][:, pm][iu], dist)[0, 1])
res["embedding_order_corr"] = {"r": obs, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)), "p": float((np.array(null) <= obs).mean())}
print("corr(cos, |i-j|)", res["embedding_order_corr"])
res["embedding_cos"] = cos


def lodo_r2(B: np.ndarray, Y: np.ndarray, lam: float = 1e-3) -> float:
    """Leave-one-digit-out R^2 of basis B predicting embeddings Y."""
    n = len(Y)
    pred = np.zeros_like(Y)
    for i in range(n):
        tr = np.arange(n) != i
        Bt = np.c_[np.ones(n - 1), B[tr]]
        coef = np.linalg.solve(Bt.T @ Bt + lam * np.eye(Bt.shape[1]), Bt.T @ Y[tr])
        pred[i] = np.r_[1, B[i]] @ coef
    return 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum()


d = np.arange(16.0)
Y = U[:, :8] * S[:8]
bases_fit = {
    "linear": d[:, None] / 15,
    "parity": (d[:, None] % 2),
    "T=2 + linear": np.c_[d / 15, d % 2],
    "T=8": np.c_[np.cos(2 * np.pi * d / 8), np.sin(2 * np.pi * d / 8)],
    "T=10": np.c_[np.cos(2 * np.pi * d / 10), np.sin(2 * np.pi * d / 10)],
    "T=16": np.c_[np.cos(2 * np.pi * d / 16), np.sin(2 * np.pi * d / 16)],
    "linear+T=8": np.c_[d / 15, np.cos(2 * np.pi * d / 8), np.sin(2 * np.pi * d / 8)],
    "linear+T=10": np.c_[d / 15, np.cos(2 * np.pi * d / 10), np.sin(2 * np.pi * d / 10)],
    "linear+T=16": np.c_[d / 15, np.cos(2 * np.pi * d / 16), np.sin(2 * np.pi * d / 16)],
    "is_letter": (d[:, None] >= 10).astype(float),
    "linear+is_letter": np.c_[d / 15, d >= 10],
    "bits": np.c_[(d.astype(int) >> 0) & 1, (d.astype(int) >> 1) & 1, (d.astype(int) >> 2) & 1, (d.astype(int) >> 3) & 1],
}
res["embedding_fits"] = {}
for name, B in bases_fit.items():
    obs_r2 = lodo_r2(B.astype(float), Y)
    nulls = [lodo_r2(B[rng.permutation(16)].astype(float), Y) for _ in range(500)]
    res["embedding_fits"][name] = {
        "lodo_r2": obs_r2,
        "null_mean": float(np.mean(nulls)),
        "null_95": float(np.quantile(nulls, 0.95)),
        "p": float((np.array(nulls) >= obs_r2).mean()),
    }
    print(f"  {name:20s} LODO R2 {obs_r2:+.3f}  null95 {res['embedding_fits'][name]['null_95']:+.3f}  p={res['embedding_fits'][name]['p']:.3f}")

# Unembedding view: logit lens of digit embedding is the same matrix, since weights are tied.
res["tied_embeddings"] = True
save_json("e03_attention_embed", res)
np.savez(Path(__file__).resolve().parent / "data" / "e03_attention.npz", **{f"b{b}": store[b] for b in BASES})
