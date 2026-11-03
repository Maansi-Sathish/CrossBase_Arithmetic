"""F09 (forward passes): the circuit that reads the base.

A. Do L0 heads at test-line tokens prefer large digits in the context? Attention per context token by digit value.
B. Which L0 head carries the edit? Pairs differ by context edits only (aligned), then patch one L0 head's output at
   test-line tokens. Also patches by token type, to see whether the signal is carried by digits or by the query.
C. Is there a base direction? Mean edit effect on layer 0's output at test-line tokens, added to held-out clean runs.
   If base is an ordered 1-D quantity, moving along 8 -> 16 should pass through the decimal rule.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_problems, sample_shots, save_json, to_base

torch.set_grad_enabled(False)
model = Model()
res: dict = {"A": {}, "B": {}, "C": {}}
PAIRS = [(2, 8, "7", 4), (8, 10, "9", 4), (8, 16, "A", 2), (10, 16, "F", 2)]


# ------------------------------------------------------------------------------------- A
for base in (8, 10, 16):
    P = sample_problems(1500, base, seed=5)
    tot = np.zeros((4, 3, 16))  # head, region (operands, plus+eq, ...), digit value
    cnt = np.zeros((3, 16))
    for i in range(0, len(P), 256):
        ch = P[i : i + 256]
        strs = [p.prompt for p in ch]
        seqs = [[model.vocab[c] for c in s] for s in strs]
        L = max(map(len, seqs))
        x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, : len(s)] = torch.tensor(s)
        _, c = model.forward(x.to(DEVICE), capture={"pattern0"})
        pat = c["pattern0"].cpu().numpy()
        for j, s in enumerate(strs):
            start = s.rfind("\n") + 1
            eq = len(s) - 1
            plus = s.rfind("+")
            ctx_ids = [t for t in range(start) if s[t] in DIGITS]
            vals = np.array([DIGITS.index(s[t]) for t in ctx_ids])
            for r, positions in enumerate(([*range(start, plus)], [*range(plus + 1, eq)], [eq])):
                for pos in positions:
                    row = pat[j, :, pos][:, ctx_ids]
                    for v in np.unique(vals):
                        m = vals == v
                        tot[:, r, v] += row[:, m].sum(1)
                        cnt[r, v] += m.sum()
    per = tot / np.maximum(cnt[None], 1)
    res["A"][base] = {"per_token_attention": per}
    for r, nm in enumerate(("a digits", "b digits", "=")):
        for h in range(4):
            row = per[h, r, : base]
            order = float(np.corrcoef(np.arange(base), row)[0, 1])
            res["A"][base][f"order_corr/{nm}/H{h}"] = order
            print(f"A b{base} {nm} H{h} corr(value, attention)={order:+.2f} attention x1000:", np.round(row * 1000, 1).tolist(), flush=True)


# ------------------------------------------------------------------------------------- pairs
def make(n, seed, low):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        ax, bx = rng.randint(0, low - 1), rng.randint(0, low - 1)
        if ax + bx < low:
            continue
        a = rng.randint(1, 999 // low) * low + ax
        b = rng.randint(1, 999 // low) * low + bx
        ctx = "\n".join(render(*s, low) for s in sample_shots(rng, (a, b)))
        out.append(dict(ctx=ctx, test=f"{to_base(a, low)}+{to_base(b, low)}=", a0=ax, b0=bx))
    return out


def edit(ctx, d, rng, n):
    c = list(ctx)
    pos = [j for j, ch in enumerate(c) if ch not in "+=\n"]
    for j in rng.sample(pos, min(n, len(pos))):
        c[j] = d
    return "".join(c)


def tens(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]
    L = max(map(len, seqs))
    x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs):
        x[j, : len(s)] = torch.tensor(s)
    q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
    idx = torch.arange(L, device=DEVICE)[None]
    start = torch.tensor([s.rfind("\n") + 1 for s in strs], device=DEVICE)
    plus = torch.tensor([s.rfind("+") for s in strs], device=DEVICE)
    masks = {
        "test_line": (idx >= start[:, None]) & (idx <= q[:, None]),
        "operands": (idx >= start[:, None]) & (idx < q[:, None]) & (idx != plus[:, None]),
        "query": idx == q[:, None],
        "plus": idx == plus[:, None],
        "context": idx < start[:, None],
    }
    return x.to(DEVICE), q, masks


def predict(strs, hook_factory=None, bs=256):
    out = []
    for i in range(0, len(strs), bs):
        x, q, masks = tens(strs[i : i + bs])
        h = hook_factory(i, x, masks) if hook_factory else None
        lg, _ = model.forward(x, hooks=h)
        out.append(lg[torch.arange(len(x), device=DEVICE), q].argmax(-1).cpu().numpy())
    return np.concatenate(out)


def cap(strs, names, bs=256):
    out = []
    for i in range(0, len(strs), bs):
        x, q, masks = tens(strs[i : i + bs])
        _, c = model.forward(x, capture=set(names))
        out.append({k: v for k, v in c.items()})
    return out


tok = np.array([model.vocab[c] for c in DIGITS])
for low, high, ch, n_edit in PAIRS:
    P = make(1600, low * 100 + high, low)
    rng = random.Random(low + high)
    clean = [p["ctx"] + "\n" + p["test"] for p in P]
    edited = [edit(p["ctx"], ch, rng, n_edit) + "\n" + p["test"] for p in P]
    s = np.array([p["a0"] + p["b0"] for p in P])
    rl, rh = tok[s % low], tok[s % high]
    pc, pe = predict(clean), predict(edited)
    flip = np.where((pc == rl) & (pe == rh) & (rl != rh))[0][:800]
    print(f"\nPAIR {low}->{high}: clean follows low {float((pc == rl).mean()):.3f}, edited follows high {float((pe == rh).mean()):.3f}, usable {len(flip)}", flush=True)
    if len(flip) < 50:
        continue
    C = [clean[i] for i in flip]
    E = [edited[i] for i in flip]
    want_h, want_l = rh[flip], rl[flip]
    names = ["z0", "resid_pre1", "mlp_post0", "resid_mid0"]
    cc, ce = cap(C, names), cap(E, names)
    out = {"usable": len(flip), "flips": {}}

    def patch_flip(name, region, head=None, direction="clean_gets_edited"):
        S = C if direction == "clean_gets_edited" else E
        src = ce if direction == "clean_gets_edited" else cc
        want = want_h if direction == "clean_gets_edited" else want_l

        def factory(i, x, masks):
            m = masks[region]
            sv = src[i // 256][name]

            def fn(t, m=m, sv=sv):
                t = t.clone()
                if head is None:
                    t[m] = sv[m]
                else:
                    t[:, :, head][m] = sv[:, :, head][m]
                return t

            return {name: fn}

        return float((predict(S, factory) == want).mean())

    for h in range(4):
        for region in ("test_line", "query", "operands"):
            out["flips"][f"L0H{h}/{region}"] = {"noise": patch_flip("z0", region, h), "restore": patch_flip("z0", region, h, "edited_gets_clean")}
    out["flips"]["L0 all heads/test_line"] = {"noise": patch_flip("z0", "test_line"), "restore": patch_flip("z0", "test_line", None, "edited_gets_clean")}
    out["flips"]["L0 mlp/test_line"] = {"noise": patch_flip("mlp_post0", "test_line"), "restore": patch_flip("mlp_post0", "test_line", None, "edited_gets_clean")}
    res["B"][f"{low}->{high}"] = out
    print("  head flip (clean gets edited / edited gets clean):", {k: (round(v["noise"], 2), round(v["restore"], 2)) for k, v in out["flips"].items() if "test_line" in k}, flush=True)
    print("  by region:", {k: round(v["noise"], 2) for k, v in out["flips"].items() if "test_line" not in k}, flush=True)

    # C: base direction, learned on half, applied to the other half
    half = len(C) // 2
    v_q, v_op = [], []
    for ci, ei, i in zip(cc, ce, range(0, len(C), 256)):
        x, q, masks = tens(C[i : i + 256])
        d = ei["resid_pre1"] - ci["resid_pre1"]
        v_q.append(d[masks["query"]])
        v_op.append(d[masks["operands"]])
    v_q = torch.cat(v_q).mean(0)
    v_op = torch.cat(v_op).mean(0)
    curve = {}
    all_rules = sorted({low, high} | ({10} if low == 8 and high == 16 else set()))
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        def factory(i, x, masks, alpha=alpha):
            def fn(t):
                t = t.clone()
                t[masks["query"]] = t[masks["query"]] + alpha * v_q
                t[masks["operands"]] = t[masks["operands"]] + alpha * v_op
                return t
            return {"resid_pre1": fn}

        pr = predict(C[half:], factory)
        sv = s[flip][half:]
        curve[alpha] = {f"rule_{r}": float((pr == tok[sv % r]).mean()) for r in all_rules}
    res["C"][f"{low}->{high}"] = {"curve": curve, "v_query_norm": float(v_q.norm()), "v_operand_norm": float(v_op.norm()), "cos_query_operand": float(torch.dot(v_q, v_op) / (v_q.norm() * v_op.norm()))}
    print("  steering:", {a: {k[5:]: round(x, 2) for k, x in r.items()} for a, r in curve.items()}, "cos(q,op)=", round(res["C"][f"{low}->{high}"]["cos_query_operand"], 2), flush=True)
    save_json("f09_base_circuit", res)

# dimension of the base code, from natural prompts
means = {}
for base in BASES:
    P = sample_problems(1500, base, seed=6)
    strs = [p.prompt for p in P]
    acc = []
    for i in range(0, len(strs), 256):
        x, q, masks = tens(strs[i : i + 256])
        _, c = model.forward(x, capture={"resid_pre1"})
        acc.append(c["resid_pre1"][masks["query"]])
    means[base] = torch.cat(acc).mean(0)
M = torch.stack([means[b] for b in BASES])
Mc = M - M.mean(0)
sv = torch.linalg.svdvals(Mc)
cos = torch.nn.functional.cosine_similarity(Mc[:, None], Mc[None], dim=-1)
res["C"]["base_means_sv_share"] = ((sv**2) / (sv**2).sum()).tolist()
res["C"]["base_means_cos"] = cos.tolist()
U, S, Vt = torch.linalg.svd(Mc, full_matrices=False)
res["C"]["pc1_coords"] = (U[:, 0] * S[0]).tolist()
print("base mean SV share", np.round(res["C"]["base_means_sv_share"], 3), "PC1 coords (2,8,10,16):", np.round(res["C"]["pc1_coords"], 2), flush=True)
save_json("f09_base_circuit", res)
print("DONE")
