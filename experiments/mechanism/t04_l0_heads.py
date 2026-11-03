"""T04 (forward passes + weights): the two layer 0 mechanisms.

A. L0H3 at digit tokens: what does it attend to as a function of index within the number? A count-like signal would show
   attention to the line start shrinking roughly as 1/(index+1).
B. Why L1H1 ignores b's digits: ablate ALL layer 0 heads (and L0 MLP) at b's digits.
C. L0H1 tiers from weights: score of a key digit d against real test-line queries, split into token and position parts;
   which embedding features produce the tiers.
D. Dose: measured L0H1 attention on edited tokens vs flip rate; and the softmax prediction from per-token scores.
E. What L0H1 writes for each tier; alignment with the measured base direction at the query.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_base
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; PE, WE = model.PE, model.WE; res = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}
tok = np.array([model.vocab[c] for c in DIGITS])
def ids(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
    return x.to(DEVICE)
def prompts(b, n, seed, minlen=1):
    rng = random.Random(seed); P = []
    while len(P) < n:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) < minlen or len(sb) < minlen: continue
        ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_))); st = len(ctx) + 1
        P.append((f"{ctx}\n{sa}+{sb}=", st, st + len(sa), st + len(sa) + 1 + len(sb), a, c_))
    return P
# A
for b in []:
    P = prompts(b, 600, b); tab = {}
    for i in range(0, 600, 150):
        ch = P[i:i + 150]; x = ids([p[0] for p in ch]); _, c = model.forward(x, capture={"pattern0"}); pat = c["pattern0"][:, 3].cpu().numpy()
        for j, (s, st, pl, eq, *_ ) in enumerate(ch):
            for pos in range(st, pl):
                l = pos - st; row = pat[j, pos]
                e = tab.setdefault(l, [0, 0., 0., 0., 0.]); e[0] += 1; e[1] += row[st - 1]; e[2] += row[st:pos].sum(); e[3] += row[pos]; e[4] += row[:st - 1].sum()
    res["A"][b] = {int(l): {"n": v[0], "newline": v[1] / v[0], "earlier digits": v[2] / v[0], "self": v[3] / v[0], "context": v[4] / v[0]} for l, v in sorted(tab.items())}
    print(f"A b{b} L0H3 at a-digit with index l from start: ", {l: (round(v['newline'], 2), round(v['earlier digits'], 2), round(v['self'], 2)) for l, v in res['A'][b].items() if v['n'] > 30}, flush=True)
# B
for b in BASES:
    P = prompts(b, 600, b + 50, 2); x = ids([p[0] for p in P]); n = x.shape[1]; idx = torch.arange(n, device=DEVICE)[None]
    st, pl, eq = [torch.tensor([p[i] for p in P], device=DEVICE) for i in (1, 2, 3)]; bm = (idx > pl[:, None]) & (idx < eq[:, None]); J = torch.arange(len(P), device=DEVICE)
    am = (idx >= st[:, None]) & (idx < pl[:, None]); qm_ = idx == eq[:, None]
    _, c0 = model.forward(x, capture={"z0", "mlp_post0", "resid_pre1"}); za = c0["z0"][am].mean(0); ma = c0["mlp_post0"][am].mean(0); ra = c0["resid_pre1"][am].mean(0)
    zq_other = c0["z0"][am].mean(0)
    out = {}
    for nm, hooks in (("clean", None), ("b digits get a-digit average of L0 heads", {"z0": lambda t: torch.where(bm[..., None, None], za, t)}), ("b digits get a-digit average of L0 MLP", {"mlp_post0": lambda t: torch.where(bm[..., None], ma, t)}), ("b digits get a-digit average of layer 0 output (except token and tag)", {"resid_pre1": lambda t: torch.where(bm[..., None], t - (t[bm].mean(0) - ra), t)}), ("all L0 heads mean-ablated at '='", {"z0": lambda t: torch.where(qm_[..., None, None], c0["z0"][qm_].mean(0), t)})):
        _, c = model.forward(x, hooks=hooks, capture={"pattern1"}); r1 = c["pattern1"][J, 1, eq]
        out[nm] = {"H1_on_b": float((r1 * bm).sum(1).mean()), "H1_on_a0": float(r1[J, pl - 1].mean())}
    res["B"][b] = out; print("B", b, {k: {a: round(v, 2) for a, v in r.items()} for k, r in out.items()}, flush=True)
# C weights
Wqkv = sd["h.0.attn.c_attn.weight"]; bqkv = sd["h.0.attn.c_attn.bias"]
Wq, Wk, Wv = Wqkv[:, 16:32], Wqkv[:, 64 + 16:64 + 32], Wqkv[:, 128 + 16:128 + 32]
bq, bk, bv = bqkv[16:32], bqkv[64 + 16:64 + 32], bqkv[128 + 16:128 + 32]
ln = lambda v: model._ln("h.0.ln_1", v)
P = prompts(16, 300, 7)
Q = []
for p in P[:300]:
    s, st, pl, eq = p[:4]
    for pos in range(st, eq + 1):
        Q.append(ln(WE[model.vocab[s[pos]]] + PE[pos]) @ Wq + bq)
Q = torch.stack(Q); qm = Q.mean(0)
ctxpos = torch.arange(10, 120, device=DEVICE)
S_full = np.zeros(16); S_tok = np.zeros(16)
for d in range(16):
    keys = ln(WE[tok[d]][None] + PE[ctxpos]) @ Wk + bk
    S_full[d] = float((Q @ keys.T).mean() / 4)
    ktok = ln(WE[tok[d]] + PE[ctxpos].mean(0)) @ Wk + bk
    S_tok[d] = float((Q @ ktok).mean() / 4)
pos_var = float(np.mean([((Q @ (ln(WE[tok[d]][None] + PE[ctxpos]) @ Wk + bk).T / 4).mean(0)).var().item() for d in range(16)]))
dv = np.arange(16)
feats = {"tier": np.select([dv <= 1, dv <= 7, dv <= 9], [0, 1, 2], 3).astype(float), "digit value": dv.astype(float), "is letter": (dv >= 10).astype(float)}
fits = {nm: float(np.corrcoef(f, S_full)[0, 1] ** 2) for nm, f in feats.items()}
Tm = np.stack([np.eye(4)[int(t)] for t in feats["tier"]])
within = float(sum(((S_full[feats["tier"] == t] - S_full[feats["tier"] == t].mean()) ** 2).sum() for t in range(4)) / ((S_full - S_full.mean()) ** 2).sum())
res["C"] = {"score_by_digit": S_full.tolist(), "token_part_by_digit": S_tok.tolist(), "variance_across_positions_mean": pos_var, "r2_with": fits, "within_tier_share": 1 - (1 - within) if False else within}
print("C L0H1 mean score by key digit 0..F:", np.round(S_full, 1).tolist())
print("   token part only:", np.round(S_tok, 1).tolist(), "| variance across key positions", round(pos_var, 2), "| R2 with tier", round(fits['tier'], 2), "value", round(fits['digit value'], 2), "letter", round(fits['is letter'], 2), "| within-tier share of variance", round(within, 3), flush=True)
# token embedding axis the key reads
g = (Wk @ qm).cpu().numpy()
E = WE[torch.as_tensor(tok, device=DEVICE)].cpu().numpy()
proj = (E - E.mean(0)) @ g
res["C"]["key_axis_projection_of_raw_embeddings"] = proj.tolist()
print("   raw digit embeddings projected on the key-reading axis:", np.round(proj, 2).tolist(), flush=True)
# D dose
for low, ch in ((8, "9"), (8, "A"), (10, "A")):
    rng = random.Random(ord(ch) + low); rows = {}
    for nedit in (1, 2, 4, 8):
        S, Cs, sums, editpos = [], [], [], []
        while len(S) < 400:
            ax, bx = rng.randint(0, low - 1), rng.randint(0, low - 1)
            if ax + bx < low: continue
            a, c_ = rng.randint(1, 999 // low) * low + ax, rng.randint(1, 999 // low) * low + bx
            ctx = list("\n".join(render(*s, low) for s in sample_shots(rng, (a, c_)))); pos = [j for j, x in enumerate(ctx) if x not in "+=\n"]
            ep = rng.sample(pos, nedit)
            for j in ep: ctx[j] = ch
            S.append("".join(ctx) + f"\n{to_base(a, low)}+{to_base(c_, low)}="); sums.append(ax + bx); editpos.append(ep)
        x = ids(S); lg, c = model.forward(x, capture={"pattern0"}); q = torch.tensor([len(s) - 1 for s in S], device=DEVICE); J = torch.arange(len(S), device=DEVICE)
        pat = c["pattern0"][J, 1, q].cpu().numpy()
        mass = np.array([pat[j, ep].sum() for j, ep in enumerate(editpos)])
        hi = 16 if ch == "A" else 10
        pr = lg[J, q].argmax(-1).cpu().numpy(); sums = np.array(sums)
        rows[nedit] = {"attention_on_edits": float(mass.mean()), "follows_new_rule": float((pr == tok[sums % hi]).mean()), "follows_old_rule": float((pr == tok[sums % low]).mean())}
    res["D"][f"{low}+{ch}"] = rows
    print(f"D context base {low}, edits to {ch}:", {k: (round(v['attention_on_edits'], 2), round(v['follows_new_rule'], 2)) for k, v in rows.items()}, flush=True)
# E OV by tier
OV = sd["h.0.attn.c_proj.weight"][16:32]
vout = {}
for d in range(16):
    vv = (ln(WE[tok[d]][None] + PE[ctxpos]) @ Wv + bv).mean(0) @ OV
    vout[d] = vv
tiers = {"0-1": [0, 1], "2-7": list(range(2, 8)), "8-9": [8, 9], "A-F": list(range(10, 16))}
tm = {k: torch.stack([vout[d] for d in v]).mean(0) for k, v in tiers.items()}
res["E"]["tier_output_cos"] = {f"{a} vs {b_}": float(torch.nn.functional.cosine_similarity(tm[a], tm[b_], dim=0)) for a in tm for b_ in tm if a < b_}
res["E"]["within_tier_cos_A-F"] = float(np.mean([torch.nn.functional.cosine_similarity(vout[i], vout[j], dim=0).item() for i in range(10, 16) for j in range(10, 16) if i < j]))
print("E L0H1 output, cosine between tier means:", {k: round(v, 2) for k, v in res["E"]["tier_output_cos"].items()}, "| within letters", round(res["E"]["within_tier_cos_A-F"], 2), flush=True)
save_json("t04_l0_heads", res); print("DONE")
