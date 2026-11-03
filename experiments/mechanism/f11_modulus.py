"""F11 (forward passes + stored): how does layer 1's MLP apply the modulus?

A. Periodicity: autocorrelation of each neuron's firing across raw sum s at every lag. Modulus-tuned neurons peak at lag = base.
B. Token writers: neurons that write a token, and when they fire. Do they fire for all sums congruent to it?
C. Logit shape: are logits a function of (t - s) mod base (a clock), per base?
D. Which neurons switch the rule? Minimal-pair prompts (context edits only), patch top-N L1 neurons at '=' between runs.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_base
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); res = {"A": {}, "B": {}, "C": {}, "D": {}}
tok = np.array([model.vocab[c] for c in DIGITS])
Wo = model.sd["h.1.mlp.c_proj.weight"]
write = (Wo @ model.WE[torch.as_tensor(tok, device=Wo.device)].T).cpu().numpy()
tunes = {}
for b in BASES:
    t = Table(b); rows = t.select(QUERY, 0); A, B = t.meta["a"][rows], t.meta["b"][rows]
    s = A % b + B % b
    X = t.get("mlp1", rows); R = t.get("resid1", rows)
    ns = 2 * b - 1
    T = np.stack([X[s == v].mean(0) for v in range(ns)])
    tunes[b] = T
    w = np.array([(s == v).sum() for v in range(ns)], float)
    Tc = T - np.average(T, axis=0, weights=w)
    var = (Tc**2 * w[:, None]).sum(0)
    act = var > 0.01 * var.max()
    lags = {}
    rng = np.random.default_rng(0)
    for lag in range(1, min(ns - 2, 20)):
        x, y = Tc[:-lag][:, act], Tc[lag:][:, act]
        lags[lag] = float((x * y).sum() / np.sqrt((x**2).sum() * (y**2).sum()))
    res["A"][b] = {"autocorr": lags, "n_active": int(act.sum())}
    print("A", b, "active", int(act.sum()), "autocorr by lag:", {k: round(v, 2) for k, v in lags.items()}, flush=True)
    if b == 2:
        continue
    # B: writers
    wr = write[:, :b] - write[:, :b].mean(1, keepdims=True)
    contrib = Tc * 0
    out = {}
    cong, other, ratio = [], [], []
    for tk in range(b):
        topn = np.argsort(-wr[:, tk])[:10]
        profile = Tc[:, topn].mean(1)
        same = np.array([v % b == tk for v in range(ns)])
        out[DIGITS[tk]] = {"neurons": topn.tolist(), "profile": profile.tolist()}
        cong.append(profile[same].mean()); other.append(profile[~same].mean())
    res["B"][b] = {"tokens": out, "mean_on_congruent": float(np.mean(cong)), "mean_on_other": float(np.mean(other))}
    # population vote: for each s, total written logit of each token by the MLP
    table = Tc @ wr
    top = table.argmax(1)
    res["B"][b]["vote_matches_s_mod_b"] = float(np.mean(top == np.arange(ns) % b))
    print("B", b, "writers fire on congruent sums", round(float(np.mean(cong)), 3), "vs other", round(float(np.mean(other)), 3), "vote matches", res["B"][b]["vote_matches_s_mod_b"], flush=True)
    for tk in (3, 5):
        print("   token", DIGITS[tk], "writer profile by s:", np.round(out[DIGITS[tk]]["profile"], 2).tolist(), flush=True)
    # C clock: logits as function of (t-s) mod b
    Lg = model.unembed(torch.as_tensor(R, device=DEVICE)).cpu().numpy()[:, tok[:b]]
    Lm = np.stack([Lg[s == v].mean(0) for v in range(ns)])
    Lc = Lm - Lm.mean(1, keepdims=True)
    offs = (np.arange(b)[None] - np.arange(ns)[:, None]) % b
    g = np.array([Lc[offs == o].mean() for o in range(b)])
    clock = g[offs]
    r2 = 1 - ((Lc - clock) ** 2).sum() / (Lc**2).sum()
    res["C"][b] = {"clock_r2": float(r2), "offset_profile": g.tolist()}
    print("C", b, "clock R2", round(float(r2), 3), "offset profile", np.round(g, 1).tolist(), flush=True)
save_json("f11_modulus", res)


def tens(strs):
    seqs = [[model.vocab[c] for c in z] for z in strs]
    L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
    for j, z in enumerate(seqs): x[j, :len(z)] = torch.tensor(z)
    return x.to(DEVICE), torch.tensor([len(z) - 1 for z in seqs], device=DEVICE)


def edit(ctx, d, rng, n):
    c = list(ctx); pos = [j for j, ch in enumerate(c) if ch not in "+=\n"]
    for j in rng.sample(pos, n): c[j] = d
    return "".join(c)


for low, high, ch, ne in [(8, 10, "9", 4), (10, 16, "F", 3), (8, 16, "A", 3)]:
    rng = random.Random(low * high); C, E, S = [], [], []
    while len(C) < 1600:
        ax, bx = rng.randint(0, low - 1), rng.randint(0, low - 1)
        if ax + bx < low: continue
        a, b_ = rng.randint(1, 999 // low) * low + ax, rng.randint(1, 999 // low) * low + bx
        ctx = "\n".join(render(*sh, low) for sh in sample_shots(rng, (a, b_)))
        t = f"{to_base(a, low)}+{to_base(b_, low)}="
        C.append(ctx + "\n" + t); E.append(edit(ctx, ch, rng, ne) + "\n" + t); S.append(ax + bx)
    S = np.array(S); rl, rh = tok[S % low], tok[S % high]
    post_c, post_e, res_c, pc, pe = [], [], [], [], []
    for i in range(0, len(C), 256):
        xc, q = tens(C[i:i + 256]); xe, _ = tens(E[i:i + 256]); J = torch.arange(len(xc), device=DEVICE)
        lc, cc = model.forward(xc, capture={"mlp_post1", "resid_post1"}); le, ce = model.forward(xe, capture={"mlp_post1"})
        post_c.append(cc["mlp_post1"][J, q]); post_e.append(ce["mlp_post1"][J, q]); res_c.append(cc["resid_post1"][J, q])
        pc.append(lc[J, q].argmax(-1).cpu().numpy()); pe.append(le[J, q].argmax(-1).cpu().numpy())
    post_c, post_e, res_c = torch.cat(post_c), torch.cat(post_e), torch.cat(res_c); pc, pe = np.concatenate(pc), np.concatenate(pe)
    keep = np.where((pc == rl) & (pe == rh) & (rl != rh))[0]
    print(f"D {low}->{high} usable {len(keep)}", flush=True)
    if len(keep) < 100: continue
    tr, te = keep[: len(keep) // 2], keep[len(keep) // 2 :]
    W = Wo @ model.WE.T  # [256, vocab]
    # direct effect on (rh - rl) logit difference, per neuron, on train half
    trt = torch.as_tensor(tr, device=DEVICE)
    delta = post_e[trt] - post_c[trt]
    gain = (delta * (W[:, torch.as_tensor(rh[tr], device=DEVICE)].T - W[:, torch.as_tensor(rl[tr], device=DEVICE)].T)).mean(0)
    order = gain.argsort(descending=True)
    share = gain.clamp_min(0).sort(descending=True).values.cumsum(0) / gain.clamp_min(0).sum()
    tet = torch.as_tensor(te, device=DEVICE)
    curve = {}
    for n in (1, 2, 4, 8, 16, 32, 64, 128, 256):
        idx = order[:n]
        d = torch.zeros_like(post_c[tet]); d[:, idx] = (post_e[tet] - post_c[tet])[:, idx]
        p = model.unembed(res_c[tet] + d @ Wo).argmax(-1).cpu().numpy()
        curve[n] = float((p == rh[te]).mean())
    res["D"][f"{low}->{high}"] = {"curve": curve, "gain_share_top": {n: float(share[n - 1]) for n in (1, 8, 32, 64)}, "top_neurons": order[:12].tolist(), "top_gain": gain[order[:12]].tolist(), "active_changed": int(((post_e[tet] - post_c[tet]).abs().mean(0) > 0.05).sum())}
    print("  flip by top-N neurons:", {k: round(v, 2) for k, v in curve.items()}, "gain share", {k: round(v, 2) for k, v in res["D"][f"{low}->{high}"]["gain_share_top"].items()}, "top", order[:8].tolist(), "changed", res["D"][f"{low}->{high}"]["active_changed"], flush=True)
    save_json("f11_modulus", res)
print("DONE")
