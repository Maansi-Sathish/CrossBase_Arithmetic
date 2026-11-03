"""V05 (forward passes + replay): verify the base and carry vectors; trace the base signal.

A. Controls: random vectors of equal norm, non-conflict accuracy after adding the base vector. Does the flip keep other answers right?
B. Mechanism of the flip: split the vector's logit effect into a constant per-token bias and a pair-dependent part. Which one flips the rule?
C. Carry axis: with random previous digits and written digit, slopes of neuron inputs on a, b, written. Are they one axis? Does carry vector = axis x base?
D. Carry rule on held-out data, and where its errors sit relative to the threshold.
E. Tracing the base signal: patch L0H1 only at query, a digits, b digits, '+'. Are the input shifts additive, and how much does each route carry?
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, render, sample_shots, save_json, to_base
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]
Wc = sd["h.1.mlp.c_fc.weight"]; bc = sd["h.1.mlp.c_fc.bias"]
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; a, c_ = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ Wc + bc
    return dict(a=a, b=c_, c=c, s=a + c_ + c, mid=mid, pre=pre, idx=t.meta["idx"][rows])
L = lambda mid, pre: model.unembed(mid + gelu_new(pre) @ Wo + bo)
D = {(b, k): load(b, k) for b in BASES for k in (0, 1)}
def base_vec(x, y, k, half):
    dx, dy = D[(x, k)], D[(y, k)]; lim = min(x, y); ds = []
    hx, hy = dx["idx"] % 2 == half, dy["idx"] % 2 == half
    for i in range(lim):
        for j in range(lim):
            for c in (0, 1):
                m1 = hx & (dx["a"] == i) & (dx["b"] == j) & (dx["c"] == c); m2 = hy & (dy["a"] == i) & (dy["b"] == j) & (dy["c"] == c)
                if m1.sum() >= 2 and m2.sum() >= 2: ds.append(dy["pre"][torch.as_tensor(m2, device=DEVICE)].mean(0) - dx["pre"][torch.as_tensor(m1, device=DEVICE)].mean(0))
    return torch.stack(ds).mean(0)
for k in (0, 1):
    for x, y in [(8, 10), (10, 8), (10, 16), (16, 10), (8, 16), (16, 8)]:
        dx = D[(x, k)]; v = base_vec(x, y, k, 0); te = dx["idx"] % 2 == 1; lim = min(x, y)
        inb = te & (dx["a"] < lim) & (dx["b"] < lim); s = dx["s"]
        conf, non = inb & (s >= lim), inb & (s < lim)
        mt = lambda m: torch.as_tensor(m, device=DEVICE)
        pr_c = L(dx["mid"][mt(conf)], dx["pre"][mt(conf)] + v).argmax(-1).cpu().numpy()
        pr_n = L(dx["mid"][mt(non)], dx["pre"][mt(non)] + v).argmax(-1).cpu().numpy()
        rx, ry = tok[s[conf] % x], tok[s[conf] % y]; dd = rx != ry
        out = {"flip_to_other_rule": float((pr_c[dd] == ry[dd]).mean()), "non_conflict_unchanged": float((pr_n == tok[s[non] % x]).mean())}
        # random controls
        g = torch.Generator(device=DEVICE).manual_seed(0); rand = []
        for _ in range(20):
            r = torch.randn(256, device=DEVICE, generator=g); r = r / r.norm() * v.norm()
            p = L(dx["mid"][mt(conf)], dx["pre"][mt(conf)] + r).argmax(-1).cpu().numpy()
            rand.append(((p[dd] == ry[dd]).mean(), (p == rx).mean()))
        out["random_vector_flip"] = float(np.mean([a for a, _ in rand])); out["random_vector_keeps_own"] = float(np.mean([b_ for _, b_ in rand]))
        # B token-level bias only
        base_lg = L(dx["mid"][mt(te)], dx["pre"][mt(te)]); sh_lg = L(dx["mid"][mt(te)], dx["pre"][mt(te)] + v)
        bias = (sh_lg - base_lg).mean(0)
        lg_c = L(dx["mid"][mt(conf)], dx["pre"][mt(conf)]); lg_sh = L(dx["mid"][mt(conf)], dx["pre"][mt(conf)] + v)
        p_bias = (lg_c + bias).argmax(-1).cpu().numpy()
        out["flip_with_constant_token_bias_only"] = float((p_bias[dd] == ry[dd]).mean())
        p_cell = (lg_sh - bias).argmax(-1).cpu().numpy()
        out["flip_with_pair_dependent_part_only"] = float((p_cell[dd] == ry[dd]).mean())
        out["bias_share_of_logit_change"] = float((bias**2).sum() * len(lg_c) / ((lg_sh - lg_c) ** 2).sum())
        out["bias_by_token"] = bias[torch.as_tensor(tok, device=DEVICE)].cpu().numpy().round(2).tolist()
        res["A"][f"k{k}/{x}->{y}"] = out
        print(f"A k{k} {x}->{y}", {a: round(c, 2) for a, c in out.items() if a != "bias_by_token"}, flush=True)
        if k == 0: print("     token bias 0..F:", out["bias_by_token"], flush=True)
    save_json("v05_vector_checks", res)
# carry vector random controls
for b in BASES:
    d = D[(b, 1)]; half = d["idx"] % 2 == 0; ds = []
    for i in range(b):
        for j in range(b):
            m0, m1 = half & (d["a"] == i) & (d["b"] == j) & (d["c"] == 0), half & (d["a"] == i) & (d["b"] == j) & (d["c"] == 1)
            if m0.sum() >= 2 and m1.sum() >= 2: ds.append(d["pre"][torch.as_tensor(m1, device=DEVICE)].mean(0) - d["pre"][torch.as_tensor(m0, device=DEVICE)].mean(0))
    v = torch.stack(ds).mean(0); te = (~half) & (d["c"] == 0); mt = torch.as_tensor(te, device=DEVICE)
    alt = tok[(d["s"][te] + 1) % b]; own = tok[d["s"][te] % b]
    p = L(d["mid"][mt], d["pre"][mt] + v).argmax(-1).cpu().numpy()
    g = torch.Generator(device=DEVICE).manual_seed(1); fl = []
    for _ in range(20):
        r = torch.randn(256, device=DEVICE, generator=g); r = r / r.norm() * v.norm(); pr = L(d["mid"][mt], d["pre"][mt] + r).argmax(-1).cpu().numpy(); fl.append((pr == alt).mean())
    res["C"][f"carry_flip_{b}"] = {"flip": float((p == alt).mean()), "random_flip": float(np.mean(fl)), "vector": v.cpu().numpy()}
    print(f"C carry vector b{b}: flip {float((p == alt).mean()):.2f}, random vectors {float(np.mean(fl)):.2f}", flush=True)
# C axis + D rule via forward passes
def rows_random(base, k, n, seed):
    rng = random.Random(seed); S, rec = [], []
    while len(S) < n:
        a, c_ = rng.randint(base**k, 999), rng.randint(base**k, 999); ap, bp, w = rng.randrange(base), rng.randrange(base), rng.randrange(base)
        A = a + (ap - (a // base ** (k - 1)) % base) * base ** (k - 1); B = c_ + (bp - (c_ // base ** (k - 1)) % base) * base ** (k - 1)
        sa, sb = to_base(A, base), to_base(B, base)
        if len(sa) <= k or len(sb) <= k: continue
        pref = to_base(A + B, base)[::-1][:k - 1] + DIGITS[w]
        S.append("\n".join(render(*s, base) for s in sample_shots(rng, (A, B))) + f"\n{sa}+{sb}={pref}")
        rec.append((ap, bp, w, (A // base**k) % base, (B // base**k) % base))
    return S, np.array(rec)
def forward(S, name="resid_mid1"):
    pre, lg = [], []
    for i in range(0, len(S), 256):
        seqs = [[model.vocab[c] for c in s] for s in S[i:i + 256]]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); J = torch.arange(len(seqs), device=DEVICE)
        l, c = model.forward(x.to(DEVICE), capture={name}); m = c[name][J, q]
        pre.append(model._ln("h.1.ln_2", m) @ Wc + bc); lg.append(l[J, q])
    return torch.cat(pre).cpu().numpy(), torch.cat(lg).cpu().numpy()
axes = {}
for b in (8, 10, 16, 2):
    S, rec = rows_random(b, 1, 5000, b * 3); pre, lg = forward(S)
    ap, bp, w, ca, cb = rec.T
    X = np.c_[np.ones(len(ap)), np.eye(16)[ca][:, 1:], np.eye(16)[cb][:, 1:], ap, bp, w].astype(np.float32)
    co = np.linalg.lstsq(X, pre, rcond=None)[0]; ua, ub, uw = co[-3], co[-2], co[-1]
    cos = lambda x, y: float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y)))
    v = res["C"][f"carry_flip_{b}"]["vector"]; ax = (ua + ub - uw) / 3
    res["C"][b] = {"cos_a_b": cos(ua, ub), "cos_a_minus_written": cos(ua, -uw), "cos_b_minus_written": cos(ub, -uw), "cos_axis_carry_vector": cos(ax, v), "norm_carry_over_base_x_axis": float(np.linalg.norm(v) / (b * np.linalg.norm(ax))), "norms": [float(np.linalg.norm(ua)), float(np.linalg.norm(ub)), float(np.linalg.norm(uw))]}
    axes[b] = ax
    print("C axis", b, {k2: (round(x, 2) if not isinstance(x, list) else np.round(x, 2).tolist()) for k2, x in res["C"][b].items()}, flush=True)
    if b != 2:
        cur = ca + cb; J = np.arange(len(lg)); ev = lg[J, tok[(cur + 1) % b]] - lg[J, tok[cur % b]]; pr = lg.argmax(1); one = pr == tok[(cur + 1) % b]; val = one | (pr == tok[cur % b])
        lin = ap + bp - w; trn = J % 2 == 0
        best = max(((((lin[trn & val] > th) == one[trn & val]).mean()), th) for th in np.arange(-b, 2 * b) + .5)
        th = best[1]; te_ = (~trn) & val; ok = (lin[te_] > th) == one[te_]
        dist = np.abs(lin[te_] - th); tab = {}
        for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 5), (5, 99)):
            m = (dist >= lo) & (dist < hi)
            if m.sum(): tab[f"{lo}-{hi}"] = (float(ok[m].mean()), int(m.sum()))
        res["D"][b] = {"threshold": float(th), "held_out_rule_accuracy": float(ok.mean()), "accuracy_by_distance_from_threshold": tab}
        print("D", b, "threshold", th, "held-out", round(float(ok.mean()), 3), "by distance", {a: (round(x[0], 2), x[1]) for a, x in tab.items()}, flush=True)
print("axis cos across bases", {f"{x}v{y}": round(float(axes[x] @ axes[y] / (np.linalg.norm(axes[x]) * np.linalg.norm(axes[y]))), 2) for x, y in [(8, 10), (10, 16), (8, 16), (2, 10)]}, flush=True)
res["C"]["axis_cos"] = {f"{x}v{y}": float(axes[x] @ axes[y] / (np.linalg.norm(axes[x]) * np.linalg.norm(axes[y]))) for x, y in [(8, 10), (10, 16), (8, 16), (2, 10)]}
for k2 in list(res["C"]):
    if str(k2).startswith("carry_flip"): res["C"][k2].pop("vector", None)
save_json("v05_vector_checks", res)
# E trace L0H1
for low, high, ch, ne in [(8, 10, "9", 4), (8, 16, "A", 3), (10, 16, "F", 3)]:
    rng = random.Random(low * 9 + high); C, E = [], []
    while len(C) < 800:
        ax, bx = rng.randint(0, low - 1), rng.randint(0, low - 1)
        if ax + bx < low: continue
        a, c_ = rng.randint(1, 999 // low) * low + ax, rng.randint(1, 999 // low) * low + bx
        ctx = "\n".join(render(*s, low) for s in sample_shots(rng, (a, c_))); e = list(ctx); pos = [j for j, x in enumerate(e) if x not in "+=\n"]
        for j in rng.sample(pos, ne): e[j] = ch
        t = f"\n{to_base(a, low)}+{to_base(c_, low)}="; C.append(ctx + t); E.append("".join(e) + t)
    all_d = {r: [] for r in ("query", "a_digits", "b_digits", "plus", "all_test_line", "true_edit")}
    for i in range(0, len(C), 200):
        cs, es = C[i:i + 200], E[i:i + 200]; seqs = [[model.vocab[c] for c in s] for s in cs]; n = max(map(len, seqs))
        xc = torch.full((len(cs), n), model.pad, dtype=torch.long); xe = xc.clone()
        for j, (s1, s2) in enumerate(zip(cs, es)):
            xc[j, :len(s1)] = torch.tensor([model.vocab[c] for c in s1]); xe[j, :len(s2)] = torch.tensor([model.vocab[c] for c in s2])
        xc, xe = xc.to(DEVICE), xe.to(DEVICE); q = torch.tensor([len(s) - 1 for s in cs], device=DEVICE); J = torch.arange(len(cs), device=DEVICE)
        st = torch.tensor([s.rfind("\n") + 1 for s in cs], device=DEVICE); pl = torch.tensor([s.rfind("+") for s in cs], device=DEVICE); idx = torch.arange(n, device=DEVICE)[None]
        masks = {"query": idx == q[:, None], "a_digits": (idx >= st[:, None]) & (idx < pl[:, None]), "b_digits": (idx > pl[:, None]) & (idx < q[:, None]), "plus": idx == pl[:, None], "all_test_line": (idx >= st[:, None]) & (idx <= q[:, None])}
        _, cc = model.forward(xc, capture={"resid_mid1"}); _, ce = model.forward(xe, capture={"z0", "resid_mid1"})
        base_pre = model._ln("h.1.ln_2", cc["resid_mid1"][J, q]) @ Wc + bc
        all_d["true_edit"].append(model._ln("h.1.ln_2", ce["resid_mid1"][J, q]) @ Wc + bc - base_pre)
        for r, m in masks.items():
            def hook(t, m=m, s=ce["z0"]):
                t = t.clone(); t[:, :, 1][m] = s[:, :, 1][m]; return t
            _, cp = model.forward(xc, hooks={"z0": hook}, capture={"resid_mid1"})
            all_d[r].append(model._ln("h.1.ln_2", cp["resid_mid1"][J, q]) @ Wc + bc - base_pre)
    mean = {r: torch.cat(v).mean(0) for r, v in all_d.items()}
    truth = mean["true_edit"]; cos = lambda x, y: float(torch.dot(x, y) / (x.norm() * y.norm() + 1e-9))
    parts = mean["query"] + mean["a_digits"] + mean["b_digits"] + mean["plus"]
    res["E"][f"{low}->{high}"] = {"cos_all_L0H1_vs_true_edit": cos(mean["all_test_line"], truth), "norm_ratio_all_vs_true": float(mean["all_test_line"].norm() / truth.norm()), "additivity_cos_parts_vs_all": cos(parts, mean["all_test_line"]), "additivity_norm_ratio": float(parts.norm() / mean["all_test_line"].norm()), "route_projection_on_true": {r: float(torch.dot(mean[r], truth) / truth.pow(2).sum()) for r in ("query", "a_digits", "b_digits", "plus")}}
    print("E", low, high, res["E"][f"{low}->{high}"], flush=True)
save_json("v05_vector_checks", res); print("DONE")
