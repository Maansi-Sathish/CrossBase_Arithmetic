"""F18 (forward passes): do the same mechanisms appear in other trained architectures?

Pre-registered predictions, from L2_H4_D64:
 P1 Clock: logits depend on (t-s) mod b, off-target R^2 well above shuffled tokens, symmetric, parity the top frequency.
 P2 Largest digit: a decimal or hex context showing only digits 0-7 gets the octal rule.
 P3 Base bias: one constant vector added to last-layer MLP inputs flips the rule (conflict trials).
 P4 Carry vector: one constant vector flips carry, and it's shared across 8, 10, 16 (cosine).
 P5 Linear carry: carry evidence ~ a + b - written, threshold near base/2.
 P6 Detector: one first-layer head prefers large digits and carries the base alone (2-layer+ models).
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, greedy, render, sample_problems, sample_shots, save_json, schoolbook, to_answer, to_base
torch.set_grad_enabled(False)
MODELS = sys.argv[1:] or ["L2_H4_D64", "L2_H8_D64", "L2_H4_D128", "L2_H2_D64", "L3_H2_D64", "L1_H4_D64", "L2_H8_D128", "L8_H4_D64"]
res = {}
tok = np.array([list("0123456789ABCDEF")]).ravel()


def build(model, strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]
    L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
    return x.to(model.device), torch.tensor([len(s) - 1 for s in seqs], device=model.device)


def lastparts(model, strs, bs=128):
    """logits, resid_mid and pre at the query, last layer."""
    L = model.n_layers - 1; out = {"lg": [], "mid": [], "pre": []}
    for i in range(0, len(strs), bs):
        x, q = build(model, strs[i:i + bs]); lg, c = model.forward(x, capture={f"resid_mid{L}"})
        r = torch.arange(len(x), device=model.device); mid = c[f"resid_mid{L}"][r, q]
        pre = model._ln(f"h.{L}.ln_2", mid) @ model.sd[f"h.{L}.mlp.c_fc.weight"] + model.sd[f"h.{L}.mlp.c_fc.bias"]
        out["lg"].append(lg[r, q].cpu()); out["mid"].append(mid); out["pre"].append(pre)
    return {k: torch.cat(v) for k, v in out.items()}


for name in MODELS:
    mid_ = f"CrossBaseArithmetic/arithmetic-model-{name}"
    try:
        model = Model(model_id=mid_)
    except Exception as e:
        print(name, "LOAD FAILED", repr(e)[:150], flush=True); continue
    T = torch.as_tensor(np.array([model.vocab[c] for c in DIGITS]), device=model.device); t_np = T.cpu().numpy()
    Wo = model.sd[f"h.{model.n_layers - 1}.mlp.c_proj.weight"]; bo = model.sd[f"h.{model.n_layers - 1}.mlp.c_proj.bias"]
    logits_of = lambda mid, pre: model.unembed(mid + gelu_new(pre) @ Wo + bo)
    r = {"accuracy": {}}
    for b in BASES:
        P = sample_problems(600, b, seed=3)
        r["accuracy"][b] = float(np.mean([o == p.answer for o, p in zip(greedy(model, P, bs=128), P)]))
    print(f"\n=== {name}: accuracy", {k: round(v, 3) for k, v in r["accuracy"].items()}, flush=True)
    res[name] = r
    if min(r["accuracy"].values()) < 0.8:
        print("  accuracy too low in some base; analyses still run for bases with accuracy >= 0.8", flush=True)
    good = [b for b in BASES if r["accuracy"][b] >= 0.8]
    data = {}
    for b in good:
        rows = {}
        for k in (0, 1):
            P = [p for p in sample_problems(3000, b, seed=11) if p.a + p.b >= b**k and len(to_base(p.a, b)) > k and len(to_base(p.b, b)) > k][:1600]
            S = [p.prompt + to_answer(p.a + p.b, b)[:k] for p in P]
            parts = lastparts(model, S)
            vals = np.array([[schoolbook(p.a, p.b, b)[v][k] for v in ("a", "b", "c_in")] for p in P])
            rows[k] = {**parts, "a": vals[:, 0], "b": vals[:, 1], "c": vals[:, 2]}
        data[b] = rows
    # P1
    r["P1"] = {}
    for b in good:
        if b == 2: continue
        d = data[b][0]; o = (d["a"] + d["b"]) % b
        Lc = d["lg"].numpy()[:, t_np[:b]]; Lc = Lc - Lc.mean(1, keepdims=True)
        off = (np.arange(b)[None] - o[:, None]) % b; g = np.array([Lc[off == x].mean() for x in range(b)]); m = off != 0
        r2 = 1 - ((Lc[m] - g[off][m]) ** 2).sum() / ((Lc[m] - Lc[m].mean()) ** 2).sum()
        rng = np.random.default_rng(0); nl = []
        for _ in range(30):
            perm = rng.permutation(b); of2 = (perm[np.arange(b)][None] - perm[o][:, None]) % b
            g2 = np.array([Lc[of2 == x].mean() for x in range(b)]); m2 = of2 != 0
            nl.append(1 - ((Lc[m2] - g2[of2][m2]) ** 2).sum() / ((Lc[m2] - Lc[m2].mean()) ** 2).sum())
        P_ = np.abs(np.fft.rfft(g - g.mean())[1:]) ** 2
        r["P1"][b] = {"r2": float(r2), "null": float(np.mean(nl)), "sym": float(np.corrcoef(g[1:], g[1:][::-1])[0, 1]), "parity_share": float(P_[-1] / P_.sum()), "top_f": int(P_.argmax() + 1)}
    print("  P1 clock (R2, null, symmetry, parity share):", {b: (round(v['r2'], 2), round(v['null'], 2), round(v['sym'], 2), round(v['parity_share'], 2)) for b, v in r["P1"].items()}, flush=True)
    # P2
    rng = random.Random(9); S = []; A0 = []
    if 8 in good:
        for t in (10, 16):
            S, s0 = [], []
            while len(S) < 500:
                ax, bx = rng.randint(0, 7), rng.randint(0, 7)
                if ax + bx < 8: continue
                lines = []
                while len(lines) < 5:
                    l = render(rng.randint(0, 999), rng.randint(0, 999), t)
                    if set(l) <= set("01234567+="): lines.append(l)
                a, b_ = rng.randint(1, 63) * 8 + ax, rng.randint(1, 63) * 8 + bx
                S.append("\n".join(lines) + f"\n{to_base(a, 8)}+{to_base(b_, 8)}="); s0.append(ax + bx)
            pr = lastparts(model, S)["lg"].argmax(-1).numpy(); s0 = np.array(s0)
            r.setdefault("P2", {})[t] = float((pr == t_np[s0 % 8]).mean())
        print("  P2 only digits 0-7 shown: follows octal rule", {k: round(v, 3) for k, v in r["P2"].items()}, flush=True)
    # P3
    r["P3"] = {}
    for x, y in [(8, 10), (10, 16), (8, 16), (16, 8)]:
        if x not in good or y not in good: continue
        for k in (0, 1):
            dx, dy = data[x][k], data[y][k]; lim = min(x, y); h = np.arange(len(dx["a"])) % 2 == 0; h2 = np.arange(len(dy["a"])) % 2 == 0; ds = []
            for i in range(lim):
                for j in range(lim):
                    for c in (0, 1):
                        m1 = h & (dx["a"] == i) & (dx["b"] == j) & (dx["c"] == c); m2 = h2 & (dy["a"] == i) & (dy["b"] == j) & (dy["c"] == c)
                        if m1.sum() and m2.sum(): ds.append(dy["pre"][torch.as_tensor(m2, device=model.device)].mean(0) - dx["pre"][torch.as_tensor(m1, device=model.device)].mean(0))
            v = torch.stack(ds).mean(0); s = dx["a"] + dx["b"] + dx["c"]; te = (~h) & (dx["a"] < lim) & (dx["b"] < lim) & (s >= lim)
            tt = torch.as_tensor(te, device=model.device); rx, ry = t_np[s[te] % x], t_np[s[te] % y]; dd = rx != ry
            pr = logits_of(dx["mid"][tt], dx["pre"][tt] + v).argmax(-1).cpu().numpy()
            r["P3"][f"k{k}/{x}->{y}"] = float((pr[dd] == ry[dd]).mean())
    print("  P3 constant base vector flips rule:", {k: round(v, 2) for k, v in r["P3"].items()}, flush=True)
    # P4
    vs = {}; r["P4"] = {"flip": {}, "cos": {}}
    for b in good:
        d = data[b][1]; ds = []
        for i in range(b):
            for j in range(b):
                m1 = (d["a"] == i) & (d["b"] == j) & (d["c"] == 0); m2 = (d["a"] == i) & (d["b"] == j) & (d["c"] == 1)
                if m1.sum() and m2.sum(): ds.append(d["pre"][torch.as_tensor(m2, device=model.device)].mean(0) - d["pre"][torch.as_tensor(m1, device=model.device)].mean(0))
        v = torch.stack(ds).mean(0); vs[b] = v
        m = d["c"] == 0; mt = torch.as_tensor(m, device=model.device); alt = t_np[(d["a"][m] + d["b"][m] + 1) % b]
        r["P4"]["flip"][b] = float((logits_of(d["mid"][mt], d["pre"][mt] + v).argmax(-1).cpu().numpy() == alt).mean())
    for x, y in [(8, 10), (10, 16), (8, 16), (2, 10)]:
        if x in vs and y in vs: r["P4"]["cos"][f"{x}v{y}"] = float(torch.nn.functional.cosine_similarity(vs[x], vs[y], dim=0))
    print("  P4 carry vector flip:", {k: round(v, 2) for k, v in r["P4"]["flip"].items()}, "cos", {k: round(v, 2) for k, v in r["P4"]["cos"].items()}, flush=True)
    # P5
    r["P5"] = {}
    for b in (8, 10, 16):
        if b not in good: continue
        rng = random.Random(b); S, rec = [], []
        while len(S) < 2500:
            a, c_ = rng.randint(b, 999), rng.randint(b, 999); ap, bp, w = rng.randrange(b), rng.randrange(b), rng.randrange(b)
            A, B = a - a % b + ap, c_ - c_ % b + bp
            S.append("\n".join(render(*s, b) for s in sample_shots(rng, (A, B))) + f"\n{to_base(A, b)}+{to_base(B, b)}={DIGITS[w]}")
            rec.append((ap, bp, w, (A // b) % b + (B // b) % b))
        Lg = lastparts(model, S)["lg"].numpy(); rec = np.array(rec); ap, bp, w, cur = rec.T
        J = np.arange(len(Lg)); ev = Lg[J, t_np[(cur + 1) % b]] - Lg[J, t_np[cur % b]]
        X = np.c_[ap, bp, w, np.ones(len(ap))].astype(float); co = np.linalg.lstsq(X, ev, rcond=None)[0]
        r2 = 1 - ((ev - X @ co) ** 2).sum() / ((ev - ev.mean()) ** 2).sum()
        pr = Lg.argmax(1); ch1 = pr == t_np[(cur + 1) % b]; val = ch1 | (pr == t_np[cur % b]); lin = ap + bp - w
        best = max((float(((lin > th) == ch1)[val].mean()), float(th)) for th in np.arange(-b, 2 * b) + .5)
        r["P5"][b] = {"a": co[0], "b": co[1], "w": co[2], "r2": float(r2), "rule": best[0], "thr_over_base": best[1] / b}
    print("  P5 linear carry (a, b, w, R2, threshold/base):", {b: (round(v['a'], 2), round(v['b'], 2), round(v['w'], 2), round(v['r2'], 2), round(v['thr_over_base'], 2)) for b, v in r["P5"].items()}, flush=True)
    # P6
    if 16 in good:
        P = sample_problems(600, 16, seed=7); tot = np.zeros((model.n_heads, 16)); cnt = np.zeros(16)
        for i in range(0, len(P), 64):
            S = [p.prompt for p in P[i:i + 64]]; x, q = build(model, S); _, c = model.forward(x, capture={"pattern0"}); pat = c["pattern0"].cpu().numpy()
            for j, s in enumerate(S):
                st = s.rfind("\n") + 1
                ids = [z for z in range(st) if s[z] in DIGITS]; vals = np.array([DIGITS.index(s[z]) for z in ids]); row = pat[j, :, q[j].item()][:, ids]
                for v in np.unique(vals):
                    tot[:, v] += row[:, vals == v].sum(1); cnt[v] += (vals == v).sum()
        per = tot / np.maximum(cnt, 1); ratio = per[:, 10:].mean(1) / (per[:, :8].mean(1) + 1e-6); best = int(ratio.argmax())
        r["P6"] = {"best_head": best, "letters_over_low": float(ratio[best]), "all_ratios": ratio.tolist()}
        print("  P6 best large-digit head H%d, letters / digits-0-7 attention = %.1f; all heads %s" % (best, ratio[best], np.round(ratio, 1).tolist()), flush=True)
        if 8 in good and model.n_layers >= 1:
            rng = random.Random(2); C, E, s0 = [], [], []
            while len(C) < 700:
                ax, bx = rng.randint(0, 7), rng.randint(0, 7)
                if ax + bx < 8: continue
                a, b_ = rng.randint(1, 120) * 8 + ax, rng.randint(1, 120) * 8 + bx
                ctx = list("\n".join(render(*s, 8) for s in sample_shots(rng, (a, b_)))); t = f"\n{to_base(a, 8)}+{to_base(b_, 8)}="
                base_ctx = "".join(ctx); pos = [i for i, ch in enumerate(ctx) if ch not in "+=\n"]
                for z in rng.sample(pos, 3): ctx[z] = "A"
                C.append(base_ctx + t); E.append("".join(ctx) + t); s0.append(ax + bx)
            s0 = np.array(s0); out = {}
            for h in range(model.n_heads):
                flips = []
                for i in range(0, len(C), 128):
                    xc, q = build(model, C[i:i + 128]); xe, _ = build(model, E[i:i + 128]); _, ce = model.forward(xe, capture={"z0"})
                    idx = torch.arange(xc.shape[1], device=model.device)[None]; st = torch.tensor([s.rfind("\n") + 1 for s in C[i:i + 128]], device=model.device)
                    mask = (idx >= st[:, None]) & (idx <= q[:, None])
                    def hook(t, m=mask, s=ce["z0"], h=h):
                        t = t.clone(); t[:, :, h][m] = s[:, :, h][m]; return t
                    lg, _ = model.forward(xc, hooks={"z0": hook}); flips.append(lg[torch.arange(len(xc), device=model.device), q].argmax(-1).cpu().numpy())
                pr = np.concatenate(flips); out[h] = float((pr != t_np[s0 % 8]).mean())
            r["P6"]["head_patch_changes_octal_answer"] = out
            print("  P6 single head patch moves answer off octal rule:", {k: round(v, 2) for k, v in out.items()}, flush=True)
    save_json("f18_replication", res)
print("DONE")
