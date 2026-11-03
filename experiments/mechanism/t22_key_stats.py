"""T22 (inference and stored activations only): 95% intervals and controls for the key results.

Run one part at a time:  python t22_key_stats.py <part>
Parts (report section in brackets):
  saved      [2.1, 3.1] Wilson intervals from saved counts: held-out accuracy; best "foreign residue" probe vs chance.
  base_patch [4.2] L0H1 patching, with a matched control: the donor's context gets the same number of edits, but with
             digits of the prompt's own base, so only the tier differs.
  operand    [5.4] Blocking '+' at b's digits, with controls that block a different single token (the line start, a's first digit).
  recency    [5.5] Reversing L0H3's outputs over a's digits, with the other three layer 0 heads reversed as controls.
  carry_rule [6.4] Bootstrap intervals for the carry-rule weights, threshold and held-out accuracy; rules that ignore the
             written digit or the operands as controls.
  carry_vec  [6.3] Carry vector with Wilson intervals; 200 random vectors of equal size; vector learned from shuffled carry labels.
  formula    [7.1] Clean replacement by mu + A[a] + B[b] + c*C + V[base], with tables fitted on shuffled digit labels as a control.
  clock      [7.7] Bootstrap over prompts for the competing fits of off-target logits, and the wrap-around comparison.
  base_vec   [8.1] Base vector with Wilson intervals; 200 random vectors; a same-size vector learned from a random split of one base's prompts.
  offset     [8.2] Bootstrap over prompts for offset-only / gain-only fits.
Intervals: Wilson 95% for proportions; percentile bootstrap (resampling prompts) for everything else.
"""
import json, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DATA, DEVICE, DIGITS, Model, gelu_new, render, sample_shots, save_json, to_answer, to_base, wilson
from probe_lib import QUERY, Table

torch.set_grad_enabled(False)
PART = sys.argv[1]
OUT = DATA / "t22_key_stats.json"
res = json.loads(OUT.read_text()) if OUT.exists() else {}
model = Model(); sd = model.sd; PE = model.PE
tok = np.array([model.vocab[c] for c in DIGITS])
Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]; Wc = sd["h.1.mlp.c_fc.weight"]; bc = sd["h.1.mlp.c_fc.bias"]
NB = 1000  # bootstrap resamples


def W(hits):
    """Proportion with Wilson 95% interval and n."""
    hits = np.asarray(hits, bool); p, lo, hi = wilson(int(hits.sum()), len(hits))
    return {"p": p, "lo": lo, "hi": hi, "n": int(len(hits))}


def Wkn(k, n):
    p, lo, hi = wilson(int(round(k)), int(n)); return {"p": p, "lo": lo, "hi": hi, "n": int(n)}


def boot(vals):
    """Point estimate (first entry) with the 2.5 / 97.5 percentiles of the rest."""
    v = np.asarray(vals, float); return {"est": float(v[0]), "lo": float(np.percentile(v[1:], 2.5)), "hi": float(np.percentile(v[1:], 97.5))}


def fmt(d):
    if "p" in d and d["n"] == 0: return "n/a (n=0)"
    if "p" in d: return f"{d['p']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}] n={d['n']}"
    return f"{d['est']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def ids(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]; n = max(map(len, seqs))
    x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
    return x.to(DEVICE)


def load_stored(b, k):
    """Stored activations at the column-k query, for correct prompts with a column-k answer digit."""
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; a, c_ = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ Wc + bc
    return dict(a=a, b=c_, c=c, s=a + c_ + c, mid=mid, pre=pre, idx=t.meta["idx"][rows])


L_out = lambda mid, pre: model.unembed(mid + gelu_new(pre) @ Wo + bo)
mt = lambda m: torch.as_tensor(m, device=DEVICE)

# ============================================================================================ saved
if PART == "saved":
    out = {}
    f01 = json.loads((DATA / "f01_heldout.json").read_text())
    out["f01_raw"] = f01
    s01 = json.loads((DATA / "s01_residues.json").read_text())["results"]
    probes = {}
    FE = ["resid0", "resid1", "mlp0", "mlp1"]
    for M, bases in ((3, BASES), (7, BASES), (9, BASES), (5, [2, 8, 16]), (10, [2, 8, 16]), (16, [10])):
        best = None
        for b in bases:
            for site, tg in s01[str(b)].items():
                for tn, feats in tg.items():
                    for f in FE:
                        r = feats[f][str(M)]
                        if best is None or r["acc"] > best[0]: best = (r["acc"], r["chance"], r["n_test"], b, site, tn, f)
        acc, ch, n, *where = best
        probes[f"mod {M}"] = {"best": Wkn(acc * n, n), "guessing": ch, "where": where,
                              "note": "best over all sites, targets, activation types and bases; n is one split's test size"}
        print(f"mod {M}: best {fmt(probes[f'mod {M}']['best'])}  guessing {ch:.3f}  at {where}", flush=True)
    # Control for binary mod 10: n mod 10 is partly fixed by the last bit (n mod 2). Best guess of (a+b) mod 10 from the low bits alone.
    from probe_lib import integer_split
    t = Table(2); rows = t.select(QUERY, 0); nn = (t.meta["a"][rows] + t.meta["b"][rows]).astype(int); ctl = {}
    for bits in (1, 2, 3):
        m = 2**bits; accs = []
        for seed in range(3):
            tr, te = integer_split(nn, seed=seed)
            guess = {r: np.bincount(nn[tr][nn[tr] % m == r] % 10, minlength=10).argmax() for r in range(m)}
            accs.append(float((np.array([guess[v % m] for v in nn[te]]) == nn[te] % 10).mean()))
        ctl[f"last {bits} bit(s)"] = float(np.mean(accs))
    probes["CONTROL binary (a+b) mod 10 guessed from the low bits only"] = ctl
    print("binary mod 10 from low bits only:", {k: round(v, 3) for k, v in ctl.items()}, flush=True)
    out["probes"] = probes
    res["saved"] = out

# ============================================================================================ base_patch
if PART == "base_patch":
    PAIRS = [(8, 10, "9", 4), (8, 16, "A", 2), (10, 16, "F", 2)]

    def make(n, seed, low):
        rng = random.Random(seed); out = []
        while len(out) < n:
            ax, bx = rng.randint(0, low - 1), rng.randint(0, low - 1)
            if ax + bx < low: continue
            a = rng.randint(1, 999 // low) * low + ax; b = rng.randint(1, 999 // low) * low + bx
            ctx = "\n".join(render(*s, low) for s in sample_shots(rng, (a, b)))
            out.append(dict(ctx=ctx, test=f"{to_base(a, low)}+{to_base(b, low)}=", s=ax + bx))
        return out

    def masks(strs, x):
        n = x.shape[1]; idx = torch.arange(n, device=DEVICE)[None]
        q = torch.tensor([len(s) - 1 for s in strs], device=DEVICE); st = torch.tensor([s.rfind("\n") + 1 for s in strs], device=DEVICE)
        return q, (idx >= st[:, None]) & (idx <= q[:, None])

    def run(strs, head=None, src=None):
        """Predict at '='. If head is given, copy z0[:, :, head] on the test line from the donor strings `src`."""
        out = []
        for i in range(0, len(strs), 256):
            ch = strs[i:i + 256]; x = ids(ch); q, m = masks(ch, x); hooks = None
            if head is not None:
                _, c = model.forward(ids(src[i:i + 256]), capture={"z0"}); zs = c["z0"]
                def fn(t, m=m, zs=zs):
                    t = t.clone(); t[:, :, head][m] = zs[:, :, head][m]; return t
                hooks = {"z0": fn}
            lg, _ = model.forward(x, hooks=hooks); out.append(lg[torch.arange(len(ch), device=DEVICE), q].argmax(-1).cpu().numpy())
        return np.concatenate(out)

    out = {}
    for low, high, ch, n_edit in PAIRS:
        P = make(1600, low * 100 + high, low)
        own = [d for d in DIGITS[:low] if d != ch]
        clean, edited, control = [], [], []
        for i, p in enumerate(P):
            r1, r2 = random.Random(i), random.Random(i)  # same positions for the real edit and the control edit
            pos_e = list(p["ctx"]); idxs = [j for j, c_ in enumerate(pos_e) if c_ not in "+=\n"]
            picks = r1.sample(idxs, min(n_edit, len(idxs)))
            e, c2 = list(p["ctx"]), list(p["ctx"])
            for j in picks: e[j] = ch; c2[j] = r2.choice(own)
            clean.append(p["ctx"] + "\n" + p["test"]); edited.append("".join(e) + "\n" + p["test"]); control.append("".join(c2) + "\n" + p["test"])
        s = np.array([p["s"] for p in P]); rl, rh = tok[s % low], tok[s % high]
        pc, pe, pk = run(clean), run(edited), run(control)
        use = np.where((pc == rl) & (pe == rh) & (rl != rh) & (pk == rl))[0][:800]
        C = [clean[i] for i in use]; E = [edited[i] for i in use]; K = [control[i] for i in use]; wl, wh = rl[use], rh[use]
        r = {"usable": int(len(use))}
        for h in range(4):
            r[f"L0H{h}: clean gets edited -> high rule"] = W(run(C, h, E) == wh)
            r[f"L0H{h}: edited gets clean -> low rule"] = W(run(E, h, C) == wl)
        r["CONTROL L0H1: clean gets matched same-tier edit -> high rule"] = W(run(C, 1, K) == wh)
        r["CONTROL L0H1: clean gets matched same-tier edit -> low rule kept"] = W(run(C, 1, K) == wl)
        out[f"{low}->{high}"] = r
        print(f"\n{low}->{high} usable {len(use)}", flush=True)
        for k_, v in r.items():
            if k_ != "usable": print(f"   {k_:62s} {fmt(v)}", flush=True)
    res["base_patch"] = out

# ============================================================================================ operand / recency
def make_equal(b, n, seed, minlen):
    rng = random.Random(seed); S, info = [], []
    while len(S) < n:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b)
        if len(sa) != len(sb) or len(sa) < minlen: continue
        ctx = "\n".join(render(*x, b) for x in sample_shots(rng, (a, c_))); st = len(ctx) + 1; L = len(sa)
        s = f"{ctx}\n{sa}+{sb}="
        if len(s) > 250: continue
        S.append(s); info.append((st, st + L, st + 2 * L + 1, int(to_answer(a + c_, b)[0], 16)))
    X = ids(S); st, pl, eq, y = [torch.tensor([x[i] for x in info], device=DEVICE) for i in range(4)]
    return X, st, pl, eq, y.cpu().numpy()


if PART == "operand":
    out = {}
    for b in BASES:
        X, st, pl, eq, y = make_equal(b, 1200, b * 11 + 5, 2); B, n = X.shape; J = torch.arange(B, device=DEVICE); idx = torch.arange(n, device=DEVICE)[None]
        bm = (idx > pl[:, None]) & (idx < eq[:, None])

        def block(key_pos):
            def pat(t):
                keym = idx == key_pos[:, None]; rows = bm[:, None, :, None] & keym[:, None, None, :]
                t = t.masked_fill(rows, 0.0); return t / t.sum(-1, keepdim=True).clamp_min(1e-9)
            return {"pattern0": pat}

        r = {}
        for nm, h in (("none", None), ("block '+'", block(pl)), ("CONTROL block line start", block(st - 1)), ("CONTROL block a's first digit", block(st))):
            lg, c = model.forward(X, hooks=h, capture={"pattern1"}); row = c["pattern1"][J, 1, eq]
            onb = (row * bm).sum(1).cpu().numpy()
            r[nm] = {"L1H1 on b (mean)": float(onb.mean()), "L1H1 on b (bootstrap)": boot([onb.mean()] + [onb[np.random.default_rng(i).integers(0, B, B)].mean() for i in range(NB)]),
                     "correct": W(lg[J, eq].argmax(-1).cpu().numpy() == tok[y])}
        out[b] = r
        print(f"base {b}:", {k_: (round(v["L1H1 on b (mean)"], 3), fmt(v["correct"])) for k_, v in r.items()}, flush=True)
    res["operand"] = out

if PART == "recency":
    out = {}
    for b in (8, 10, 16):
        X, st, pl, eq, y = make_equal(b, 1200, b * 3 + 7, 2); B, n = X.shape; J = torch.arange(B, device=DEVICE); idx = torch.arange(n, device=DEVICE)[None]
        am = (idx >= st[:, None]) & (idx < pl[:, None]); src = torch.where(am, st[:, None] + pl[:, None] - 1 - idx, idx).clamp(0, n - 1)
        r = {}
        for nm, h in [("none", None)] + [(f"L0H{h} reversed over a's digits" + ("" if h == 3 else " (CONTROL)"), h) for h in range(4)]:
            hooks = None
            if h is not None:
                def rev(t, h=h):
                    t = t.clone(); t[:, :, h] = t[J[:, None], src, h]; return t
                hooks = {"z0": rev}
            lg, c = model.forward(X, hooks=hooks, capture={"pattern1"}); row = c["pattern1"][J, 1, eq]
            last, first = row[J, pl - 1].cpu().numpy(), row[J, st].cpu().numpy()
            r[nm] = {"L1H1 on a's last digit": boot([last.mean()] + [last[np.random.default_rng(i).integers(0, B, B)].mean() for i in range(NB)]),
                     "L1H1 on a's first digit": boot([first.mean()] + [first[np.random.default_rng(i).integers(0, B, B)].mean() for i in range(NB)]),
                     "correct": W(lg[J, eq].argmax(-1).cpu().numpy() == tok[y])}
        out[b] = r
        print(f"base {b}:", flush=True)
        for k_, v in r.items():
            last, first, ok = v["L1H1 on a's last digit"], v["L1H1 on a's first digit"], v["correct"]
            print(f"   {k_:42s} last {fmt(last)} | first {fmt(first)} | correct {fmt(ok)}", flush=True)
    res["recency"] = out

# ============================================================================================ carry_rule
if PART == "carry_rule":
    out = {}
    for base in (8, 10, 16):
        rng = random.Random(base * 3); S, rec = [], []
        while len(S) < 6000:
            a, c_ = rng.randint(base, 999), rng.randint(base, 999); ap, bp, w = rng.randrange(base), rng.randrange(base), rng.randrange(base)
            A = a + (ap - a % base); B = c_ + (bp - c_ % base); sa, sb = to_base(A, base), to_base(B, base)
            if len(sa) <= 1 or len(sb) <= 1: continue
            S.append("\n".join(render(*s, base) for s in sample_shots(rng, (A, B))) + f"\n{sa}+{sb}={DIGITS[w]}")
            rec.append((ap, bp, w, (A // base) % base + (B // base) % base))
        lg = []
        for i in range(0, len(S), 256):
            x = ids(S[i:i + 256]); q = torch.tensor([len(s) - 1 for s in S[i:i + 256]], device=DEVICE)
            l, _ = model.forward(x); lg.append(l[torch.arange(len(q), device=DEVICE), q].cpu().numpy())
        lg = np.concatenate(lg); ap, bp, w, cur = np.array(rec).T; J = np.arange(len(lg))
        c0, c1 = tok[cur % base], tok[(cur + 1) % base]; ev = lg[J, c1] - lg[J, c0]; pr = lg.argmax(1); one = pr == c1; val = one | (pr == c0)
        lin = ap + bp - w; trn = J % 2 == 0; te = (~trn) & val
        TH = np.arange(-base, 2 * base) + .5

        def stats(ix):
            X = np.c_[ap[ix], bp[ix], w[ix], np.ones(len(ix))].astype(float); co, *_ = np.linalg.lstsq(X, ev[ix], rcond=None)
            r2 = 1 - ((ev[ix] - X @ co) ** 2).sum() / ((ev[ix] - ev[ix].mean()) ** 2).sum()
            tr_ = ix[trn[ix] & val[ix]]; te_ = ix[te[ix]]
            th = TH[np.argmax([((lin[tr_] > t) == one[tr_]).mean() for t in TH])]
            return [co[0], co[1], co[2], r2, th, ((lin[te_] > th) == one[te_]).mean()]

        allix = np.arange(len(J)); g = np.random.default_rng(0)
        B_ = np.array([stats(allix)] + [stats(g.integers(0, len(J), len(J))) for _ in range(NB)])
        names = ["a weight", "b weight", "written weight", "R2", "threshold", "held-out rule accuracy"]
        r = {nm: boot(B_[:, i]) for i, nm in enumerate(names)}
        r["held-out rule accuracy (Wilson)"] = W((lin[te] > B_[0, 4]) == one[te])
        r["CONTROL operands only: a+b >= base"] = W((ap[te] + bp[te] >= base) == one[te])
        r["CONTROL written only (best threshold on train)"] = W((w[te] < TH[np.argmax([((w[trn & val] < t) == one[trn & val]).mean() for t in TH])]) == one[te])
        r["CONTROL always the majority answer"] = W(one[te] == (one[trn & val].mean() > .5))
        out[base] = r
        print(f"base {base}:", flush=True)
        for k_, v in r.items(): print(f"   {k_:48s} {fmt(v)}", flush=True)
    res["carry_rule"] = out

# ============================================================================================ carry_vec
if PART == "carry_vec":
    out = {}
    for b in BASES:
        d = load_stored(b, 1); half = d["idx"] % 2 == 0

        def vec(c):
            ds = []
            for i in range(b):
                for j in range(b):
                    m0, m1 = half & (d["a"] == i) & (d["b"] == j) & (c == 0), half & (d["a"] == i) & (d["b"] == j) & (c == 1)
                    if m0.sum() >= 2 and m1.sum() >= 2: ds.append(d["pre"][mt(m1)].mean(0) - d["pre"][mt(m0)].mean(0))
            return torch.stack(ds).mean(0)

        v = vec(d["c"]); te = (~half) & (d["c"] == 0); alt = tok[(d["s"][te] + 1) % b]
        flip = lambda u: L_out(d["mid"][mt(te)], d["pre"][mt(te)] + u).argmax(-1).cpu().numpy() == alt
        r = {"carry vector flips carry": W(flip(v))}
        g = torch.Generator(device=DEVICE).manual_seed(1); rf = []
        for _ in range(200):
            u = torch.randn(256, device=DEVICE, generator=g); rf.append(flip(u / u.norm() * v.norm()).mean())
        rf = np.array(rf); r["CONTROL 200 random vectors, same size"] = {"mean": float(rf.mean()), "p95": float(np.percentile(rf, 95)), "max": float(rf.max())}
        # Noise control: the same recipe applied to a random split of carry-0 prompts (pseudo carry labels), rescaled to the real
        # vector's size. It carries the data's own variation but no carry. (Shuffling real carry labels is not a fair null here,
        # because within a digit-pair cell most of the remaining variation is the carry itself.)
        rng = np.random.default_rng(0); sh = []
        for _ in range(20):
            c = np.full(len(d["c"]), -1); z = np.where(d["c"] == 0)[0]; c[z] = rng.integers(0, 2, len(z))
            u = vec(c); sh.append(flip(u / u.norm() * v.norm()).mean())
        sh = np.array(sh); r["CONTROL vector from a random split of no-carry prompts, same size (20 splits)"] = {"mean": float(sh.mean()), "p95": float(np.percentile(sh, 95)), "max": float(sh.max())}
        out[b] = r
        print(f"base {b}: flip {fmt(r['carry vector flips carry'])}; random mean {rf.mean():.3f} p95 {np.percentile(rf, 95):.3f} max {rf.max():.3f}; noise-split mean {sh.mean():.3f} max {sh.max():.3f}", flush=True)
    res["carry_vec"] = out

# ============================================================================================ base_vec
if PART == "base_vec":
    D = {b: load_stored(b, 0) for b in (8, 10, 16)}; out = {}

    def vec(dx, dy, lab_x, lab_y, lim):
        """Mean difference y - x over shared (a, b, c) cells, train half only; lab_* say which rows count as each base."""
        ds = []
        for i in range(lim):
            for j in range(lim):
                for c in (0, 1):
                    m1 = lab_x & (dx["a"] == i) & (dx["b"] == j) & (dx["c"] == c); m2 = lab_y & (dy["a"] == i) & (dy["b"] == j) & (dy["c"] == c)
                    if m1.sum() >= 2 and m2.sum() >= 2: ds.append(dy["pre"][mt(m2)].mean(0) - dx["pre"][mt(m1)].mean(0))
        return torch.stack(ds).mean(0)

    for x, y in [(8, 10), (10, 8), (10, 16), (16, 10), (8, 16), (16, 8)]:
        dx, dy = D[x], D[y]; lim = min(x, y); hx, hy = dx["idx"] % 2 == 0, dy["idx"] % 2 == 0
        v = vec(dx, dy, hx, hy, lim); te = ~hx; s = dx["s"]; inb = te & (dx["a"] < lim) & (dx["b"] < lim)
        conf = inb & (s >= lim) & (s % x != s % y); non = inb & (s < lim)
        pc = lambda u: L_out(dx["mid"][mt(conf)], dx["pre"][mt(conf)] + u).argmax(-1).cpu().numpy()
        r = {"switches to the other rule": W(pc(v) == tok[s[conf] % y]),
             "non-conflict answers kept": W(L_out(dx["mid"][mt(non)], dx["pre"][mt(non)] + v).argmax(-1).cpu().numpy() == tok[s[non] % x])}
        g = torch.Generator(device=DEVICE).manual_seed(0); rf = []
        for _ in range(200):
            u = torch.randn(256, device=DEVICE, generator=g); rf.append((pc(u / u.norm() * v.norm()) == tok[s[conf] % y]).mean())
        rf = np.array(rf); r["CONTROL 200 random vectors, same size"] = {"mean": float(rf.mean()), "p95": float(np.percentile(rf, 95)), "max": float(rf.max())}
        # Noise control: the same recipe applied to a random split of base-x prompts into two pseudo-bases, rescaled to the real
        # vector's size. It carries the data's own variation but no base difference.
        rng = np.random.default_rng(0); sh = []
        for _ in range(20):
            lx = hx & (rng.random(len(hx)) < .5)
            u = vec(dx, dx, lx, hx & ~lx, x); sh.append((pc(u / u.norm() * v.norm()) == tok[s[conf] % y]).mean())
        sh = np.array(sh); r["CONTROL vector from a random split of base-x prompts, same size (20 splits)"] = {"mean": float(sh.mean()), "p95": float(np.percentile(sh, 95)), "max": float(sh.max())}
        out[f"{x}->{y}"] = r
        print(f"{x}->{y}: switch {fmt(r['switches to the other rule'])}; kept {fmt(r['non-conflict answers kept'])}; random mean {rf.mean():.3f} max {rf.max():.3f}; noise-split mean {sh.mean():.3f} max {sh.max():.3f}", flush=True)
    res["base_vec"] = out

# ============================================================================================ formula
if PART == "formula":
    out = {}
    for k in (0, 1, 2):
        Dk = [load_stored(b, k) for b in (8, 10, 16)]
        d = {f: (np.concatenate([x[f] for x in Dk]) if f != "pre" and f != "mid" else torch.cat([x[f] for x in Dk])) for f in ("a", "b", "c", "s", "idx", "pre", "mid")}
        d["base"] = np.concatenate([np.full(len(x["a"]), b) for x, b in zip(Dk, (8, 10, 16))])
        pre = d["pre"].cpu().numpy(); n = len(d["a"]); tr = d["idx"] % 2 == 0; te = ~tr
        oh = lambda v: np.eye(16)[v]; bv = np.stack([d["base"] == 10, d["base"] == 16], 1).astype(np.float32)

        def acc(a_lab, b_lab, c_lab, base_lab):
            X = np.c_[np.ones(n), oh(a_lab), oh(b_lab), c_lab, base_lab].astype(np.float32)
            co = np.linalg.lstsq(X[tr], pre[tr], rcond=None)[0]
            Xt = np.c_[np.ones(n), oh(d["a"]), oh(d["b"]), d["c"], bv].astype(np.float32)[te]  # always predict from the true labels
            pred = torch.as_tensor(Xt @ co, device=DEVICE)
            m = torch.zeros_like(d["mid"][mt(te)])
            for b in (8, 10, 16):
                m[mt(d["base"][te] == b)] = d["mid"][mt(tr & (d["base"] == b))].mean(0)
            lg = L_out(m, pred).argmax(-1).cpu().numpy(); s = d["s"][te]
            hit = lg == tok[s % d["base"][te]]; bt = d["base"][te]
            r_ = {b: W(hit[bt == b]) for b in (8, 10, 16)}
            # The base matters only where the bases disagree (sum >= 8). Digits 8-F occur only in bases 10 and 16, so the digit tables
            # themselves reveal the base for pairs that use them; the fair subset is both digits below 8 and sum >= 8.
            amb = (d["a"][te] < 8) & (d["b"][te] < 8) & (s >= 8)
            r_.update({f"{b}, digits < 8, sum >= 8": W(hit[(bt == b) & amb]) for b in (8, 10, 16)})
            return r_

        rng = np.random.default_rng(k)
        r = {"formula": acc(d["a"], d["b"], d["c"], bv)}
        sh = lambda v: rng.permutation(v)
        r["CONTROL digit labels shuffled when fitting"] = acc(sh(d["a"]), sh(d["b"]), d["c"], bv)
        r["CONTROL base labels shuffled when fitting"] = acc(d["a"], d["b"], d["c"], bv[rng.permutation(n)])
        if k: r["CONTROL carry labels shuffled when fitting"] = acc(d["a"], d["b"], sh(d["c"]), bv)
        out[k] = r
        print(f"column {k}:", flush=True)
        for nm, v in r.items():
            print(f"   {nm:44s}", " | ".join(f"b{b} {fmt(v[b])}" for b in (8, 10, 16)), flush=True)
            print(f"   {'   only digits < 8 and sum >= 8':44s}", " | ".join(f"b{b} {fmt(v[f'{b}, digits < 8, sum >= 8'])}" for b in (8, 10, 16)), flush=True)
    res["formula"] = out

# ============================================================================================ clock
if PART == "clock":
    WEm = model.WE[torch.as_tensor(tok, device=DEVICE)].cpu().numpy(); out = {}

    def loo_r2(table, offs, mask):
        n = table.shape[0]; pred = np.zeros_like(table)
        for o in range(n):
            other = np.arange(n) != o
            for lab in np.unique(offs[o]):
                m = (offs[other] == lab) & mask[other]; pred[o, offs[o] == lab] = table[other][m].mean() if m.any() else 0.0
        return float(1 - ((table[mask] - pred[mask]) ** 2).sum() / ((table[mask] - table[mask].mean()) ** 2).sum())

    for b in (8, 10, 16):
        t = Table(b); rows = t.select(QUERY, 0); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
        keep = t.meta["correct"][rows] == 1; rows, A, B = rows[keep], A[keep], B[keep]
        lg = model.unembed(torch.as_tensor(t.get("resid1", rows), device=DEVICE)).cpu().numpy()[:, tok[:b]]
        o = (A % b + B % b) % b; Lc = lg - lg.mean(1, keepdims=True)
        tv = np.arange(b); diff = tv[None] - tv[:, None]; mask = diff != 0; circ = diff % b; dist = np.abs(diff); cd = np.minimum(circ, b - circ)
        near, far = (cd == 1) & (dist == 1), (cd == 1) & (dist == b - 1)

        def stats(ix):
            T = np.stack([Lc[ix][o[ix] == v].mean(0) for v in range(b)])
            rc, rd = loo_r2(T, circ, mask), loo_r2(T, dist, mask)
            return [rc, rd, rd - rc, T[near].mean(), T[far].mean(), T[near].mean() - T[far].mean()]

        g = np.random.default_rng(b); N = len(o)
        Bs = np.array([stats(np.arange(N))] + [stats(g.integers(0, N, N)) for _ in range(300)])
        names = ["R2 circular offset", "R2 token distance", "distance minus circular", "ordinary neighbours", "wrap-around neighbours", "ordinary minus wrap-around"]
        out[b] = {nm: boot(Bs[:, i]) for i, nm in enumerate(names)}; out[b]["n prompts"] = int(N)
        print(f"base {b} (n={N}):", {nm: fmt(v) for nm, v in out[b].items() if nm != "n prompts"}, flush=True)
    res["clock"] = out

# ============================================================================================ offset
if PART == "offset":
    D = {}
    for b in (8, 10, 16):
        t = Table(b); rows = t.select(QUERY, 0); rows = rows[t.meta["correct"][rows] == 1]
        R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
        mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
        D[b] = (t.meta["a"][rows] % b, t.meta["b"][rows] % b, (model._ln("h.1.ln_2", mid) @ Wc + bc).cpu().numpy())
    WoN = np.linalg.norm(Wo.cpu().numpy(), axis=1); out = {}

    def fit(x, y, ix, iy):
        lim = min(x, y); ax_, bx_, px = D[x]; ay_, by_, py = D[y]; ax_, bx_, px = ax_[ix], bx_[ix], px[ix]; ay_, by_, py = ay_[iy], by_[iy], py[iy]
        kx = ax_ * 16 + bx_; ky = ay_ * 16 + by_; keys = [k for k in range(lim * 16) if k % 16 < lim and (kx == k).any() and (ky == k).any()]
        Px = np.stack([px[kx == k].mean(0) for k in keys]); Py = np.stack([py[ky == k].mean(0) for k in keys])
        tot = ((Py - Py.mean(0)) ** 2).sum(0) + 1e-9
        r_off = 1 - (((Py - Px) - (Py - Px).mean(0)) ** 2).sum(0) / tot
        g_only = (Px * Py).sum(0) / ((Px ** 2).sum(0) + 1e-9); r_gain = 1 - ((Py - g_only * Px) ** 2).sum(0) / tot
        xc = Px - Px.mean(0); gain = (xc * (Py - Py.mean(0))).sum(0) / ((xc ** 2).sum(0) + 1e-9); off = Py.mean(0) - gain * Px.mean(0)
        r_both = 1 - ((Py - (gain * Px + off)) ** 2).sum(0) / tot
        post = gelu_new(torch.as_tensor(Px)).numpy(); imp = np.abs(post - post.mean(0)).mean(0) * WoN; w = imp / imp.sum()
        return [(w * r_off).sum(), (w * r_gain).sum(), (w * r_both).sum(), np.median(gain[w > np.median(w)])]

    for x, y in ((10, 16), (16, 10), (8, 10), (10, 8)):
        nx, ny = len(D[x][0]), len(D[y][0]); g = np.random.default_rng(x * y)
        Bs = np.array([fit(x, y, np.arange(nx), np.arange(ny))] + [fit(x, y, g.integers(0, nx, nx), g.integers(0, ny, ny)) for _ in range(300)])
        out[f"{x}->{y}"] = {nm: boot(Bs[:, i]) for i, nm in enumerate(["offset only R2", "gain only R2", "gain + offset R2", "typical gain"])}
        print(f"{x}->{y}:", {nm: fmt(v) for nm, v in out[f'{x}->{y}'].items()}, flush=True)
    res["offset"] = out

save_json("t22_key_stats", res)
print("DONE", PART)
