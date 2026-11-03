"""F15: carry as a bias, binary's place in the picture, binary's errors, and the stop signal.

A. Carry bias: constant pre-activation vector from carry 0 vs 1 on matched digit pairs. Add it: does output become s+1? Transfer across bases.
B. Base bias including binary: same test with (a,b) = (1,1) conflict.
C. Parity neurons in binary vs other bases.
D. Binary length-gap errors (forward passes): attention at the digit, and whether errors come at the end of the shorter operand.
E. Stop: at k = max(len a, len b), compare queries that stop and those with a final carry. Which part flips it?
"""
import itertools, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, render, sample_shots, save_json, to_answer, to_base
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); tt = torch.as_tensor(tok, device=DEVICE)
Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]; W_tok = (Wo @ model.WE[tt].T).cpu().numpy()
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; ak, bk = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    return dict(a=ak, b=bk, c=c, s=ak + bk + c, mid=mid, pre=pre, idx=t.meta["idx"][rows], z=Z, R0=R0)
D = {(b, k): load(b, k) for b in BASES for k in (0, 1, 2)}
logits = lambda mid, pre: model.unembed(mid + gelu_new(pre) @ Wo + bo)
def carry_vec(d, b, half):
    diffs = []
    for i in range(b):
        for j in range(b):
            m0 = half & (d["a"] == i) & (d["b"] == j) & (d["c"] == 0); m1 = half & (d["a"] == i) & (d["b"] == j) & (d["c"] == 1)
            if m0.sum() >= 2 and m1.sum() >= 2:
                diffs.append(d["pre"][torch.as_tensor(m1, device=DEVICE)].mean(0) - d["pre"][torch.as_tensor(m0, device=DEVICE)].mean(0))
    x = torch.stack(diffs); v = x.mean(0)
    return v, float(v.pow(2).sum() / x.pow(2).sum(1).mean())
vecs = {}
for b in BASES:
    for k in (1, 2):
        d = D[(b, k)]; half = d["idx"] % 2 == 0
        v, share = carry_vec(d, b, half); vecs[(b, k)] = v
        te = ~half; res["A"][f"{b}/k{k}"] = {"constant_share": share, "curve": {}}
        for direction, sign in (("add carry", 1), ("remove carry", -1)):
            cv = 0 if sign == 1 else 1
            m = te & (d["c"] == cv); mt = torch.as_tensor(m, device=DEVICE)
            cur, alt = tok[(d["a"][m] + d["b"][m] + cv) % b], tok[(d["a"][m] + d["b"][m] + 1 - cv) % b]
            for al in (.5, 1., 1.5):
                p = logits(d["mid"][mt], d["pre"][mt] + sign * al * v).argmax(-1).cpu().numpy()
                res["A"][f"{b}/k{k}"]["curve"][f"{direction}/{al}"] = {"stays": float((p == cur).mean()), "flips": float((p == alt).mean())}
        print(f"A b{b} k{k} constant share {share:.2f}:", {a: round(v2['flips'], 2) for a, v2 in res['A'][f'{b}/k{k}']['curve'].items()}, flush=True)
# transfer within k=1
tab = {}
for x, y in itertools.permutations(BASES, 2):
    d = D[(y, 1)]; m = (d["idx"] % 2 == 1) & (d["c"] == 0); mt = torch.as_tensor(m, device=DEVICE)
    alt = tok[(d["a"][m] + d["b"][m] + 1) % y]
    p = logits(d["mid"][mt], d["pre"][mt] + vecs[(x, 1)]).argmax(-1).cpu().numpy()
    tab[f"{x}->{y}"] = float((p == alt).mean())
res["A"]["transfer_k1"] = tab
cos = {f"{x}v{y}": float(torch.nn.functional.cosine_similarity(vecs[(x, 1)], vecs[(y, 1)], dim=0)) for x, y in itertools.combinations(BASES, 2)}
res["A"]["cos_k1"] = cos
print("A transfer add-carry using another base's vector:", {a: round(v, 2) for a, v in tab.items()}, "\n  cosine", {a: round(v, 2) for a, v in cos.items()}, flush=True)
# B base bias with binary
for k in (0, 1):
    for x, y in itertools.permutations(BASES, 2):
        if 2 not in (x, y): continue
        dx, dy = D[(x, k)], D[(y, k)]
        lim = 2
        trx, try_ = dx["idx"] % 2 == 0, dy["idx"] % 2 == 0; diffs = []
        for i, j, c in itertools.product(range(2), range(2), (0, 1)):
            mx = trx & (dx["a"] == i) & (dx["b"] == j) & (dx["c"] == c); my = try_ & (dy["a"] == i) & (dy["b"] == j) & (dy["c"] == c)
            if mx.sum() >= 2 and my.sum() >= 2: diffs.append(dy["pre"][torch.as_tensor(my, device=DEVICE)].mean(0) - dx["pre"][torch.as_tensor(mx, device=DEVICE)].mean(0))
        v = torch.stack(diffs).mean(0)
        te = (~trx) & (dx["a"] < 2) & (dx["b"] < 2) & (dx["s"] >= 2)
        if te.sum() < 20: continue
        tet = torch.as_tensor(te, device=DEVICE); s = dx["s"][te]; rx, ry = tok[s % x], tok[s % y]; dd = rx != ry
        p = logits(dx["mid"][tet], dx["pre"][tet] + v).argmax(-1).cpu().numpy()
        res["B"][f"k{k}/{x}->{y}"] = {"own": float((p[dd] == rx[dd]).mean()), "other": float((p[dd] == ry[dd]).mean()), "n": int(dd.sum()), "constant_share": float(v.pow(2).sum() / torch.stack(diffs).pow(2).sum(1).mean())}
        print("B", k, x, y, res["B"][f"k{k}/{x}->{y}"], flush=True)
# C parity neurons
top = {}
for b in BASES:
    d = D[(b, 0)]; Q = gelu_new(d["pre"]).cpu().numpy(); s = d["s"]; even = s % 2 == 0
    amp = Q[even].mean(0) - Q[~even].mean(0)
    w = W_tok[:, :b]; writes = np.array([w[:, ::2].mean(1) - w[:, 1::2].mean(1)])[0]
    top[b] = set(np.argsort(-(amp * writes))[:16].tolist())
res["C"]["overlap_top16"] = {f"{x}v{y}": len(top[x] & top[y]) for x, y in itertools.combinations(BASES, 2)}
print("C parity-neuron overlap (top 16):", res["C"]["overlap_top16"], flush=True)
save_json("f15_carry_binary_stop", res)
# D binary errors
rng = random.Random(202); recs = {}
P = []
while len(P) < 6000:
    a, b_ = rng.randint(0, 999), rng.randint(0, 999)
    sa, sb = to_base(a, 2), to_base(b_, 2)
    if abs(len(sa) - len(sb)) < 3 and rng.random() > .08: continue
    P.append(("\n".join(render(*s, 2) for s in sample_shots(rng, (a, b_))), sa, sb, to_answer(a + b_, 2)))
stats = {}
for i in range(0, len(P), 128):
    ch = P[i:i + 128]; S = [f"{c}\n{x}+{y}={z}" for c, x, y, z in ch]
    seq = [[model.vocab[c] for c in s] + [model.eos] for s in S]; L = max(map(len, seq)); ids = torch.full((len(seq), L), model.pad, dtype=torch.long)
    for j, s in enumerate(seq): ids[j, :len(s)] = torch.tensor(s)
    lg, cc = model.forward(ids.to(DEVICE), capture={"pattern1"}); pat = cc["pattern1"].cpu().numpy(); pr = lg.argmax(-1).cpu().numpy()
    for j, (c, x, y, z) in enumerate(ch):
        eq = len(c) + 1 + len(x) + 1 + len(y); pl = eq - len(y) - 1; gap = abs(len(x) - len(y)); short = min(len(x), len(y))
        for k in range(len(z)):
            q = eq + k; wrong = pr[j, q] != model.vocab[z[k]]
            rel = k - short
            for key in ((gap >= 6, rel), ):
                e = stats.setdefault(key, [0, 0, 0., 0.]); e[0] += 1; e[1] += wrong
                # attention on the digit of the LONGER operand at k
                if len(x) >= len(y):
                    if k < len(x): e[2] += pat[j, 1, q, pl - 1 - k]
                else:
                    if k < len(y): e[3] += pat[j, 0, q, eq - 1 - k]
res["D"] = {f"gap>=6={g}/k-short={r}": {"n": v[0], "digit_error": v[1] / v[0]} for (g, r), v in sorted(stats.items())}
print("D binary digit error by position relative to end of shorter operand:", flush=True)
for g in (False, True):
    print("  gap>=6" if g else "  gap<6", {r: (round(v[1] / v[0], 3), v[0]) for (g2, r), v in sorted(stats.items()) if g2 == g and -4 <= r <= 5}, flush=True)
save_json("f15_carry_binary_stop", res)
# E stop: replay swap between stop and continue queries with k = max len
for b in BASES:
    t = Table(b); m = t.meta
    q = np.where((m["kind"] == QUERY) & (m["correct"] == 1))[0]
    A, B, k = m["a"][q].astype(np.int64), m["b"][q].astype(np.int64), m["j"][q].astype(np.int64)
    la = np.array([len(np.base_repr(int(x), b)) for x in A]); lb = np.array([len(np.base_repr(int(x), b)) for x in B])
    at = k == np.maximum(la, lb)
    stop = at & ((A + B) < b**k); cont = at & ((A + B) >= b**k)
    q_s, q_c = q[stop], q[cont]
    k_s, k_c = k[stop], k[cont]
    Rz = lambda rows: (torch.as_tensor(t.get("resid0", rows), device=DEVICE), torch.as_tensor(t.get("z1", rows), device=DEVICE))
    out = {"n_stop": int(stop.sum()), "n_final_carry": int(cont.sum())}
    if stop.sum() < 50 or cont.sum() < 50: res["E"][b] = out; continue
    rng2 = np.random.default_rng(b)
    pairs = []
    for kv in np.unique(k_c):
        s_idx, c_idx = np.where(k_s == kv)[0], np.where(k_c == kv)[0]
        if len(s_idx) == 0: continue
        n = min(500, max(len(s_idx), len(c_idx)))
        pairs.append((rng2.choice(s_idx, n), rng2.choice(c_idx, n)))
    si = np.concatenate([p[0] for p in pairs]); ci = np.concatenate([p[1] for p in pairs])
    Rs, Zs = Rz(q_s[si]); Rc, Zc = Rz(q_c[ci])
    def go(pre, z):
        mid = pre + z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
        p = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
        return model.unembed(mid + gelu_new(p) @ Wo + bo).argmax(-1).cpu().numpy()
    out["clean_stop_rate"] = float((go(Rs, Zs) == model.eos).mean())
    comps = {}
    for nm in ("R", "H0", "H1", "H2", "H3", "all heads"):
        z = Zs.clone(); r = Rs
        if nm == "R": r = Rc
        elif nm == "all heads": z = Zc
        else:
            h = int(nm[1]); z[:, 16 * h:16 * h + 16] = Zc[:, 16 * h:16 * h + 16]
        comps[nm] = float((go(r, z) == model.eos).mean())
    out["still_stops_after_swap_from_final_carry"] = comps
    res["E"][b] = out
    print("E", b, out, flush=True)
save_json("f15_carry_binary_stop", res); print("DONE")
