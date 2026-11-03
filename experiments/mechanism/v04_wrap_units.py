"""V04 (stored + weights): are there wrap units, and do they implement the modulus?

Wrap unit: fires when s >= base, not below. Selected on half the prompts, tested on the other half.
A. Double dissociation: mean-ablate top wrap units or top no-wrap units. Do wrapped and unwrapped sums fail separately?
B. Collective step: the group's total firing against s on held-out prompts. Is there a step at s = base?
C. Shape: per unit, is firing better described by a diagonal (s >= t) or a rectangle (a >= t and b >= t)?
D. Base gating: apply another base's vector to inputs. Does the step move to that base? Which units turn on or off?
E. What they write: wrap units should favour wrapped tokens (s - base), no-wrap units unwrapped ones.
F. Same analysis at digit 1 (uses carry-inclusive s). Overlap with digit 0's units.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}, "E": {}, "F": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); tt = torch.as_tensor(tok, device=DEVICE)
Wo = sd["h.1.mlp.c_proj.weight"]; bo = sd["h.1.mlp.c_proj.bias"]; W_tok = (Wo @ model.WE[tt].T).cpu().numpy()
def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; a, c_ = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    return dict(a=a, b=c_, c=c, s=a + c_ + c, mid=mid, pre=pre, post=gelu_new(pre), idx=t.meta["idx"][rows])
def logits(mid, post):
    return model.unembed(mid + post @ Wo + bo).argmax(-1).cpu().numpy()
units = {}
for b in BASES:
    for k in (0, 1):
        d = load(b, k); tr = d["idx"] % 2 == 0; te = ~tr; s = d["s"]; X = d["post"].cpu().numpy()
        wrap = s >= b
        sel = X[tr & wrap].mean(0) - X[tr & ~wrap].mean(0)
        sd_ = X[tr].std(0) + 1e-6
        order = np.argsort(-sel); wu, nu = order, order[::-1]
        if k == 0: units[b] = (order[:32], order[::-1][:32], sel)
        tet = torch.as_tensor(te, device=DEVICE); post = d["post"][tet]; mid = d["mid"][tet]; m = d["post"][torch.as_tensor(tr, device=DEVICE)].mean(0)
        st, oh = s[te], tok[s[te] % b]
        def acc(ids):
            p = post.clone()
            if len(ids): p[:, torch.as_tensor(ids.copy(), device=DEVICE)] = m[torch.as_tensor(ids.copy(), device=DEVICE)]
            pr = logits(mid, p)
            return float((pr[st >= b] == oh[st >= b]).mean()), float((pr[st < b] == oh[st < b]).mean())
        row = {"baseline": acc(np.array([], dtype=int))}
        for N in (8, 16, 32):
            row[f"wrap top {N}"] = acc(wu[:N]); row[f"no-wrap top {N}"] = acc(nu[:N])
            rs = np.random.default_rng(N); row[f"random {N}"] = tuple(np.mean([acc(rs.choice(256, N, replace=False)) for _ in range(3)], 0).tolist())
        res["A"][f"{b}/k{k}"] = row
        print(f"A b{b} k{k} (accuracy on wrapped, unwrapped):", {a: (round(v[0], 2), round(v[1], 2)) for a, v in row.items()}, flush=True)
        # B collective step, held out
        if k == 0:
            ids = wu[:16]; curve = np.array([X[te & (s == v)][:, ids].sum(1).mean() if (te & (s == v)).any() else np.nan for v in range(2 * b - 1)])
            cn = np.array([X[te & (s == v)][:, nu[:16]].sum(1).mean() if (te & (s == v)).any() else np.nan for v in range(2 * b - 1)])
            steps = np.diff(curve); best = int(np.nanargmax(steps) + 1)
            res["B"][b] = {"wrap_group_by_s": curve.tolist(), "no_wrap_group_by_s": cn.tolist(), "largest_rise_at_s": best}
            print(f"B b{b} wrap group total by s {np.round(curve, 1).tolist()} largest rise at s={best}", flush=True)
            print(f"     no-wrap group by s {np.round(cn, 1).tolist()}", flush=True)
            if b > 2:
                # C shape
                aa, bb = d["a"][te], d["b"][te]; XX = X[te]
                diag, rect, ths = [], [], []
                for n in wu[:16]:
                    T = np.array([[XX[(aa == i) & (bb == j), n].mean() if ((aa == i) & (bb == j)).any() else XX[:, n].mean() for j in range(b)] for i in range(b)])
                    tot = ((T - T.mean()) ** 2).sum(); S = np.add.outer(np.arange(b), np.arange(b)); bd, br, bt = -9, -9, 0
                    for t_ in range(1, 2 * b - 1):
                        m1 = S >= t_; p = np.where(m1, T[m1].mean(), T[~m1].mean()); r = 1 - ((T - p) ** 2).sum() / tot
                        if r > bd: bd, bt = r, t_
                    for t_ in range(1, b):
                        m1 = np.outer(np.arange(b) >= t_, np.arange(b) >= t_); p = np.where(m1, T[m1].mean(), T[~m1].mean()); br = max(br, 1 - ((T - p) ** 2).sum() / tot)
                    diag.append(bd); rect.append(br); ths.append(bt)
                res["C"][b] = {"diagonal_r2_mean": float(np.mean(diag)), "rectangle_r2_mean": float(np.mean(rect)), "best_diagonal_thresholds": ths}
                print(f"C b{b} top wrap units: diagonal fit {np.mean(diag):.2f}, rectangle fit {np.mean(rect):.2f}; best diagonal thresholds {ths}", flush=True)
                # E writes
                wr = W_tok[:, :b] - W_tok[:, :b].mean(1, keepdims=True); low = np.arange(b) < b // 2
                W_low = wr[:, low].mean(1) - wr[:, ~low].mean(1)
                res["E"][b] = {"wrap_units_write_low_minus_high": float(W_low[wu[:16]].mean()), "no_wrap_units": float(W_low[nu[:16]].mean()), "all_units": float(W_low.mean()), "frac_wrap_units_favour_low": float((W_low[wu[:16]] > 0).mean())}
                print("E", b, {a: round(v, 3) for a, v in res["E"][b].items()}, flush=True)
        if k == 1 and b in units:
            res["F"][b] = {"overlap_top16_wrap_digit0_digit1": len(set(units[b][0][:16].tolist()) & set(wu[:16].tolist()))}
# D base gating moves the step
for x, y in [(8, 10), (10, 8), (10, 16), (16, 10), (8, 16)]:
    dx, dy = load(x, 0), load(y, 0); lim = min(x, y); diffs = []
    for i in range(lim):
        for j in range(lim):
            m1 = (dx["a"] == i) & (dx["b"] == j); m2 = (dy["a"] == i) & (dy["b"] == j)
            if m1.sum() and m2.sum(): diffs.append(dy["pre"][torch.as_tensor(m2, device=DEVICE)].mean(0) - dx["pre"][torch.as_tensor(m1, device=DEVICE)].mean(0))
    v = torch.stack(diffs).mean(0)
    wu_x, wu_y = units[x][0][:16], units[y][0][:16]
    Xn = gelu_new(dx["pre"]).cpu().numpy(); Xs = gelu_new(dx["pre"] + v).cpu().numpy(); s = dx["s"]
    cur = lambda X, ids: np.array([X[s == q][:, ids].sum(1).mean() if (s == q).any() else np.nan for q in range(lim * 2 - 1)])
    r = {"overlap_units": len(set(wu_x.tolist()) & set(wu_y.tolist())), "own_units_before": cur(Xn, wu_x).tolist(), "own_units_after": cur(Xs, wu_x).tolist(), "other_units_before": cur(Xn, wu_y).tolist(), "other_units_after": cur(Xs, wu_y).tolist()}
    res["D"][f"{x}->{y}"] = r
    print(f"D rows in base {x}, add vector to {y}; shared units {r['overlap_units']}/16\n   {x}'s units before {np.round(r['own_units_before'], 1).tolist()}\n   {x}'s units after  {np.round(r['own_units_after'], 1).tolist()}\n   {y}'s units before {np.round(r['other_units_before'], 1).tolist()}\n   {y}'s units after  {np.round(r['other_units_after'], 1).tolist()}", flush=True)
print("F overlap", res["F"], flush=True)
save_json("v04_wrap_units", res); print("DONE")
