"""S08 (CPU): last-layer replay. Recomputes layer 1's post-attention step from stored values; no generation.

logits at a position = f(resid_post0, z1). Validated against stored resid1/mlp1 and the team's drops.
1. Where does the base enter? Same digits in two bases, swap subsets of {residual, H0..H3}; which wrap rule wins?
2. Synthetic addition: swap subsets within a base; whose a, b, carry does the output follow?
3. Ablate every L1 neuron at query positions, with error signatures.
4. Minimal circuit: keep top-N neurons, mean-replace the others.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from core import BASES, DIGITS, Model, save_json
from probe_lib import QUERY, Table
from core import gelu_new

torch.set_grad_enabled(False)
model = Model()
sd = model.sd
tok = torch.tensor([model.vocab[c] for c in DIGITS])
res = {}


def forward_l1(pre, z):
    attn = z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    mid = pre + attn
    post = gelu_new(model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"])
    return mid, post


def to_resid(mid, post):
    return mid + post @ sd["h.1.mlp.c_proj.weight"] + sd["h.1.mlp.c_proj.bias"]


def logits(resid):
    return model.unembed(resid)


def load(base, k):
    t = Table(base)
    rows = t.select(QUERY, k)
    a, b = t.meta["a"][rows], t.meta["b"][rows]
    keep = ((a + b) >= base**k) & (t.meta["correct"][rows] == 1)
    rows, a, b = rows[keep], a[keep], b[keep]
    ak, bk = (a // base**k) % base, (b // base**k) % base
    m = base**k
    c = (((a % m) + (b % m)) >= m).astype(int) if k else np.zeros(len(a), int)
    return {"pre": torch.as_tensor(t.get("resid0", rows)), "z": torch.as_tensor(t.get("z1", rows)), "a": ak, "b": bk, "c": c, "post": t.get("mlp1", rows), "resid": t.get("resid1", rows)}


D = {(b, k): load(b, k) for b in BASES for k in (0, 1)}

# ---- validation
res["validation"] = {}
for (b, k), d in D.items():
    mid, post = forward_l1(d["pre"], d["z"])
    r = to_resid(mid, post)
    e1, e2 = float((r - torch.as_tensor(d["resid"])).abs().max()), float((post - torch.as_tensor(d["post"])).abs().max())
    pred = logits(r).argmax(-1).numpy()
    truth = tok[torch.as_tensor((d["a"] + d["b"] + d["c"]) % b)].numpy()
    res["validation"][f"{b}/k{k}"] = {"max_resid_err": e1, "max_mlp_err": e2, "digit_acc": float((pred == truth).mean()), "n": len(pred)}
    print("VAL", b, k, res["validation"][f"{b}/k{k}"], flush=True)
    assert e1 < 1e-2 and e2 < 1e-2, "replay does not reproduce stored activations"

COMPS = ["R", "H0", "H1", "H2", "H3"]
SUBSETS = [s for n in range(1, 6) for s in itertools.combinations(COMPS, n)]


def swap(rec_pre, rec_z, don_pre, don_z, subset):
    pre = don_pre if "R" in subset else rec_pre
    z = rec_z.clone()
    for h in range(4):
        if f"H{h}" in subset:
            z[:, 16 * h : 16 * h + 16] = don_z[:, 16 * h : 16 * h + 16]
    return logits(to_resid(*forward_l1(pre, z))).argmax(-1).numpy()


# ---- 1. where the base enters
rng = np.random.default_rng(0)
res["base_entry"] = {}
for k in (0, 1):
    for A, B in itertools.permutations(BASES, 2):
        rec, don = D[(A, k)], D[(B, k)]
        lim = min(A, B)
        ri, di = [], []
        keyd = {}
        for i, key in enumerate(zip(don["a"], don["b"], don["c"])):
            keyd.setdefault(key, []).append(i)
        for i, key in enumerate(zip(rec["a"], rec["b"], rec["c"])):
            if sum(key) >= lim and key in keyd:  # digits valid in both, and the two rules disagree
                ri.append(i)
                di.append(keyd[key][rng.integers(len(keyd[key]))])
        if len(ri) < 30:
            continue
        ri, di = np.array(ri), np.array(di)
        s = rec["a"][ri] + rec["b"][ri] + rec["c"][ri]
        ra, rb = tok[torch.as_tensor(s % A)].numpy(), tok[torch.as_tensor(s % B)].numpy()
        out = {}
        for sub in SUBSETS:
            p = swap(rec["pre"][ri], rec["z"][ri], don["pre"][di], don["z"][di], sub)
            out["+".join(sub)] = {"receiver_rule": float((p == ra).mean()), "donor_rule": float((p == rb).mean())}
        res["base_entry"][f"k{k}/{A}<-{B}"] = {"n": len(ri), "subsets": out}
        show = {x: round(out[x]["donor_rule"], 2) for x in ["R", "H0", "H1", "H2", "H3", "H0+H1+H2+H3", "R+H0+H1+H2+H3"]}
        print(f"BASE k{k} receiver {A} donor {B} n={len(ri)} donor-rule rate:", show, flush=True)
    save_json("s08_replay", res)

# ---- 2. synthetic addition, within base
res["synthetic"] = {}
HYP = {"a": (1, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1), "a+b": (1, 1, 0), "a+c": (1, 0, 1), "b+c": (0, 1, 1), "a+b+c": (1, 1, 1)}
for (base, k), d in D.items():
    n = len(d["a"])
    di = rng.permutation(n)
    tabs = {}
    recv = (d["a"] + d["b"] + d["c"]) % base
    for sub in SUBSETS:
        p = swap(d["pre"], d["z"], d["pre"][di], d["z"][di], sub)
        row = {"unchanged": float((p == tok[torch.as_tensor(recv)].numpy()).mean())}
        for name, (x, y, w) in HYP.items():
            if k == 0 and w:
                continue
            A = d["a"][di] if x else d["a"]
            B = d["b"][di] if y else d["b"]
            C = d["c"][di] if w else d["c"]
            tgt = (A + B + C) % base
            diff = tgt != recv
            row[name] = float((p[diff] == tok[torch.as_tensor(tgt[diff])].numpy()).mean()) if diff.sum() >= 30 else None
        tabs["+".join(sub)] = row
    res["synthetic"][f"{base}/k{k}"] = tabs
    print(f"SYN b{base} k{k}", {x: {h: round(v, 2) for h, v in tabs[x].items() if v is not None and (v > 0.5 or h == "unchanged")} for x in ["R", "H0", "H1", "H2", "H3", "H0+H1", "H0+H3", "H1+H3", "H0+H1+H3", "H0+H1+H2+H3"]}, flush=True)
save_json("s08_replay", res)

# ---- 3 + 4. all query positions
Wo = sd["h.1.mlp.c_proj.weight"]
res["ablation"] = {}
res["minimal"] = {}
ranks = {}
rows_all = {}
for base in BASES:
    t = Table(base)
    m = t.meta
    q = np.where((m["kind"] == QUERY) & (m["correct"] == 1))[0]
    a, b, kk = m["a"][q], m["b"][q], m["j"][q]
    idx = m["idx"][q]
    post = torch.as_tensor(t.get("mlp1", q))
    resid = torch.as_tensor(t.get("resid1", q))
    ak, bk = (a // base**kk.astype(np.int64)) % base, (b // base**kk.astype(np.int64)) % base
    M = base ** kk.astype(np.int64)
    c = (((a % M) + (b % M)) >= M).astype(int)
    s = ak + bk + c
    sumdig = ((a + b) // M) % base
    is_eos = (a + b) < M
    is_eos[(a + b == 0) & (kk == 0)] = False
    truth = tok[torch.as_tensor(sumdig)].numpy().copy()
    truth[is_eos] = model.eos
    base_ok = logits(resid).argmax(-1).numpy() == truth
    print("base", base, "rows", len(q), "baseline digit acc", base_ok.mean(), flush=True)
    uniq = np.unique(idx)
    def exact(okv, idx=idx, uniq=uniq):
        bad = np.bincount(np.searchsorted(uniq, idx), weights=(~okv).astype(float), minlength=len(uniq))
        return float((bad == 0).mean())
    drop = np.zeros(256)
    sig = np.zeros((256, 6))
    dig = np.zeros((256, 4))
    for n in range(256):
        p = logits(resid - post[:, n:n + 1] * Wo[n]).argmax(-1).numpy()
        ok = p == truth
        drop[n] = exact(base_ok) - exact(ok)
        w = ~ok & base_ok
        pw = p[w]
        dc = (pw == tok[torch.as_tensor((ak[w] + bk[w]) % base)].numpy()) & (c[w] == 1)
        sc = (pw == tok[torch.as_tensor((ak[w] + bk[w] + 1) % base)].numpy()) & (c[w] == 0)
        ob = np.zeros(len(pw), bool)
        for T in BASES:
            if T != base:
                ob |= pw == tok[torch.as_tensor(s[w] % T)].numpy()
        ob &= ~is_eos[w]
        cp = (pw == tok[torch.as_tensor(ak[w])].numpy()) | (pw == tok[torch.as_tensor(bk[w])].numpy())
        e = pw == model.eos
        claimed = np.zeros(len(pw), bool)
        for i, mask in enumerate((dc, sc, ob, cp, e)):
            sig[n, i] = (mask & ~claimed).sum()
            claimed |= mask
        sig[n, 5] = (~claimed).sum()
        for i, kv in enumerate((0, 1, 2)):
            mk = (kk == kv) & ~is_eos
            dig[n, i] = (ok[mk].mean() - base_ok[mk].mean()) if mk.any() else 0
        dig[n, 3] = ok[is_eos].mean() - base_ok[is_eos].mean()
    order = np.argsort(-drop)
    ranks[base] = order
    rows_all[base] = (post, resid, truth, base_ok, exact)
    res["ablation"][base] = {"drop": drop, "signature_total": sig.sum(0), "digit_change": dig, "n_gt_01": int((drop > 0.01).sum()), "n_gt_05": int((drop > 0.05).sum()), "top": [(int(i), round(float(drop[i]), 3)) for i in order[:8]], "signature_top20": sig[order[:20]].sum(0)}
    print("ABL", base, "n>0.05", res["ablation"][base]["n_gt_05"], "top", res["ablation"][base]["top"], "sig", (sig[order[:20]].sum(0) / max(sig[order[:20]].sum(), 1)).round(2), flush=True)
    save_json("s08_replay", res)

# validation against team, and cross-base
import io, pickle, re, zipfile
class _S:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == "collections" and n == "OrderedDict":
            import collections
            return collections.OrderedDict
        return _S
    def persistent_load(self, pid): return None
for f in (Path(__file__).resolve().parents[2] / "ablations").iterdir():
    z = zipfile.ZipFile(f)
    d = _U(io.BytesIO(z.read([n for n in z.namelist() if n.endswith("data.pkl")][0]))).load()
    base = int(re.search(r"b=(\d+)", f.name).group(1))
    pairs = [(x["local_idx"], x["accuracy_drop"]) for x in d["ablations"] if x["layer_idx"] == 1 and x["type"] == "mlp"]
    ids = [p[0] for p in pairs]
    r = float(np.corrcoef([p[1] for p in pairs], res["ablation"][base]["drop"][ids])[0, 1])
    miss = np.setdiff1d(np.arange(256), ids)
    res["ablation"][base]["corr_with_team"] = r
    res["ablation"][base]["not_shortlisted_max_drop"] = float(res["ablation"][base]["drop"][miss].max())
    res["ablation"][base]["not_shortlisted_n_gt_01"] = int((res["ablation"][base]["drop"][miss] > 0.01).sum())
    print("TEAMVAL", base, "corr", round(r, 3), "not shortlisted: max drop", res["ablation"][base]["not_shortlisted_max_drop"], "n>0.01", res["ablation"][base]["not_shortlisted_n_gt_01"], flush=True)

SIZES = [4, 8, 16, 32, 64, 128, 256]
for base in BASES:
    post, resid, truth, base_ok, exact = rows_all[base]
    mean = post.mean(0)
    full = resid - post @ Wo
    curves = {}
    for src in BASES:
        row = {}
        for n in SIZES:
            keep = torch.as_tensor(ranks[src][:n].copy())
            pp = mean.expand_as(post).clone()
            pp[:, keep] = post[:, keep]
            row[n] = exact(logits(full + pp @ Wo).argmax(-1).numpy() == truth)
        curves[src] = row
    res["minimal"][base] = curves
    print("MIN base", base, {s: [round(v, 2) for v in r.values()] for s, r in curves.items()}, flush=True)
save_json("s08_replay", res)
print("DONE")
