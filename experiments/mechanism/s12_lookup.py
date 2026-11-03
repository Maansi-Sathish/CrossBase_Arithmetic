"""S12 (stored + weights, CPU, no generation): is there a lookup table inside the model?

A. Substitution: replace layer 1 neurons at digit queries with table entries learned on other prompts.
   Keys: (k,a,b,c) full table; (k,s) sum table; (k,o) output table; additive f(k,a)+g(k,b)+h(k,c).
B. Table shape: variance into additive / function of sum / pair-specific; effective rank; neuron selectivity.
C. Damage: when a neuron is zeroed, do individual pairs break or whole sums? (null: shuffled sum labels)
D. Reuse: is the table the same at each digit position, and is carry just "one more"?
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import torch
from core import BASES, DIGITS, Model, save_json
from probe_lib import QUERY, Table

torch.set_grad_enabled(False)
model = Model(); Wo = model.sd["h.1.mlp.c_proj.weight"]; tok = np.array([model.vocab[c] for c in DIGITS])
res = {"A": {}, "B": {}, "C": {}, "D": {}}


def load(b):
    t = Table(b); m = t.meta
    q = np.where((m["kind"] == QUERY) & (m["correct"] == 1))[0]
    A, B, k, idx = m["a"][q].astype(np.int64), m["b"][q].astype(np.int64), m["j"][q].astype(np.int64), m["idx"][q]
    M = b**k
    keep = (A + B) >= M
    q, A, B, k, idx, M = q[keep], A[keep], B[keep], k[keep], idx[keep], M[keep]
    ak, bk = (A // M) % b, (B // M) % b
    c = (((A % M) + (B % M)) >= M).astype(int)
    return dict(post=t.get("mlp1", q), resid=t.get("resid1", q), k=k, a=ak, b=bk, c=c, s=ak + bk + c, o=(ak + bk + c) % b, idx=idx)


def group_means(key, X, tr, default):
    uk, inv = np.unique(key, return_inverse=True)
    sums = np.zeros((len(uk), X.shape[1])); cnt = np.zeros(len(uk))
    np.add.at(sums, inv[tr], X[tr]); np.add.at(cnt, inv[tr], 1)
    out = sums / np.maximum(cnt, 1)[:, None]
    out[cnt == 0] = default
    return out[inv], int((cnt > 0).sum()), inv, cnt


for b in BASES:
    d = load(b)
    X, R = d["post"], d["resid"]
    te = d["idx"] % 5 == 0; tr = ~te
    truth = tok[d["o"]]
    uniq = np.unique(d["idx"][te]); pos = np.searchsorted(uniq, d["idx"][te])
    glob = X[tr].mean(0)

    def score(Xhat):
        r = torch.as_tensor(R[te]) + (torch.as_tensor(Xhat[te], dtype=torch.float32) - torch.as_tensor(X[te])) @ Wo
        p = model.unembed(r).argmax(-1).numpy()
        ok = p == truth[te]
        ex = float((np.bincount(pos, weights=(~ok).astype(float), minlength=len(uniq)) == 0).mean())
        return {"digit": float(ok.mean()), "digit_by_k": {int(x): float(ok[d["k"][te] == x].mean()) for x in (0, 1, 2) if (d["k"][te] == x).any()}, "whole_answer_digits": ex}

    table = {}
    key_full = d["k"] * 100000 + d["a"] * 1000 + d["b"] * 10 + d["c"]
    key_s = d["k"] * 1000 + d["s"]
    key_o = d["k"] * 1000 + d["o"]
    key_sc = d["k"] * 10000 + d["s"] * 10 + d["c"]
    subs = {}
    subs["unchanged"] = (X, 0)
    h, n, *_ = group_means(key_full, X, tr, glob); subs["full table (k,a,b,c)"] = (h, n)
    h, n, *_ = group_means(key_s, X, tr, glob); subs["sum table (k,s)"] = (h, n)
    h, n, *_ = group_means(key_o, X, tr, glob); subs["output table (k,o)"] = (h, n)
    ha, n1, *_ = group_means(d["k"] * 100 + d["a"], X, tr, glob)
    hb, n2, *_ = group_means(d["k"] * 100 + d["b"], X, tr, glob)
    hc, n3, *_ = group_means(d["k"] * 100 + d["c"], X, tr, glob)
    hk, n4, *_ = group_means(d["k"], X, tr, glob)
    subs["additive f(a)+g(b)+h(c)"] = (ha + hb + hc - 2 * hk, n1 + n2 + n3)
    subs["no information (k only)"] = (hk, n4)
    res["A"][b] = {nm: {"entries": n, **score(h)} for nm, (h, n) in subs.items()}
    print("A", b, {nm: (r["entries"], round(r["digit"], 3), round(r["whole_answer_digits"], 3), {k2: round(v, 3) for k2, v in r["digit_by_k"].items()}) for nm, r in res["A"][b].items()}, flush=True)

    # how well does each key explain neuron activity on held-out rows?
    tv = ((X[te] - X[tr].mean(0)) ** 2).sum()
    res["A"][b]["heldout_r2"] = {nm: float(1 - ((h[te] - X[te]) ** 2).sum() / tv) for nm, (h, n) in subs.items() if nm != "unchanged"}
    print("  heldout neuron R2", {a: round(v, 3) for a, v in res["A"][b]["heldout_r2"].items()}, flush=True)

    # ---- B + C at digit 0
    m0 = d["k"] == 0
    a0, b0 = d["a"][m0], d["b"][m0]
    X0 = X[m0]
    tab = np.zeros((b, b, 256))
    for i in range(b):
        for j in range(b):
            w = (a0 == i) & (b0 == j)
            tab[i, j] = X0[w].mean(0) if w.any() else np.nan
    if np.isnan(tab).any():
        tab = np.where(np.isnan(tab), np.nanmean(tab, (0, 1), keepdims=True), tab)
    t = tab - tab.mean((0, 1), keepdims=True)
    add = t.mean(1, keepdims=True) + t.mean(0, keepdims=True)
    I = t - add
    S = np.add.outer(np.arange(b), np.arange(b))
    Is = np.zeros_like(I)
    for v in np.unique(S):
        Is[S == v] = I[S == v].mean(0)
    rem = I - Is
    tot = (t**2).sum()
    sv = np.linalg.svd(t.reshape(b * b, -1), compute_uv=False)
    cum = np.cumsum(sv**2) / (sv**2).sum()
    active = (t**2).sum((0, 1)) > 0.01 * (t**2).sum((0, 1)).max()
    # shares per neuron
    pn = (t**2).sum((0, 1)) + 1e-9
    frac_rem = (rem**2).sum((0, 1)) / pn; frac_add = (add**2).sum((0, 1)) / pn; frac_s = (Is**2).sum((0, 1)) / pn
    part = ((t**2).sum((0, 1)) ** 2 / ((t**4).sum((0, 1)) + 1e-12)) / (b * b)
    res["B"][b] = {"share_additive": float((add**2).sum() / tot), "share_sum_function": float((Is**2).sum() / tot), "share_pair_specific": float((rem**2).sum() / tot), "cells": b * b, "rank_95": int(np.searchsorted(cum, 0.95)) + 1, "rank_99": int(np.searchsorted(cum, 0.99)) + 1, "n_active": int(active.sum()), "neurons_mostly_additive": int((frac_add[active] > .6).sum()), "neurons_mostly_sum": int((frac_s[active] > .6).sum()), "neurons_mostly_pair": int((frac_rem[active] > .6).sum()), "mean_cells_active_fraction": float(part[active].mean()), "neurons_lt_10pct_cells": int((part[active] < .1).sum())}
    print("B", b, res["B"][b], flush=True)

    post0 = torch.as_tensor(X0); r0 = torch.as_tensor(R[m0]); truth0 = truth[m0]
    uc = a0 * b + b0
    n_cell = np.bincount(uc, minlength=b * b)
    rng = np.random.default_rng(0)
    hits = np.zeros(b * b); rows = []
    for n in range(256):
        p = model.unembed(r0 - post0[:, n:n + 1] * Wo[n]).argmax(-1).numpy()
        wrong = np.bincount(uc, weights=(p != truth0).astype(float), minlength=b * b)
        rate = wrong / np.maximum(n_cell, 1)
        if wrong.sum() < 20:
            continue
        hits += rate > 0.5
        rows.append((n, rate))
    def s_share(rate, s):
        ok = np.ones(len(rate), bool)
        pred = np.zeros(len(rate))
        for c in range(len(rate)):
            o = (s == s[c]) & (np.arange(len(rate)) != c)
            pred[c] = rate[o].mean() if o.any() else rate.mean()
        return 1 - ((rate - pred) ** 2).sum() / ((rate - rate.mean()) ** 2).sum()
    Sr = S.ravel()
    obs, nulls, broken = [], [], []
    for n, rate in rows:
        obs.append(s_share(rate, Sr))
        nulls.append(np.mean([s_share(rate, rng.permutation(Sr)) for _ in range(20)]))
        broken.append(float((rate > .5).mean()))
    res["C"][b] = {"n_neurons_with_damage": len(rows), "mean_r2_from_sum": float(np.mean(obs)) if obs else None, "null_r2": float(np.mean(nulls)) if nulls else None, "median_share_pairs_broken": float(np.median(broken)) if broken else None, "neurons_breaking_le_10pct_pairs": int(np.sum(np.array(broken) <= .1)), "mean_neurons_breaking_each_pair": float(hits.mean()), "max_neurons_breaking_a_pair": int(hits.max()), "pairs_with_no_critical_neuron": int((hits == 0).sum()), "pairs": b * b}
    print("C", b, res["C"][b], flush=True)

    # ---- D: same table at other digit positions? carry as "one more"?
    if b > 2:
        def cellmean(mask, A2, B2):
            out = {}
            for i in range(b):
                for j in range(b):
                    w = mask & (d["a"] == i) & (d["b"] == j)
                    if w.sum() >= 3:
                        out[(i, j)] = X[w].mean(0)
            return out
        c0 = cellmean((d["k"] == 0), None, None)
        c1 = cellmean((d["k"] == 1) & (d["c"] == 0), None, None)
        c1c = cellmean((d["k"] == 1) & (d["c"] == 1), None, None)
        def corr(u, v, mapfn=lambda p: p):
            ks = [p for p in u if mapfn(p) in v]
            x = np.stack([u[p] for p in ks]); y = np.stack([v[mapfn(p)] for p in ks])
            return float(np.corrcoef((x - x.mean(0)).ravel(), (y - y.mean(0)).ravel())[0, 1])
        res["D"][b] = {"digit0_vs_digit1_table": corr(c0, c1), "carry_vs_nocarry_same_pair": corr(c1, c1c), "carry_vs_a_plus_one": corr(c1c, c1, lambda p: (p[0] + 1, p[1]) if p[0] + 1 < b else (-1, -1)), "carry_vs_b_plus_one": corr(c1c, c1, lambda p: (p[0], p[1] + 1) if p[1] + 1 < b else (-1, -1))}
        print("D", b, {a: round(v, 3) for a, v in res["D"][b].items()}, flush=True)
save_json("s12_lookup", res)
print("DONE")
