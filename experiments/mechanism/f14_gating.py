"""F14 (stored + weights, replay): how does base select the modulus, how does carry enter the clock, and is the clock reused?

A. Gating: add ONE constant vector (mean pre-activation difference between bases, on shared digit pairs, learned on half the
   prompts) to all L1 neuron inputs in the other half. Conflict pairs only. Does the rule flip? No digit information changes.
   Also: the gain test. Across neurons, does the constant shift predict a change in tuning amplitude?
B. Carry as phase: at digit 1, compare clock-mode phase for carry 0 vs 1. A clean carry rotates phase by 2*pi*f/b.
C. Reuse: at digits 0, 1, 2, are the top neurons per frequency the same?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "gain": {}, "B": {}, "C": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); tt = torch.as_tensor(tok, device=DEVICE)
Wo = sd["h.1.mlp.c_proj.weight"]; W_tok = (Wo @ model.WE[tt].T).cpu().numpy()
bo = sd["h.1.mlp.c_proj.bias"]


def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; ak, bk = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]
    return dict(a=ak, b=bk, c=c, s=ak + bk + c, mid=mid, pre=pre, idx=t.meta["idx"][rows])


D = {(b, k): load(b, k) for b in (8, 10, 16) for k in (0, 1, 2)}
def logits(mid, pre):
    return model.unembed(mid + gelu_new(pre) @ Wo + bo)

for k in (0, 1):
    for x, y in [(8, 10), (10, 8), (10, 16), (16, 10), (8, 16), (16, 8)]:
        dx, dy = D[(x, k)], D[(y, k)]
        lim = min(x, y)
        # cell-balanced mean difference on shared cells, on train half
        trx, try_ = dx["idx"] % 2 == 0, dy["idx"] % 2 == 0
        diffs = []
        for i in range(lim):
            for j in range(lim):
                for c in (0, 1):
                    mx = trx & (dx["a"] == i) & (dx["b"] == j) & (dx["c"] == c); my = try_ & (dy["a"] == i) & (dy["b"] == j) & (dy["c"] == c)
                    if mx.sum() >= 2 and my.sum() >= 2:
                        diffs.append(dy["pre"][torch.as_tensor(my, device=DEVICE)].mean(0) - dx["pre"][torch.as_tensor(mx, device=DEVICE)].mean(0))
        if not diffs: continue
        delta = torch.stack(diffs); cdelta = delta.mean(0)
        share = float(cdelta.pow(2).sum() / delta.pow(2).sum(1).mean())
        te = ~trx & (dx["a"] < lim) & (dx["b"] < lim) & (dx["s"] >= lim)
        tet = torch.as_tensor(te, device=DEVICE)
        s = dx["s"][te]; rx, ry = tok[s % x], tok[s % y]
        d = rx != ry
        curve = {}
        for al in (0., .25, .5, .75, 1., 1.25, 1.5):
            p = logits(dx["mid"][tet], dx["pre"][tet] + al * cdelta).argmax(-1).cpu().numpy()
            curve[al] = {"own": float((p[d] == rx[d]).mean()), "other": float((p[d] == ry[d]).mean())}
        res["A"][f"k{k}/{x}->{y}"] = {"curve": curve, "constant_share_of_shift": share, "n": int(d.sum())}
        print(f"A k{k} {x}->{y} n={int(d.sum())} constant share of per-pair shift {share:.2f}:", {a: (round(v['own'], 2), round(v['other'], 2)) for a, v in curve.items()}, flush=True)
    save_json("f14_gating", res)

# gain: does a constant shift predict amplitude change?
for x, y in [(8, 10), (10, 16), (8, 16)]:
    dx, dy = D[(x, 0)], D[(y, 0)]
    lim = min(x, y)
    def tab(d):
        P = d["pre"].cpu().numpy(); Q = gelu_new(d["pre"]).cpu().numpy(); T = np.zeros((lim, lim, 256)); M = np.zeros((lim, lim, 256))
        for i in range(lim):
            for j in range(lim):
                m = (d["a"] == i) & (d["b"] == j); T[i, j] = Q[m].mean(0); M[i, j] = P[m].mean(0)
        return T, M
    Tx, Mx = tab(dx); Ty, My = tab(dy)
    shift = (My - Mx).mean((0, 1))
    ax, ay = Tx.std((0, 1)), Ty.std((0, 1))
    act = (ax + ay) > np.quantile(ax + ay, .5)
    res["gain"][f"{x}v{y}"] = {"corr_shift_vs_amp_change": float(np.corrcoef(shift[act], (ay - ax)[act])[0, 1]), "corr_table_change_vs_constant": float(1 - ((Ty - Tx - (Ty - Tx).mean((0, 1), keepdims=True)) ** 2).sum() / ((Ty - Tx) ** 2).sum())}
    print("gain", x, y, res["gain"][f"{x}v{y}"], flush=True)

# B carry phase
for b, fs in ((8, [1, 2, 3]), (10, [1, 2, 3]), (16, [1, 2, 3, 5])):
    d = D[(b, 1)]; Q = gelu_new(d["pre"]).cpu().numpy()
    raw = (d["a"] + d["b"]) % b
    T = np.zeros((2, b, 256)); ok = np.zeros((2, b), bool)
    for c in (0, 1):
        for r in range(b):
            m = (d["c"] == c) & (raw == r)
            if m.sum() >= 3: T[c, r] = Q[m].mean(0); ok[c, r] = True
    T = T - T.mean((1), keepdims=True)
    F0, F1 = np.fft.fft(T[0], axis=0), np.fft.fft(T[1], axis=0)
    w = W_tok[:, :b] - W_tok[:, :b].mean(1, keepdims=True); FW = np.fft.fft(w, axis=1)
    out = {}
    for f in fs:
        al = np.real(F0[f] * np.conj(FW[:, f])) + np.real(F1[f] * np.conj(FW[:, f]))
        top = np.argsort(-al)[:16]
        rot = np.angle(np.exp(1j * (np.angle(F1[f, top]) - np.angle(F0[f, top]))).mean())
        R = float(np.abs(np.exp(1j * (np.angle(F1[f, top]) - np.angle(F0[f, top]))).mean()))
        out[f] = {"measured_rotation": float(rot), "predicted_rotation": float(np.angle(np.exp(2j * np.pi * f / b))), "R": R}
    res["B"][b] = out
    print("B", b, {f: (round(v["measured_rotation"], 2), round(v["predicted_rotation"], 2), round(v["R"], 2)) for f, v in out.items()}, flush=True)

# C reuse across digit index
for b, fs in ((8, [1, 2, 4]), (10, [1, 2, 3, 5]), (16, [1, 2, 3, 5, 8])):
    tops = {}
    for k in (0, 1, 2):
        d = D[(b, k)]; Q = gelu_new(d["pre"]).cpu().numpy(); o = d["s"] % b
        T = np.stack([Q[o == v].mean(0) if (o == v).any() else Q.mean(0) for v in range(b)]); T -= T.mean(0)
        F = np.fft.fft(T, axis=0); w = W_tok[:, :b] - W_tok[:, :b].mean(1, keepdims=True); FW = np.fft.fft(w, axis=1)
        tops[k] = {f: set(np.argsort(-np.real(F[f] * np.conj(FW[:, f])))[:16].tolist()) for f in fs}
    res["C"][b] = {f: {"k0k1": len(tops[0][f] & tops[1][f]), "k1k2": len(tops[1][f] & tops[2][f]), "k0k2": len(tops[0][f] & tops[2][f])} for f in fs}
    print("C overlap of 16:", b, res["C"][b], flush=True)
save_json("f14_gating", res); print("DONE")
