"""F12 (stored + weights): is the output a Fourier clock over the prompt's base?

A. Per base and digit: g(offset) = mean logit at (t - s) mod b. DFT of g: which frequencies? Is g even (symmetric)?
   R^2 excluding offset 0, since a correct peak alone would inflate fit. Null: shuffle token labels within alphabet.
B. Sufficiency: keep only selected frequencies of the digit logits, per row. Is argmax still right?
C. Neurons: per neuron, tuning over s mod b, and write vector over the alphabet. Dominant frequency of each, phase alignment,
   and each neuron's contribution to the clock mode. Same neurons across bases?
D. Knockout in neuron space: remove neurons by dominant frequency and measure digit accuracy (replay).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); res = {"A": {}, "B": {}, "C": {}, "D": {}}
tok = np.array([model.vocab[c] for c in DIGITS])
tt = torch.as_tensor(tok, device=DEVICE)
Wo = model.sd["h.1.mlp.c_proj.weight"]
W_tok = (Wo @ model.WE[tt].T).cpu().numpy()


def load(b, k):
    t = Table(b); rows = t.select(QUERY, k); A, B = t.meta["a"][rows].astype(np.int64), t.meta["b"][rows].astype(np.int64)
    keep = ((A + B) >= b**k) & (t.meta["correct"][rows] == 1); rows, A, B = rows[keep], A[keep], B[keep]
    M = b**k; ak, bk = (A // M) % b, (B // M) % b; c = (((A % M) + (B % M)) >= M).astype(int)
    return t.get("resid1", rows), t.get("mlp1", rows), ak + bk + c


store = {}
for b in BASES:
    res["A"][b] = {}
    for k in (0, 1, 2):
        R, X, s = load(b, k)
        if len(s) < 200 or b == 2 and k > 2: continue
        Lg = model.unembed(torch.as_tensor(R, device=DEVICE)).cpu().numpy()[:, tok[:b]]
        Lc = Lg - Lg.mean(1, keepdims=True)
        o = s % b
        off = (np.arange(b)[None] - o[:, None]) % b
        g = np.array([Lc[off == x].mean() for x in range(b)])
        pred = g[off]
        mask = off != 0
        r2 = 1 - ((Lc[mask] - pred[mask]) ** 2).sum() / ((Lc[mask] - Lc[mask].mean()) ** 2).sum()
        rng = np.random.default_rng(0); nulls = []
        for _ in range(100):
            perm = rng.permutation(b)
            of2 = (perm[np.arange(b)][None] - perm[o][:, None]) % b
            g2 = np.array([Lc[of2 == x].mean() for x in range(b)])
            m2 = of2 != 0
            nulls.append(1 - ((Lc[m2] - g2[of2][m2]) ** 2).sum() / ((Lc[m2] - Lc[m2].mean()) ** 2).sum())
        F = np.fft.rfft(g - g.mean())
        power = np.abs(F[1:]) ** 2
        sym = float(np.corrcoef(g[1:], g[1:][::-1])[0, 1]) if b > 2 else 1.0
        res["A"][b][k] = {"g": g, "r2_off_target": float(r2), "null_mean": float(np.mean(nulls)), "null_max": float(np.max(nulls)), "symmetry_corr": sym, "freq_power_share": (power / power.sum()).tolist(), "n": len(s)}
        print(f"A b{b} k{k}: off-target R2 {r2:.3f} (null mean {np.mean(nulls):.3f}, max {np.max(nulls):.3f}), symmetry {sym:.3f}, freq share f=1..: {np.round(power / power.sum(), 2).tolist()}", flush=True)
        if k == 0 and b > 2:
            # B sufficiency
            D = np.fft.rfft(Lc, axis=1)
            truth = o
            base_acc = float((Lc.argmax(1) == truth).mean())
            order = np.argsort(-power) + 1
            rowsuff = {"all": base_acc}
            for m in range(1, len(power) + 1):
                keep = order[:m]
                E = np.zeros_like(D); E[:, keep] = D[:, keep]
                rowsuff[f"top{m}"] = float((np.fft.irfft(E, n=b, axis=1).argmax(1) == truth).mean())
            singles = {}
            for f in range(1, len(power) + 1):
                E = D.copy(); E[:, f] = 0
                singles[f] = float((np.fft.irfft(E, n=b, axis=1).argmax(1) == truth).mean())
            res["B"][b] = {"keep_top": rowsuff, "order": order.tolist(), "remove_single": singles}
            print("  B keep top freqs", order.tolist(), {a: round(v, 3) for a, v in rowsuff.items()}, "remove one:", {a: round(v, 3) for a, v in singles.items()}, flush=True)
            store[b] = (X, s, Lc, R)

# C neuron level
pw = {}
for b in (8, 10, 16):
    X, s, Lc, R = store[b]
    o = s % b
    T = np.stack([X[o == v].mean(0) for v in range(b)])  # by s mod b
    Tc = T - T.mean(0)
    w = W_tok[:, :b] - W_tok[:, :b].mean(1, keepdims=True)
    FT = np.fft.rfft(Tc, axis=0)[1:]  # [freq, neurons]
    FW = np.fft.rfft(w, axis=1)[:, 1:]  # [neurons, freq]
    # contribution of neuron n at frequency f to the clock mode: Re(FT * conj(FW)), positive means aligned
    align = np.real(FT.T * np.conj(FW))  # [n, f]
    total = align.sum(0)
    domT = np.abs(FT).argmax(0) + 1; domW = np.abs(FW).argmax(1) + 1
    amp = np.abs(FT).max(0) * np.abs(FW).max(1)
    act = amp > np.quantile(amp, 0.5)
    pwT = (np.abs(FT) ** 2).sum(1) / (np.abs(FT) ** 2).sum()
    pwW = (np.abs(FW) ** 2).sum(0) / (np.abs(FW) ** 2).sum()
    frac_pos = align.clip(0).sum(0) / np.abs(align).sum(0)
    # how much of each neuron's mod-b tuning power sits in one frequency?
    conc = (np.abs(FT) ** 2).max(0) / (np.abs(FT) ** 2).sum(0).clip(1e-9)
    res["C"][b] = {"tuning_power_by_f": pwT, "write_power_by_f": pwW, "clock_contribution_by_f": total, "aligned_fraction_by_f": frac_pos, "same_dom_freq_rate": float((domT[act] == domW[act]).mean()), "chance_same": float(1 / FT.shape[0]), "single_freq_concentration_top_half": float(conc[act].mean()), "neurons_top_by_f": {int(f + 1): np.argsort(-align[:, f])[:12].tolist() for f in range(align.shape[1])}}
    pw[b] = align
    print(f"C b{b}: tuning power by f {np.round(pwT, 2).tolist()}\n    write power by f {np.round(pwW, 2).tolist()}\n    clock contribution by f {np.round(total / total.sum(), 2).tolist()}, aligned fraction {np.round(frac_pos, 2).tolist()}\n    neurons with same dominant freq in tuning and write: {res['C'][b]['same_dom_freq_rate']:.2f} (chance {1 / FT.shape[0]:.2f}), concentration {res['C'][b]['single_freq_concentration_top_half']:.2f}", flush=True)
    # D knockout by neurons contributing to a frequency, digit accuracy on digit 0
    tR = torch.as_tensor(R, device=DEVICE); tX = torch.as_tensor(X, device=DEVICE); m = tX.mean(0)
    drops = {}
    for f in range(align.shape[1]):
        top = np.argsort(-align[:, f])[:16]
        idx = torch.as_tensor(top.copy(), device=DEVICE)
        d = torch.zeros_like(tX); d[:, idx] = (m - tX)[:, idx]
        p = model.unembed(tR + d @ Wo).cpu().numpy()[:, tok[:b]].argmax(1)
        drops[f + 1] = float((p == o).mean())
    rng = np.random.default_rng(b); ctrl = []
    for _ in range(5):
        idx = torch.as_tensor(rng.choice(256, 16, replace=False), device=DEVICE)
        d = torch.zeros_like(tX); d[:, idx] = (m - tX)[:, idx]
        ctrl.append(float((model.unembed(tR + d @ Wo).cpu().numpy()[:, tok[:b]].argmax(1) == o).mean()))
    res["D"][b] = {"digit0_acc_after_removing_top16_for_f": drops, "random16": float(np.mean(ctrl))}
    print("  D accuracy after mean-ablating top 16 neurons per frequency:", {a: round(v, 3) for a, v in drops.items()}, "random 16:", round(float(np.mean(ctrl)), 3), flush=True)
# overlap between bases for shared frequencies: parity modes, low modes
for x, y, fx, fy, nm in [(8, 10, 4, 5, "parity"), (10, 16, 5, 8, "parity"), (8, 16, 4, 8, "parity"), (8, 16, 1, 2, "period 8 in both"), (8, 10, 1, 1, "lowest"), (10, 16, 1, 1, "lowest")]:
    a, c = set(np.argsort(-pw[x][:, fx - 1])[:16].tolist()), set(np.argsort(-pw[y][:, fy - 1])[:16].tolist())
    res["C"][f"overlap {x}f{fx} vs {y}f{fy}"] = {"name": nm, "overlap_of_16": len(a & c)}
    print("overlap", x, fx, y, fy, nm, len(a & c), "of 16 (chance about 1)", flush=True)
save_json("f12_clock", res); print("DONE")
