"""F13 (stored + weights): how are the clock modes computed? Tests the trig-identity (phase sum) prediction.

Tables at digit 0 over (a_0, b_0):
A. Pre-activation vs post-activation: pre should be additive (no cross-terms); GELU should create them.
B. 2-D DFT of top clock neurons: power at sum (f,f), difference (f,-f), axes (f,0)+(0,f), other.
C. Phase-sum law: phase of tuning at (f,f), or a's phase + b's phase, should equal the phase of the neuron's write vector.
   Null: pair phases from shuffled neurons.
D. Digit code at the source: DFT of head outputs over digit value, per base. Does code differ by base?
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, gelu_new, save_json
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; res = {"A": {}, "B": {}, "C": {}, "D": {}}
tok = np.array([model.vocab[c] for c in DIGITS]); tt = torch.as_tensor(tok, device=DEVICE)
Wo = sd["h.1.mlp.c_proj.weight"]; W_tok = (Wo @ model.WE[tt].T).cpu().numpy()
FREQS = {8: [1, 2, 4], 10: [1, 2, 3, 5], 16: [1, 2, 3, 5, 8]}
codes = {}
for b in (8, 10, 16):
    t = Table(b); rows = t.select(QUERY, 0); A, B = t.meta["a"][rows] % b, t.meta["b"][rows] % b
    R0 = torch.as_tensor(t.get("resid0", rows), device=DEVICE); Z = torch.as_tensor(t.get("z1", rows), device=DEVICE)
    mid = R0 + Z @ sd["h.1.attn.c_proj.weight"] + sd["h.1.attn.c_proj.bias"]
    pre = (model._ln("h.1.ln_2", mid) @ sd["h.1.mlp.c_fc.weight"] + sd["h.1.mlp.c_fc.bias"]).cpu().numpy()
    post = gelu_new(torch.as_tensor(pre, device=DEVICE)).cpu().numpy()
    def table(X):
        T = np.zeros((b, b, X.shape[1]))
        for i in range(b):
            for j in range(b):
                m = (A == i) & (B == j)
                T[i, j] = X[m].mean(0) if m.any() else X.mean(0)
        return T - T.mean((0, 1), keepdims=True)
    Tp, Tq = table(pre), table(post)
    def shares(T):
        add = T.mean(1, keepdims=True) + T.mean(0, keepdims=True); I = T - add
        return float((I**2).sum() / (T**2).sum())
    res["A"][b] = {"interaction_share_pre": shares(Tp), "interaction_share_post": shares(Tq)}
    print("A", b, res["A"][b], flush=True)
    F = np.fft.fft2(Tq, axes=(0, 1))
    Fp = np.fft.fft2(Tp, axes=(0, 1))
    w = W_tok[:, :b] - W_tok[:, :b].mean(1, keepdims=True); FW = np.fft.fft(w, axis=1)
    out = {}
    rng = np.random.default_rng(0)
    for f in FREQS[b]:
        sumc = F[f, f]  # activation component e^{i2pi f (a+b)/b}
        # contribution to clock: neurons whose (f,f) tuning and write at f are aligned
        align = np.real(sumc * np.conj(FW[:, f]))
        top = np.argsort(-align)[:16]
        P = np.abs(F[:, :, top]) ** 2
        tot = P.sum() - P[0, 0].sum()
        neg = (-f) % b
        cats = {"sum (f,f)": P[f, f].sum() + (P[neg, neg].sum() if neg != f else 0), "difference (f,-f)": (P[f, neg].sum() + P[neg, f].sum()) if neg != f else 0.0, "axes (f,0)+(0,f)": P[f, 0].sum() + P[0, f].sum() + (P[neg, 0].sum() + P[0, neg].sum() if neg != f else 0)}
        cats = {k: float(v / tot) for k, v in cats.items()}
        # phase law at f, using pre-activation single-digit phases
        pa = np.angle(Fp[f, 0]); pb = np.angle(Fp[0, f]); pw = np.angle(FW[:, f]); pss = np.angle(sumc)
        idx = top
        law1 = np.exp(1j * (pss[idx] - pw[idx]))  # F convention: table ~ conj-phase; write vs tuning, same sign under aligned selection
        law2 = np.exp(1j * (pa[idx] + pb[idx] - pss[idx]))
        R1, R2 = float(np.abs(law1.mean())), float(np.abs(law2.mean()))
        nulls = []
        for _ in range(200):
            sh = rng.permutation(256)[:16]; sh2 = rng.permutation(256)[:16]
            nulls.append(np.abs(np.exp(1j * (pa[sh] + pb[sh2] - pss[idx])).mean()))
        out[f] = {"cats": cats, "law_sum_vs_write_R": R1, "law_prea_preb_vs_sum_R": R2, "mean_phase_gap": float(np.angle(law2.mean())), "null95": float(np.quantile(nulls, .95)), "top": top.tolist()}
        print(f"B/C b{b} f={f}: power {({k: round(v, 2) for k, v in cats.items()})} | phase a+b vs sum R={R2:.2f} (gap {out[f]['mean_phase_gap']:.2f}, null95 {out[f]['null95']:.2f})", flush=True)
    res["B"][b] = out
    # source code: z of reading heads over digit value
    Hs = {"H1 for a": (1, A), "H0 for b": (0, B), "H3 for b": (3, B)}
    res["D"][b] = {}
    for nm, (h, lab) in Hs.items():
        zh = Z[:, 16 * h:16 * h + 16].cpu().numpy()
        M = np.stack([zh[lab == v].mean(0) for v in range(b)]); Mc = M - M.mean(0)
        F1 = np.abs(np.fft.rfft(Mc, axis=0)[1:]) ** 2; sh = F1.sum(1) / F1.sum()
        res["D"][b][nm] = {"freq_share": sh.tolist(), "var_explained_by_digit": float(((Mc[lab]) ** 2).sum() / ((zh - zh.mean(0)) ** 2).sum())}
        codes[(b, nm)] = M
        print(f"D b{b} {nm}: digit explains {res['D'][b][nm]['var_explained_by_digit']:.2f}; freq share {np.round(sh, 2).tolist()}", flush=True)
for x, y in [(8, 10), (10, 16), (8, 16)]:
    for nm in ("H1 for a", "H0 for b"):
        n = min(x, y)
        a, c = codes[(x, nm)][:n], codes[(y, nm)][:n]
        res["D"][f"{x}v{y}/{nm}"] = float(np.corrcoef((a - a.mean(0)).ravel(), (c - c.mean(0)).ravel())[0, 1])
        print("D code similarity", x, y, nm, round(res["D"][f"{x}v{y}/{nm}"], 3), flush=True)
save_json("f13_trig", res); print("DONE")
