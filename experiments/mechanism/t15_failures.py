"""T15 (forward passes). Q9: what goes wrong on large octal (and other) numbers? Q10: inside the hex top-digit stop failure.
Q9. Operands up to 4095 (decimal value). Teacher-forced digit accuracy by digit position k and by whether a_k/b_k values occur at
    position k in training numbers (0..999). Error types: missed/extra carry, digit from a neighbouring column, other.
Q10. Hex, 3-digit operands, no carries, after writing 3 digits: direct contribution of each component (L1 heads, L1 MLP, rest)
    to logit(EOS) - logit('1'). Fit each component on seen top digits (0-3), predict unseen (4-F). Which component goes wrong?
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, schoolbook, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); sd = model.sd; tok = np.array([model.vocab[c] for c in DIGITS]); res = {"Q9": {}, "Q10": {}}
def run(S, cap=None):
    seqs = [[model.vocab[x] for x in s] for s in S]; n = max(map(len, seqs)); X = torch.full((len(S), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): X[j, :len(s)] = torch.tensor(s)
    q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE); lg, c = model.forward(X.to(DEVICE), capture=cap or set()); J = torch.arange(len(S), device=DEVICE)
    return lg[J, q], c, q, J
for b in (8, 10, 16):
    seen = {k: {schoolbook(v, 0, b)["a"][k] for v in range(1000)} for k in range(5)}
    seen_top = {k: max(seen[k]) for k in range(5)}
    rng = random.Random(b * 29); rows = {}
    for k in [k for k in range(5) if b**k * 3 < 8190]:
        S, info = [], []
        while len(S) < 1500:
            a, c_ = rng.randint(0, 4095), rng.randint(0, 4095)
            if a + c_ < b**k: continue
            sb_ = schoolbook(a, c_, b); ans = to_answer(a + c_, b)
            s = "\n".join(render(*x, b) for x in sample_shots(rng, (a, c_))) + f"\n{to_base(a, b)}+{to_base(c_, b)}={ans[:k]}"
            if len(s) > 250: continue
            S.append(s); info.append((sb_["a"][k], sb_["b"][k], sb_["c_in"][k], sb_["o"][k], sb_["a"][k - 1] + sb_["b"][k - 1] if k else -1, sb_["a"][k + 1] + sb_["b"][k + 1], sb_["c_out"][k]))
        pr = []
        for i in range(0, len(S), 300):
            lg, _, _, _ = run(S[i:i + 300]); pr.append(lg.argmax(-1).cpu().numpy())
        pr = np.concatenate(pr); I = np.array(info); ak, bk, c, o, prev, nxt, co = I.T
        unseen = ~(np.isin(ak, list(seen[k])) & np.isin(bk, list(seen[k])))
        ok = pr == tok[o]
        err = ~ok
        types = {"missed or extra carry": pr == tok[(ak + bk + 1 - c) % b], "previous column's digit": (k > 0) & (pr == tok[np.maximum(prev, 0) % b]), "next column's digit": pr == tok[nxt % b], "carry-free top value (a+b+c written unwrapped)": (pr == tok[np.minimum(ak + bk + c, 15)]) & (ak + bk + c < 16) & (ak + bk + c >= b), "EOS or newline": (pr == model.eos) | (pr == model.vocab["\n"])}
        rows[k] = {"accuracy, seen digit values": float(ok[~unseen].mean()) if (~unseen).any() else None, "accuracy, unseen digit values": float(ok[unseen].mean()) if unseen.any() else None,
                   "n unseen": int(unseen.sum()), "error types (share of errors)": {t: float((m & err).sum() / max(err.sum(), 1)) for t, m in types.items()}}
        print(f"Q9 b{b} digit {k}: seen {rows[k]['accuracy, seen digit values']}, unseen {rows[k]['accuracy, unseen digit values']} (n {rows[k]['n unseen']}); errors:", {t: round(v, 2) for t, v in rows[k]["error types (share of errors)"].items()}, flush=True)
    res["Q9"][b] = rows
    save_json("t15_failures", res)
# Q10
rng = random.Random(31); S, info = [], []
while len(S) < 3000:
    x, y = rng.randint(1, 14), rng.randint(1, 14)
    if x + y >= 16: continue
    lo = [rng.randrange(16) for _ in range(2)]; lb = [rng.randrange(16 - lo[i]) for i in range(2)]
    a = x * 256 + lo[0] + 16 * lo[1]; c_ = y * 256 + lb[0] + 16 * lb[1]; ans = to_answer(a + c_, 16)
    s = "\n".join(render(*z, 16) for z in sample_shots(rng, (a, c_))) + f"\n{to_base(a, 16)}+{to_base(c_, 16)}={ans}"
    S.append(s); info.append((x, y, x + y))
I = np.array(info); x, y, w = I.T; seen = (x <= 3) & (y <= 3)
one = model.vocab["1"]; d_tok = (model.WE[model.eos] - model.WE[one]) * sd["ln_f.weight"]
contrib = {nm: [] for nm in ("L1H0", "L1H1", "L1H2", "L1H3", "L1 MLP", "layer 0 + embeddings")}; stop = []
Wp = sd["h.1.attn.c_proj.weight"]
for i in range(0, len(S), 300):
    lg, c, q, J = run(S[i:i + 300], {"z1", "mlp_out1", "resid_pre1", "resid_post1"})
    R = c["resid_post1"][J, q]; scale = (R - R.mean(-1, keepdim=True)).pow(2).mean(-1, keepdim=True).add(1e-5).sqrt()
    z = c["z1"][J, q]
    for h in range(4):
        part = z[:, h] @ Wp[16 * h:16 * h + 16]; contrib[f"L1H{h}"].append(((part - part.mean(-1, keepdim=True)) / scale @ d_tok).cpu().numpy())
    for nm, t in (("L1 MLP", c["mlp_out1"][J, q]), ("layer 0 + embeddings", c["resid_pre1"][J, q])):
        contrib[nm].append(((t - t.mean(-1, keepdim=True)) / scale @ d_tok).cpu().numpy())
    stop.append((lg.argmax(-1) == model.eos).cpu().numpy())
stop = np.concatenate(stop); out = {"stop rate seen": float(stop[seen].mean()), "stop rate unseen": float(stop[~seen].mean())}
Xf = np.c_[x, y, w, np.ones(len(x))].astype(float)
for nm in contrib:
    v = np.concatenate(contrib[nm]); co = np.linalg.lstsq(Xf[seen], v[seen], rcond=None)[0]; pred = Xf @ co
    out[nm] = {"mean seen": float(v[seen].mean()), "mean unseen": float(v[~seen].mean()), "unseen minus prediction from seen fit": float((v - pred)[~seen].mean()),
               "unseen, stopping failed: minus prediction": float((v - pred)[(~seen) & (stop == 0)].mean()) if ((~seen) & (stop == 0)).any() else None}
# attention of L1 heads at the final query to a's top digit and b's top digit, seen vs unseen
att = {h: {"a_top": [], "b_top": []} for h in range(4)}
for i in range(0, len(S), 300):
    ch = S[i:i + 300]; lg, c, q, J = run(ch, {"pattern1"})
    st = torch.tensor([s.rfind("\n") + 1 for s in ch], device=DEVICE); pl = torch.tensor([s.rfind("+") for s in ch], device=DEVICE)
    for h in range(4):
        r = c["pattern1"][J, h, q]; att[h]["a_top"].append(r[J, st].cpu().numpy()); att[h]["b_top"].append(r[J, pl + 1].cpu().numpy())
for h in range(4):
    for kk in ("a_top", "b_top"):
        v = np.concatenate(att[h][kk]); out[f"L1H{h} attention on {kk} (seen / unseen)"] = (float(v[seen].mean()), float(v[~seen].mean()))
res["Q10"] = out
print("Q10:", {k: ({a: (round(b_, 2) if isinstance(b_, float) else b_) for a, b_ in v.items()} if isinstance(v, dict) else (tuple(round(t, 2) for t in v) if isinstance(v, tuple) else round(v, 2))) for k, v in out.items()}, flush=True)
save_json("t15_failures", res); print("DONE")
