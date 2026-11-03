"""S11: does neuron 177 control stopping? Replay predictions at EOS queries, and digit-only ablation scores."""
import io, pickle, re, sys, zipfile, collections, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DIGITS, Model, save_json, BASE_COLOR, FIGS, INK, style
from probe_lib import QUERY, Table
torch.set_grad_enabled(False)
model = Model(); Wo = model.sd["h.1.mlp.c_proj.weight"]; tok = np.array([model.vocab[c] for c in DIGITS])
class _S:
    def __init__(self, *a, **k): pass
    def __setstate__(self, s): pass
class _U(pickle.Unpickler):
    def find_class(self, m, n):
        if m == "collections" and n == "OrderedDict": return collections.OrderedDict
        return _S
    def persistent_load(self, p): return None
team = {}
for f in (Path(__file__).resolve().parents[2] / "ablations").iterdir():
    z = zipfile.ZipFile(f); d = _U(io.BytesIO(z.read([n for n in z.namelist() if n.endswith("data.pkl")][0]))).load()
    team[int(re.search(r"b=(\d+)", f.name).group(1))] = {x["local_idx"]: x["accuracy_drop"] for x in d["ablations"] if x["layer_idx"] == 1 and x["type"] == "mlp"}
res = {}; ranks = {}; store = {}
for b in BASES:
    t = Table(b); m = t.meta
    q = np.where((m["kind"] == QUERY) & (m["correct"] == 1))[0]
    A, B, k, idx = m["a"][q].astype(np.int64), m["b"][q].astype(np.int64), m["j"][q].astype(np.int64), m["idx"][q]
    M = b**k; eos = ((A + B) < M) & ~((A + B == 0) & (k == 0))
    sd = ((A + B) // M) % b
    post = torch.as_tensor(t.get("mlp1", q)); resid = torch.as_tensor(t.get("resid1", q))
    uniq = np.unique(idx); pos = np.searchsorted(uniq, idx)
    dig = ~eos; truth = tok[sd]
    p = model.unembed(resid[torch.as_tensor(eos)] - post[torch.as_tensor(eos)][:, 177:178] * Wo[177]).argmax(-1).numpy()
    c = collections.Counter(model.itos[int(x)] for x in p)
    digit_only = lambda P, pos=pos, dig=dig, truth=truth, uniq=uniq: float((np.bincount(pos[dig], weights=(P[dig] != truth[dig]).astype(float), minlength=len(uniq)) == 0).mean())
    base = digit_only(model.unembed(resid).argmax(-1).numpy())
    drop = np.array([base - digit_only(model.unembed(resid - post[:, n:n+1] * Wo[n]).argmax(-1).numpy()) for n in range(256)])
    ids = sorted(team[b]); tv = np.array([team[b][i] for i in ids])
    miss = np.setdiff1d(np.arange(256), ids)
    ranks[b] = np.argsort(-drop); store[b] = (post, resid, digit_only)
    res[b] = {"eos_pred_177_zero": dict(c.most_common(4)), "n_eos": int(eos.sum()), "drop177_digit_only": float(drop[177]), "corr_team_digit_only": float(np.corrcoef(tv, drop[ids])[0, 1]), "max_gap": float(np.abs(tv - drop[ids]).max()), "top": [(int(i), round(float(drop[i]), 3)) for i in ranks[b][:8]], "n_gt05": int((drop > .05).sum()), "not_shortlisted_max": float(drop[miss].max()), "not_shortlisted_top": [int(i) for i in miss[np.argsort(-drop[miss])][:3]], "drop": drop}
    print(b, {k2: v for k2, v in res[b].items() if k2 != "drop"}, flush=True)
SIZES = [8, 16, 32, 64, 128]
res["minimal_digit_only"] = {}
for b in BASES:
    post, resid, dof = store[b]; mean = post.mean(0); full = resid - post @ Wo
    res["minimal_digit_only"][b] = {}
    for src in BASES:
        row = {}
        for n in SIZES:
            keep = torch.as_tensor(ranks[src][:n].copy()); pp = mean.expand_as(post).clone(); pp[:, keep] = post[:, keep]
            row[n] = dof(model.unembed(full + pp @ Wo).argmax(-1).numpy())
        res["minimal_digit_only"][b][src] = row
    print("MIN", b, {s: [round(x, 2) for x in r.values()] for s, r in res["minimal_digit_only"][b].items()}, flush=True)
save_json("s11_eos_check", res)
import matplotlib; import matplotlib.pyplot as plt
style()
s02 = json.load(open(Path(__file__).parent / "data" / "s09_wrap.json"))["profile"]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), sharey=True)
for ax, n in zip(axes, ("22", "119")):
    for bb in (8, 10, 16):
        y = s02[n][f"by_s_b{bb}"]; ax.plot(range(len(y)), y, color=BASE_COLOR[bb], marker="o", ms=3, lw=1.3, label=f"Base {bb}")
    ax.set_title(f"Layer 1 neuron {n}"); ax.set_xlabel("Raw digit sum s (digit 0)"); ax.legend(fontsize=8)
axes[0].set_ylabel("Mean activation")
fig.suptitle("Top neurons alternate with parity instead of stepping at the base", color=INK); fig.tight_layout(); fig.savefig(FIGS / "parity_neurons.png"); plt.close(fig)
g = json.load(open(Path(__file__).parent / "data" / "s10_rest.json"))["C"]
fig, ax = plt.subplots(figsize=(5.6, 3))
for bb in BASES:
    v = g[f"gap{bb}"]; x = sorted(map(int, v)); ax.plot(x, [v[str(i)][0] for i in x], color=BASE_COLOR[bb], marker="o", ms=3, label=f"Base {bb}")
ax.set_xlabel("Difference in operand lengths (digits)"); ax.set_ylabel("Exact accuracy"); ax.legend(fontsize=8); ax.set_title("Binary errors grow with operand length mismatch")
fig.tight_layout(); fig.savefig(FIGS / "length_gap.png")
