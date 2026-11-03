"""F02 (forward passes): what does layer 0 do, and by which route does its damage reach the answer?

Ablate each important L0 component (zero at all positions), teacher-forced, digit accuracy only. Then:
  - where: zero only over context, operands, or query positions;
  - attention shift: total-variation change of L1 attention at queries;
  - rescue: restore clean L1 attention patterns; or clean residual at query positions only; or clean head outputs.
Recovery near 1 for attention means the neuron works by steering what layer 1 reads (the addressing idea).
"""
import io, pickle, re, sys, zipfile, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Batch, Model, iter_batches, sample_problems, save_json, schoolbook
torch.set_grad_enabled(False)
model = Model()
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
    team[int(re.search(r"b=(\d+)", f.name).group(1))] = {x["local_idx"]: x["accuracy_drop"] for x in d["ablations"] if x["layer_idx"] == 0 and x["type"] == "mlp"}
KMAX = 3
res = {}


def prep(base, n=1200, seed=91):
    out = []
    for chunk in iter_batches(sample_problems(n, base, seed=seed), 300):
        bt = Batch(model, chunk)
        rows, q, k, tr = [], [], [], []
        L = bt.ids.shape[1]
        region = torch.zeros((len(chunk), L), dtype=torch.int8)
        for i, p in enumerate(chunk):
            sb = schoolbook(p.a, p.b, base)
            region[i, bt.p_astart[i]:bt.p_eq[i]] = 1
            region[i, bt.p_eq[i]:] = 2
            for j in range(bt.len_ans[i]):
                rows.append(i); q.append(bt.p_eq[i] + j); k.append(j); tr.append(model.vocab[DIGITS[sb["o"][j]]])
        out.append(dict(ids=bt.ids, region=region.to(DEVICE), rows=torch.tensor(rows, device=DEVICE), q=torch.tensor(q, device=DEVICE), k=torch.tensor(k, device=DEVICE), truth=torch.tensor(tr, device=DEVICE), n=len(chunk)))
    return out


def run(batches, comp, cond="all", rescue=None, clean=None):
    L, kind, idx = comp
    name = f"mlp_post{L}" if kind == "mlp" else f"z{L}"
    ok_tot, n_tot, tv = torch.zeros(KMAX + 1), torch.zeros(KMAX + 1), np.zeros(4)
    exact, probs = 0, 0
    tv_n = 0
    for bi, B in enumerate(batches):
        mask = None if cond == "all" else B["region"] == {"context": 0, "operands": 1, "query": 2}[cond]
        def ab(t):
            t = t.clone()
            if kind == "mlp":
                if mask is None: t[..., idx] = 0.
                else: t[..., idx][mask] = 0.
            else:
                if mask is None: t[:, :, idx] = 0.
                else: t[:, :, idx][mask] = 0.
            return t
        hooks = {name: ab}
        if rescue == "attention":
            hooks["pattern1"] = lambda t, c=clean[bi]: c["pattern1"]
        if rescue == "heads":
            hooks["z1"] = lambda t, c=clean[bi]: c["z1"]
        if rescue == "residual_query":
            qm = B["region"] == 2
            def rq(t, c=clean[bi], qm=qm):
                t = t.clone(); t[qm] = c["resid_pre1"][qm]; return t
            hooks["resid_pre1"] = rq
        lg, cache = model.forward(B["ids"], hooks=hooks, capture={"pattern1"})
        pred = lg[B["rows"], B["q"]].argmax(-1)
        ok = pred == B["truth"]
        kb = B["k"].clamp_max(KMAX)
        ok_tot += torch.bincount(kb, weights=ok.float(), minlength=KMAX + 1).cpu()
        n_tot += torch.bincount(kb, minlength=KMAX + 1).cpu()
        exact += (torch.bincount(B["rows"], weights=(~ok).float(), minlength=B["n"]) == 0).sum().item(); probs += B["n"]
        if clean is not None and rescue is None:
            p, c0 = cache["pattern1"], clean[bi]["pattern1"]
            d = 0.5 * (p - c0).abs().sum(-1)  # [b, h, pos]
            tv += d[B["rows"][:, None].expand(-1, 4), torch.arange(4, device=DEVICE)[None], B["q"][:, None].expand(-1, 4)].sum(0).cpu().numpy()
            tv_n += len(B["rows"])
    return {"exact": exact / probs, "by_k": (ok_tot / n_tot.clamp_min(1)).numpy(), "tv_per_head": tv / max(tv_n, 1)}


for base in BASES:
    batches = prep(base)
    clean = []
    for B in batches:
        _, c = model.forward(B["ids"], capture={"pattern1", "z1", "resid_pre1"})
        clean.append(c)
    base_r = run(batches, (0, "mlp", 0), cond="context")  # placeholder overwritten below
    lg_ok = []
    ex, pr = 0, 0
    for B in batches:
        lg, _ = model.forward(B["ids"])
        o = lg[B["rows"], B["q"]].argmax(-1) == B["truth"]
        ex += (torch.bincount(B["rows"], weights=(~o).float(), minlength=B["n"]) == 0).sum().item(); pr += B["n"]
    base_exact = ex / pr
    tops = [(0, "mlp", i) for i in sorted(team[base], key=lambda x: -team[base][x])[:10]] + [(0, "head", h) for h in range(4)]
    rest = [(0, "mlp", i) for i in sorted(team[base], key=lambda x: team[base][x])[:4]]
    rows = {}
    for comp in tops + rest:
        r = run(batches, comp, clean=clean)
        row = {"team_drop": team[base].get(comp[2]) if comp[1] == "mlp" else None, "drop": base_exact - r["exact"], "by_k": r["by_k"], "tv_per_head": r["tv_per_head"]}
        for c in ("context", "operands", "query"):
            row[f"drop_{c}"] = base_exact - run(batches, comp, cond=c)["exact"]
        if row["drop"] >= 0.03:
            for rs in ("attention", "residual_query", "heads"):
                e = run(batches, comp, rescue=rs, clean=clean)["exact"]
                row[f"recovery_{rs}"] = (e - r["exact"]) / max(base_exact - r["exact"], 1e-9)
        rows[f"L{comp[0]}{comp[1]}{comp[2]}"] = row
        print(f"b{base} {comp[1]}{comp[2]:>3} drop {row['drop']:.3f} | ctx {row['drop_context']:.3f} oper {row['drop_operands']:.3f} query {row['drop_query']:.3f} | L1 TV {np.round(r['tv_per_head'], 2).tolist()} | recover attn {row.get('recovery_attention', float('nan')):.2f} resid@q {row.get('recovery_residual_query', float('nan')):.2f} heads {row.get('recovery_heads', float('nan')):.2f}", flush=True)
    res[base] = {"baseline_exact": base_exact, "components": rows}
    save_json("f02_layer0", res)
print("DONE")
