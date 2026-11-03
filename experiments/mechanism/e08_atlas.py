"""E8: exhaustive component atlas. Every MLP neuron and head, in every base, with digit-level detail.

The team's sweep ablated 321 of 520 components (those Lasso flagged) and kept one number per component.
This sweep covers all 520 and records what actually goes wrong:

  * teacher-forced digit accuracy by digit index and by carry-in;
  * the ERROR SIGNATURE: when the digit becomes wrong, what did the model say instead?
       dropped carry, spurious carry, another base's digit (the translator signature), operand copy...
  * zero ablation versus mean ablation, because zero is a big perturbation;
  * WHERE it matters: ablating only over few-shot context, operand tokens, or query positions.

Uses the team's exact 1000 prompts per base, read from ablations/*.pt, and validates against their drops.
"""

import io
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, DATA, DIGITS, Batch, Model, Problem, save_json, schoolbook, to_base

model = Model()
ROOT = Path(__file__).resolve().parents[2]
SIGS = ["dropped_carry", "spurious_carry", "other_base_digit", "copy_operand", "eos", "other"]
CONDS = ["zero_all", "mean_all", "zero_context", "zero_operands", "zero_query"]
KMAX = 4


class _Stub:
    def __init__(self, *a, **k):
        pass

    def __setstate__(self, s):
        pass


class _U(pickle.Unpickler):
    """Reads the pickled summary without executing anything from the file."""

    def find_class(self, m, n):
        if m == "collections" and n == "OrderedDict":
            import collections

            return collections.OrderedDict
        return _Stub

    def persistent_load(self, pid):
        return None


def load_team(base: int) -> tuple[list[Problem], dict]:
    """The team's prompts and accuracy drops for one base."""
    f = [str(x) for x in (ROOT / "ablations").iterdir() if f"[b={base}]" in x.name][0]
    z = zipfile.ZipFile(f)
    name = [n for n in z.namelist() if n.endswith("data.pkl")][0]
    d = _U(io.BytesIO(z.read(name))).load()
    problems = []
    for r in d["baseline"]:
        lines = r["prompt"].split("\n")
        shots = []
        for line in lines[:-1]:
            x, y = line.split("=")[0].split("+")
            shots.append((int(x, base), int(y, base)))
        problems.append(Problem(r["metadata"]["a"], r["metadata"]["b"], base, shots))
    drops = {(x["layer_idx"], x["type"], x["local_idx"]): x["accuracy_drop"] for x in d["ablations"]}
    return problems, {"drops": drops, "baseline": d["baseline_accuracy"]}


class Prepared:
    """Pre-tokenised batches and per-query ground truth for fast repeated forward passes."""

    def __init__(self, problems: list[Problem], base: int, bs: int = 500) -> None:
        self.base = base
        self.batches = []
        for i in range(0, len(problems), bs):
            chunk = problems[i : i + bs]
            bt = Batch(model, chunk)
            n = bt.ids.shape[1]
            region = np.zeros((len(chunk), n), dtype=np.int8)  # 0 context, 1 operands, 2 query, 3 after answer
            rows, qpos, ks, truth, cin, sigs = [], [], [], [], [], {t: [] for t in [*SIGS[:2], "copy_a", "copy_b", *[f"b{x}" for x in BASES]]}
            for j, p in enumerate(chunk):
                region[j, bt.p_astart[j] : bt.p_eq[j]] = 1
                region[j, bt.p_eq[j] : bt.p_eq[j] + bt.len_ans[j] + 1] = 2
                region[j, bt.p_eq[j] + bt.len_ans[j] + 1 :] = 3
                sb = schoolbook(p.a, p.b, base)
                for k in range(bt.len_ans[j] + 1):  # includes the EOS query
                    rows.append(j)
                    qpos.append(bt.p_eq[j] + k)
                    ks.append(k)
                    if k == bt.len_ans[j]:
                        truth.append(model.eos)
                    else:
                        truth.append(model.vocab[DIGITS[sb["o"][k]]])
                    cin.append(sb["c_in"][k])
                    raw = sb["a"][k] + sb["b"][k]
                    sigs["dropped_carry"].append(model.vocab[DIGITS[raw % base]])
                    sigs["spurious_carry"].append(model.vocab[DIGITS[(raw + 1) % base]])
                    sigs["copy_a"].append(model.vocab[DIGITS[sb["a"][k]]])
                    sigs["copy_b"].append(model.vocab[DIGITS[sb["b"][k]]])
                    # what the digit would be under another base's modulus, reusing the raw digit sum
                    s = sb["s"][k]
                    for t in BASES:
                        v = s % t
                        sigs[f"b{t}"].append(model.vocab[DIGITS[v]])
            self.batches.append(
                {
                    "ids": bt.ids,
                    "region": torch.as_tensor(region, device=model.device),
                    "rows": torch.tensor(rows, device=model.device),
                    "q": torch.tensor(qpos, device=model.device),
                    "k": torch.tensor(ks, device=model.device),
                    "truth": torch.tensor(truth, device=model.device),
                    "cin": torch.tensor(cin, device=model.device),
                    **{f"sig_{s}": torch.tensor(v, device=model.device) for s, v in sigs.items()},
                    "n_problems": len(chunk),
                }
            )

    @torch.no_grad()
    def evaluate(self, hook_fn=None) -> dict:
        """Run with a hook factory (batch -> hooks) and collect digit-level statistics."""
        kk = KMAX + 1  # last bin pools k >= KMAX
        correct = torch.zeros((kk, 2), device=model.device)
        total = torch.zeros((kk, 2), device=model.device)
        sig = torch.zeros(len(SIGS), device=model.device)
        all_ok = 0
        n_prob = 0
        for B in self.batches:
            hooks = hook_fn(B) if hook_fn else None
            logits, _ = model.forward(B["ids"], hooks=hooks)
            pred = logits[B["rows"], B["q"]].argmax(-1)
            ok = pred == B["truth"]
            kb = B["k"].clamp_max(KMAX)
            for c in (0, 1):
                m = B["cin"] == c
                correct[:, c] += torch.bincount(kb[m], weights=ok[m].float(), minlength=kk)
                total[:, c] += torch.bincount(kb[m], minlength=kk)
            # exact: all queries of a problem are right
            wrong_per_prob = torch.bincount(B["rows"], weights=(~ok).float(), minlength=B["n_problems"])
            all_ok += (wrong_per_prob == 0).sum().item()
            n_prob += B["n_problems"]
            w = ~ok
            if w.any():
                pw, cw = pred[w], B["cin"][w]
                is_eos = pw == model.eos
                dc = (pw == B["sig_dropped_carry"][w]) & (cw == 1)
                sc = (pw == B["sig_spurious_carry"][w]) & (cw == 0)
                ob = torch.zeros_like(pw, dtype=torch.bool)
                for t in BASES:
                    if t != self.base:
                        ob |= pw == B[f"sig_b{t}"][w]
                cp = (pw == B["sig_copy_a"][w]) | (pw == B["sig_copy_b"][w])
                claimed = torch.zeros_like(pw, dtype=torch.bool)
                for si, mask in enumerate([dc, sc, ob, cp, is_eos]):
                    sig[si] += (mask & ~claimed).sum()
                    claimed |= mask
                sig[len(SIGS) - 1] += (~claimed).sum()
        return {
            "exact": all_ok / n_prob,
            "digit_acc": (correct / total.clamp_min(1)).cpu().numpy(),
            "digit_n": total.cpu().numpy(),
            "signature": sig.cpu().numpy(),
        }


def means_for(prep: Prepared) -> dict:
    """Mean neuron and head activation for mean-ablation, over test-line and answer positions."""
    sums = {f"mlp_post{L}": 0 for L in range(2)} | {f"z{L}": 0 for L in range(2)}
    cnt = 0
    with torch.no_grad():
        for B in prep.batches:
            _, c = model.forward(B["ids"], capture=set(sums))
            mask = (B["region"] == 1) | (B["region"] == 2)
            for k in sums:
                sums[k] = sums[k] + c[k][mask].sum(0)
            cnt += mask.sum().item()
    return {k: v / cnt for k, v in sums.items()}


def make_hook(layer: int, kind: str, idx: int, cond: str, means: dict):
    """Build a hook factory that ablates one component under one condition."""
    name = f"mlp_post{layer}" if kind == "mlp" else f"z{layer}"

    def factory(B: dict) -> dict:
        region = B["region"]
        if cond in ("zero_all", "mean_all"):
            mask = None
        elif cond == "zero_context":
            mask = region == 0
        elif cond == "zero_operands":
            mask = region == 1
        else:
            mask = region == 2
        value = means[name][idx] if cond == "mean_all" else 0.0

        def fn(t: torch.Tensor) -> torch.Tensor:
            t = t.clone()
            if kind == "mlp":
                if mask is None:
                    t[..., idx] = value
                else:
                    t[..., idx][mask] = 0.0
            else:
                if mask is None:
                    t[:, :, idx] = value
                else:
                    t[:, :, idx][mask] = 0.0
            return t

        return {name: fn}

    return factory


components = [(L, "head", h) for L in range(2) for h in range(4)] + [(L, "mlp", n) for L in range(2) for n in range(256)]
out: dict = {"conds": CONDS, "sigs": SIGS, "kmax": KMAX, "components": components, "bases": {}}
arrs: dict = {}
for base in BASES:
    t0 = time.time()
    problems, team = load_team(base)
    prep = Prepared(problems, base)
    means = means_for(prep)
    baseline = prep.evaluate()
    print(f"base {base}: teacher-forced exact {baseline['exact']:.4f}, team generation baseline {team['baseline']:.4f}", flush=True)
    n_comp, n_cond = len(components), len(CONDS)
    exact = np.zeros((n_comp, n_cond))
    dacc = np.zeros((n_comp, n_cond, KMAX + 1, 2))
    sigs = np.zeros((n_comp, n_cond, len(SIGS)))
    for ci, (L, kind, idx) in enumerate(components):
        for cj, cond in enumerate(CONDS):
            r = prep.evaluate(make_hook(L, kind, idx, cond, means))
            exact[ci, cj] = r["exact"]
            dacc[ci, cj] = r["digit_acc"]
            sigs[ci, cj] = r["signature"]
        if ci % 40 == 0:
            print(f"  base {base} comp {ci}/{n_comp}  {time.time() - t0:.0f}s", flush=True)
    arrs[base] = {"exact": exact, "digit_acc": dacc, "sig": sigs}
    team_drop = np.array([team["drops"].get((L, k, i), np.nan) for L, k, i in components])
    drop_tf = baseline["exact"] - exact[:, 0]
    ok = ~np.isnan(team_drop)
    corr = float(np.corrcoef(team_drop[ok], drop_tf[ok])[0, 1])
    print(f"  validation: corr(team drop, my teacher-forced zero drop) = {corr:.4f}, max gap {np.nanmax(np.abs(team_drop - drop_tf)):.3f}", flush=True)
    out["bases"][base] = {
        "baseline_exact": baseline["exact"],
        "baseline_digit_acc": baseline["digit_acc"],
        "baseline_digit_n": baseline["digit_n"],
        "team_baseline": team["baseline"],
        "validation_corr": corr,
        "validation_max_gap": float(np.nanmax(np.abs(team_drop - drop_tf))),
    }
    np.savez_compressed(DATA / f"e08_atlas_b{base}.npz", exact=exact, digit_acc=dacc, sig=sigs, team_drop=team_drop, baseline_exact=baseline["exact"], baseline_digit=baseline["digit_acc"])
    save_json("e08_atlas", out)
print("done")
