"""E11: causal tests. Probes can only show what is present; these test what is used.

  A. Cross-base activation patching (the plan's RQ5). Patch a stage's residual at q_k from a source problem in
     base A into a receiver in base B, and sort the output digit:
        receiver's digit: patch ignored
        source's digit: patch carries the output token
        source's RAW SUM wrapped by the RECEIVER's base: patch carries an abstract sum, re-encoded by the
        receiver (the signature of "sum first, base-specific wrap later")
     Also runs with the SAME integers in both bases: if an abstract number lived here, the answer stays right.
  B. Interchange of digit state (same base). Swap the residual at q_k between problems, measuring whether output
     follows the source's digit sum, per stage. Finds where digit state becomes sufficient for output.
  C. Base-swap on identical test-line tokens: only the few-shot context differs. Patch neurons between runs to find
     which ones carry the base-dependent decision (the operational definition of a translator neuron).
  D. Base-vector steering. Mean difference between base contexts at q_k, added to a run: can it switch modulus?
  E. Distributed alignment search: learn a small subspace whose interchange changes carry (or sum) as a high-level
     schoolbook model predicts, then test it in other bases.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, DATA, DEVICE, DIGITS, Batch, Model, Problem, iter_batches, sample_problems, sample_shots, save_json, schoolbook, to_base

model = Model()
digit_tok = torch.tensor([model.vocab[c] for c in DIGITS], device=DEVICE)
res: dict = {}
STAGES = ["resid_pre0", "resid_mid0", "resid_post0", "resid_mid1", "resid_post1"]


def q_index(bt: Batch, k: int) -> torch.Tensor:
    return torch.tensor(bt.p_eq + k, device=DEVICE)


@torch.no_grad()
def run_patch(receivers: list[Problem], sources: list[Problem], stage: str, k: int) -> np.ndarray:
    """Copy the source's activation at (stage, q_k) into the receiver; return the receiver's prediction at q_k."""
    preds = []
    for i in range(0, len(receivers), 256):
        r_chunk, s_chunk = receivers[i : i + 256], sources[i : i + 256]
        rb, sb = Batch(model, r_chunk), Batch(model, s_chunk)
        _, cache = model.forward(sb.ids, capture={stage})
        idx = torch.arange(len(r_chunk), device=DEVICE)
        src_act = cache[stage][idx, q_index(sb, k)]
        qr = q_index(rb, k)

        def hook(t: torch.Tensor) -> torch.Tensor:
            t = t.clone()
            t[idx, qr] = src_act
            return t

        logits, _ = model.forward(rb.ids, hooks={stage: hook})
        preds.append(logits[idx, qr].argmax(-1).cpu().numpy())
    return np.concatenate(preds)


def tok(d: int) -> int:
    return model.vocab[DIGITS[d]]


def categorise(preds: np.ndarray, receivers: list[Problem], sources: list[Problem], k: int) -> dict:
    """Sort patched outputs. Only trials where the three candidates are all different are scored."""
    n = {"receiver_digit": 0, "source_digit": 0, "source_sum_rewrapped": 0, "other": 0, "trials": 0}
    for p, r, s in zip(preds, receivers, sources):
        vr = schoolbook(r.a, r.b, r.base)
        vs = schoolbook(s.a, s.b, s.base)
        c_rec = tok(vr["o"][k])
        c_src = tok(vs["o"][k])
        c_wrap = tok(vs["s"][k] % r.base)
        if len({c_rec, c_src, c_wrap}) < 3:
            continue
        n["trials"] += 1
        if p == c_rec:
            n["receiver_digit"] += 1
        elif p == c_src:
            n["source_digit"] += 1
        elif p == c_wrap:
            n["source_sum_rewrapped"] += 1
        else:
            n["other"] += 1
    t = max(n["trials"], 1)
    return {**{a: b / t for a, b in n.items() if a != "trials"}, "trials": n["trials"]}


def matched(n: int, ba: int, bb: int, seed: int, same_ints: bool, k: int) -> tuple[list[Problem], list[Problem]]:
    """Receiver in base bb, source in base ba. Contexts are independent, operands kept long enough for digit k."""
    rng = random.Random(seed)
    R, S = [], []
    while len(R) < n:
        ra, rb_ = rng.randint(0, 999), rng.randint(0, 999)
        sa, sb_ = (ra, rb_) if same_ints else (rng.randint(0, 999), rng.randint(0, 999))
        if ra + rb_ < bb**k or sa + sb_ < ba**k or ra + rb_ < ba**k:
            continue
        R.append(Problem(ra, rb_, bb, sample_shots(rng, (ra, rb_))))
        S.append(Problem(sa, sb_, ba, sample_shots(rng, (sa, sb_))))
    return R, S


# ------------------------------------------------------------------------------------------------ A
res["A"] = {}
N = 3000
for stage in STAGES[1:]:
    res["A"][stage] = {}
    for k in [0, 1]:
        for same in [False, True]:
            mat = {}
            for ba in BASES:
                for bb in BASES:
                    R, S = matched(N, ba, bb, seed=ba * 100 + bb + k, same_ints=same, k=k)
                    preds = run_patch(R, S, stage, k)
                    if same:
                        # same integers: is the receiver's answer preserved?
                        rec = np.array([tok(schoolbook(r.a, r.b, bb)["o"][k]) for r in R])
                        src = np.array([tok(schoolbook(s.a, s.b, ba)["o"][k]) for s in S])
                        diff = rec != src
                        mat[f"{ba}->{bb}"] = {
                            "keeps_receiver_answer": float((preds[diff] == rec[diff]).mean()) if diff.any() else None,
                            "takes_source_digit": float((preds[diff] == src[diff]).mean()) if diff.any() else None,
                            "trials": int(diff.sum()),
                        }
                    else:
                        mat[f"{ba}->{bb}"] = categorise(preds, R, S, k)
            key = f"k{k}/{'same_integers' if same else 'different_integers'}"
            res["A"][stage][key] = mat
            if same:
                print(stage, key, {x: (round(y["keeps_receiver_answer"], 2) if y["keeps_receiver_answer"] is not None else None) for x, y in mat.items() if x[:2] != x[-2:]}, flush=True)
            else:
                print(stage, key, {x: (round(y["receiver_digit"], 2), round(y["source_digit"], 2), round(y["source_sum_rewrapped"], 2)) for x, y in mat.items()}, flush=True)
    save_json("e11_causal", res)

# ------------------------------------------------------------------------------------------------ C
# Base-swap: the test line has tokens valid in both bases. Only context differs.
@torch.no_grad()
def predict_digit(ps: list[Problem], k: int, prefixes: list[str] | None = None, hooks_fn=None) -> np.ndarray:
    out = []
    for i in range(0, len(ps), 256):
        chunk = ps[i : i + 256]
        pre = prefixes[i : i + 256] if prefixes else [""] * len(chunk)
        seqs = [[model.vocab[c] for c in p.prompt + x] for p, x in zip(chunk, pre)]
        n = max(map(len, seqs))
        ids = torch.full((len(seqs), n), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs):
            ids[j, : len(s)] = torch.tensor(s)
        ids = ids.to(DEVICE)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
        hooks = hooks_fn(ids, q, i, i + len(chunk)) if hooks_fn else None
        logits, _ = model.forward(ids, hooks=hooks)
        out.append(logits[torch.arange(len(chunk), device=DEVICE), q].argmax(-1).cpu().numpy())
    return np.concatenate(out)


res["C"] = {}
for b_low, b_high in [(8, 10), (10, 16), (8, 16)]:
    rng = random.Random(b_low * b_high)
    # test-line tokens are valid in the smaller base. Choose units digits whose sum wraps differently.
    ps_low, ps_high = [], []
    while len(ps_low) < 3000:
        ax, bx = rng.randint(0, b_low - 1), rng.randint(0, b_low - 1)
        if ax + bx < b_low:  # both bases agree unless the smaller base wraps
            continue
        hi_a = rng.randint(0, 40)
        hi_b = rng.randint(0, 40)
        a_int, b_int = hi_a * b_low + ax, hi_b * b_low + bx
        shots = sample_shots(rng, (a_int, b_int))
        ps_low.append(Problem(a_int, b_int, b_low, shots))
        # identical test-line tokens, context from the larger base
        ps_high.append(Problem(a_int, b_int, b_low, shots, ctx_base=b_high))
    pred_low = predict_digit(ps_low, 0)
    pred_high = predict_digit(ps_high, 0)
    own_low = np.array([tok((p.a % b_low + p.b % b_low) % b_low) for p in ps_low])
    own_high = np.array([tok((p.a % b_low + p.b % b_low) % b_high) for p in ps_low])
    follow_low = float((pred_low == own_low).mean())
    follow_high = float((pred_high == own_high).mean())
    high_says_low = float((pred_high == own_low).mean())
    out = {"ctx_low_follows_low_rule": follow_low, "ctx_high_follows_high_rule": follow_high, "ctx_high_still_uses_low_rule": high_says_low}
    print("C", b_low, b_high, out, flush=True)

    # Which neurons carry this? For trials where the swap flips the digit, patch one neuron's value from the low-context
    # run into the high-context run, at '=', and count how often the digit flips to the low rule.
    flip = np.where((pred_low == own_low) & (pred_high == own_high))[0]
    flip = flip[:800]
    out["n_flip_trials"] = int(len(flip))
    if len(flip) >= 50:
        P_low = [ps_low[i] for i in flip]
        P_high = [ps_high[i] for i in flip]
        target_low = own_low[flip]
        target_high = own_high[flip]
        with torch.no_grad():
            caches = []
            for i in range(0, len(flip), 256):
                bt = Batch(model, P_low[i : i + 256], with_answer=False)
                _, c = model.forward(bt.ids, capture={"mlp_post0", "mlp_post1", "z0", "z1", "resid_post0", "resid_mid1"})
                idx = torch.arange(bt.ids.shape[0], device=DEVICE)
                caches.append({k2: v[idx, q_index(bt, 0)].clone() for k2, v in c.items()})
        src = {k2: torch.cat([c[k2] for c in caches]) for k2 in caches[0]}

        def patcher(name: str, comp: int | None):
            def factory(ids, q, i0, i1):
                idx = torch.arange(ids.shape[0], device=DEVICE)
                val = src[name][i0:i1]

                def fn(t: torch.Tensor) -> torch.Tensor:
                    t = t.clone()
                    if comp is None:
                        t[idx, q] = val
                    else:
                        t[idx, q, comp] = val[:, comp]
                    return t

                return {name: fn}

            return factory

        # whole-stage patches first
        stage_flip = {}
        for name in ["resid_post0", "resid_mid1", "mlp_post0", "mlp_post1", "z1", "z0"]:
            pr = predict_digit(P_high, 0, hooks_fn=patcher(name, None))
            stage_flip[name] = {"to_low_rule": float((pr == target_low).mean()), "keeps_high_rule": float((pr == target_high).mean())}
        out["stage_flip"] = stage_flip
        print("  stage flip", {a: round(b["to_low_rule"], 3) for a, b in stage_flip.items()}, flush=True)
        # each neuron alone
        single = {}
        for L in range(2):
            flips = np.zeros(256)
            for n in range(256):
                pr = predict_digit(P_high, 0, hooks_fn=patcher(f"mlp_post{L}", n))
                flips[n] = (pr == target_low).mean()
            single[L] = flips
            top = np.argsort(-flips)[:8]
            print(f"  L{L} single-neuron flip rate top:", [(int(t), round(float(flips[t]), 3)) for t in top], flush=True)
        out["single_neuron_flip"] = single
        # greedy minimal set in L1: add neurons (ordered by how much the two contexts differ) until it flips
        with torch.no_grad():
            hi_c = []
            for i in range(0, len(flip), 256):
                bt = Batch(model, P_high[i : i + 256], with_answer=False)
                _, c = model.forward(bt.ids, capture={"mlp_post1"})
                idx = torch.arange(bt.ids.shape[0], device=DEVICE)
                hi_c.append(c["mlp_post1"][idx, q_index(bt, 0)])
            hi = torch.cat(hi_c)
        delta = (src["mlp_post1"] - hi).abs().mean(0)
        order = delta.argsort(descending=True).tolist()
        curve = {}
        for size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            comps = order[:size]

            def factory(ids, q, i0, i1, comps=comps):
                idx = torch.arange(ids.shape[0], device=DEVICE)
                val = src["mlp_post1"][i0:i1]

                def fn(t: torch.Tensor) -> torch.Tensor:
                    t = t.clone()
                    for c_ in comps:
                        t[idx, q, c_] = val[:, c_]
                    return t

                return {"mlp_post1": fn}

            pr = predict_digit(P_high, 0, hooks_fn=factory)
            curve[size] = float((pr == target_low).mean())
        out["cumulative_L1_flip"] = curve
        print("  cumulative L1 neurons:", {a: round(b, 3) for a, b in curve.items()}, flush=True)
    res["C"][f"{b_low}vs{b_high}"] = out
save_json("e11_causal", res)


# ------------------------------------------------------------------------------------------------ D
@torch.no_grad()
def steer_test(src_base: int, dst_base: int, stage: str, scale: float, vec: torch.Tensor, n: int = 2000) -> dict:
    """Add a base-difference vector at q_0 and score which modulus rule the digit now follows."""
    ps = []
    rng = random.Random(src_base + dst_base)
    lim = min(src_base, dst_base)
    while len(ps) < n:
        ax, bx = rng.randint(0, lim - 1), rng.randint(0, lim - 1)
        if ax + bx < lim:
            continue
        a_int, b_int = rng.randint(0, 40) * src_base + ax, rng.randint(0, 40) * src_base + bx
        ps.append(Problem(a_int, b_int, src_base, sample_shots(rng, (a_int, b_int))))

    def f(ids, q, i0, i1):
        idx = torch.arange(ids.shape[0], device=DEVICE)

        def fn(t: torch.Tensor) -> torch.Tensor:
            t = t.clone()
            t[idx, q] = t[idx, q] + scale * vec
            return t

        return {stage: fn}

    pr = predict_digit(ps, 0, hooks_fn=f)
    own = np.array([tok((p.a % src_base + p.b % src_base) % src_base) for p in ps])
    dst = np.array([tok((p.a % src_base + p.b % src_base) % dst_base) for p in ps])
    d = own != dst
    return {"own_rule": float((pr[d] == own[d]).mean()), "dst_rule": float((pr[d] == dst[d]).mean()), "n": int(d.sum())}


means = {}
with torch.no_grad():
    for b in BASES:
        ps = sample_problems(3000, b, seed=61)
        acc = {s: [] for s in STAGES}
        for chunk in iter_batches(ps, 256):
            bt = Batch(model, chunk, with_answer=False)
            _, c = model.forward(bt.ids, capture=set(STAGES))
            idx = torch.arange(len(chunk), device=DEVICE)
            for s in STAGES:
                acc[s].append(c[s][idx, q_index(bt, 0)])
        means[b] = {s: torch.cat(v).mean(0) for s, v in acc.items()}
res["D"] = {}
for src_base, dst_base in [(8, 10), (10, 8), (10, 16), (16, 10), (8, 16), (16, 8)]:
    for stage in ["resid_post0", "resid_mid1"]:
        vec = means[dst_base][stage] - means[src_base][stage]
        row = {}
        for scale in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
            row[scale] = steer_test(src_base, dst_base, stage, scale, vec)
        res["D"][f"{src_base}->{dst_base}/{stage}"] = row
        print("D", src_base, dst_base, stage, {a: (round(b["own_rule"], 2), round(b["dst_rule"], 2)) for a, b in row.items()}, flush=True)
save_json("e11_causal", res)


# ------------------------------------------------------------------------------------------------ E: DAS
def schoolbook_cf(r: Problem, s: Problem, k: int, var: str) -> int:
    """High-level counterfactual digit: receiver's digit with one variable taken from the source."""
    vr = schoolbook(r.a, r.b, r.base)
    vs = schoolbook(s.a, s.b, s.base)
    if var == "carry":
        return (vr["a"][k] + vr["b"][k] + vs["c_in"][k]) % r.base
    if var == "output":
        return vs["s"][k] % r.base
    raise ValueError(var)


def das_data(bx: int, by: int, k: int, var: str, n: int, seed: int) -> dict:
    """Pairs where the high-level swap changes the answer. Receiver in by, source in bx."""
    rng = random.Random(seed)
    R, S, y = [], [], []
    while len(R) < n:
        ra, rb_, sa, sb_ = [rng.randint(0, 999) for _ in range(4)]
        if ra + rb_ < by**k or sa + sb_ < bx**k:
            continue
        r = Problem(ra, rb_, by, sample_shots(rng, (ra, rb_)))
        s = Problem(sa, sb_, bx, sample_shots(rng, (sa, sb_)))
        target = schoolbook_cf(r, s, k, var)
        if target == schoolbook(ra, rb_, by)["o"][k]:
            continue
        R.append(r)
        S.append(s)
        y.append(tok(target))
    return {"R": R, "S": S, "y": torch.tensor(y, device=DEVICE)}


@torch.no_grad()
def site_activations(ps: list[Problem], stage: str, k: int) -> torch.Tensor:
    out = []
    for i in range(0, len(ps), 256):
        bt = Batch(model, ps[i : i + 256])
        _, c = model.forward(bt.ids, capture={stage})
        idx = torch.arange(bt.ids.shape[0], device=DEVICE)
        out.append(c[stage][idx, q_index(bt, k)])
    return torch.cat(out)


def forward_with_swap(R: list[Problem], src: torch.Tensor, Q: torch.Tensor, stage: str, k: int) -> torch.Tensor:
    """Replace the subspace projection at the site with the source's; return logits at q_k."""
    bt = Batch(model, R)
    idx = torch.arange(len(R), device=DEVICE)
    qr = q_index(bt, k)

    def hook(t: torch.Tensor) -> torch.Tensor:
        t = t.clone()
        v = t[idx, qr]
        t[idx, qr] = v + (src - v) @ (Q @ Q.T)
        return t

    logits, _ = model.forward(bt.ids, hooks={stage: hook})
    return logits[idx, qr]


def train_das(base: int, k: int, var: str, stage: str, dim: int, steps: int = 400) -> tuple[torch.Tensor, float]:
    """Learn an orthonormal subspace with gradient descent on interchange accuracy."""
    tr = das_data(base, base, k, var, 3000, seed=base * 7 + k)
    src_act = site_activations(tr["S"], stage, k)
    torch.manual_seed(base + k + dim)
    raw = torch.nn.Parameter(torch.randn(64, dim, device=DEVICE, dtype=torch.float32))
    opt = torch.optim.Adam([raw], lr=5e-3)
    n = len(tr["R"])
    for step in range(steps):
        idx = np.random.default_rng(step).choice(n, 256, replace=False)
        Q, _ = torch.linalg.qr(raw)
        R = [tr["R"][i] for i in idx]
        logits = forward_with_swap(R, src_act[idx], Q, stage, k)
        loss = torch.nn.functional.cross_entropy(logits, tr["y"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return torch.linalg.qr(raw.detach())[0], float(loss.item())


@torch.no_grad()
def iia(Q: torch.Tensor, bx: int, by: int, k: int, var: str, stage: str) -> float:
    """Interchange intervention accuracy on fresh pairs."""
    te = das_data(bx, by, k, var, 1500, seed=9999 + bx * 13 + by + k)
    src = site_activations(te["S"], stage, k)
    ok = 0
    for i in range(0, len(te["R"]), 256):
        lg = forward_with_swap(te["R"][i : i + 256], src[i : i + 256], Q, stage, k)
        ok += (lg.argmax(-1) == te["y"][i : i + 256]).sum().item()
    return ok / len(te["R"])


res["E"] = {}
for var, ks in [("carry", [1]), ("output", [0, 1])]:
    for k in ks:
        for stage in ["resid_post0", "resid_mid1"]:
            for dim in [1, 2, 4, 8, 16]:
                row = {}
                subspaces = {}
                for b in BASES:
                    Q, loss = train_das(b, k, var, stage, dim)
                    subspaces[b] = Q
                mat = np.zeros((4, 4))
                align = np.zeros((4, 4))
                for i, bx in enumerate(BASES):
                    for j, by in enumerate(BASES):
                        # subspace trained in bx, tested on a within-base swap in by
                        mat[i, j] = iia(subspaces[bx], by, by, k, var, stage)
                        # principal-angle overlap between subspaces (1 = same subspace)
                        align[i, j] = float((subspaces[bx].T @ subspaces[by]).norm().item() ** 2 / dim)
                # random-subspace control
                ctrl = np.zeros(4)
                for j, by in enumerate(BASES):
                    Q = torch.linalg.qr(torch.randn(64, dim, device=DEVICE))[0]
                    ctrl[j] = iia(Q, by, by, k, var, stage)
                # chance overlap between random subspaces is dim/64
                key = f"{var}/k{k}/{stage}/d{dim}"
                res["E"][key] = {"iia_transfer": mat, "alignment": align, "random_subspace_iia": ctrl, "random_alignment": dim / 64}
                print("E", key, "\n", np.round(mat, 2), "\n  align", np.round(align[np.triu_indices(4, 1)], 2), " rand", round(dim / 64, 2), " ctrl", np.round(ctrl, 2), flush=True)
                save_json("e11_causal", res)
