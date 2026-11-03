"""E1: behavioural tests that separate the candidate mechanisms before opening the model.

Each test is built so that "convert to one internal number, then add" and "add digit by digit"
predict different outcomes:

  A. per-digit accuracy and error profile on the training distribution
  B. magnitude extrapolation: same digit LENGTH as training but values above 999 (never seen)
  C. bases never trained on (3..15): does arithmetic generalise, or snap to a trained base?
  D. context/operand conflicts: which base does the model actually use, and can it translate?
  E. number of few-shot examples (0..5): how much of the base comes from context?
  F. carry chains: accuracy as a function of carry-chain length
  G. input-sensitivity matrix: which input digits does the output at digit k depend on?
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import (
    BASES,
    DIGITS,
    Batch,
    Model,
    Problem,
    greedy,
    iter_batches,
    sample_problems,
    sample_shots,
    save_json,
    schoolbook,
    to_answer,
    to_base,
    wilson,
)

model = Model()
res: dict = {}
N = 4000


def acc(problems: list[Problem]) -> dict:
    """Exact-match accuracy with a Wilson interval."""
    out = greedy(model, problems)
    ok = sum(o == p.answer for o, p in zip(out, problems))
    p, lo, hi = wilson(ok, len(problems))
    return {"acc": p, "lo": lo, "hi": hi, "n": len(problems)}


def digit_acc(out: list[str], answers: list[str], kmax: int = 11) -> list[float | None]:
    """Accuracy of the k-th emitted digit, over problems whose true answer has a k-th digit."""
    r = []
    for k in range(kmax):
        pairs = [(o, a) for o, a in zip(out, answers) if len(a) > k]
        r.append(sum(len(o) > k and o[k] == a[k] for o, a in pairs) / len(pairs) if pairs else None)
    return r


# --------------------------------------------------------------------------------------------- A
res["A"] = {}
for b in BASES:
    ps = sample_problems(N, b, seed=11)
    out = greedy(model, ps)
    ans = [p.answer for p in ps]
    wrong = [(p, o) for p, o in zip(ps, out) if o != p.answer]
    # classify errors: first wrong digit, was it at a carry position? length error?
    kinds = {"carry_in_at_first_error": 0, "no_carry_at_first_error": 0, "length": 0}
    for p, o in wrong:
        sb = schoolbook(p.a, p.b, b)
        k = next((i for i, (x, y) in enumerate(zip(o, p.answer)) if x != y), None)
        if k is None:
            kinds["length"] += 1
        else:
            kinds["carry_in_at_first_error" if sb["c_in"][k] else "no_carry_at_first_error"] += 1
    res["A"][b] = {
        "exact": wilson(sum(o == a for o, a in zip(out, ans)), N),
        "digit_acc": digit_acc(out, ans),
        "error_kinds": kinds,
        "examples": [(p.a, p.b, to_base(p.a, b), to_base(p.b, b), p.answer, o) for p, o in wrong[:6]],
    }
    print("A", b, res["A"][b]["exact"], kinds)

# --------------------------------------------------------------------------------------------- B
# A digit-local algorithm does not care about magnitude beyond the digits it reads; a value-based one
# has never seen these integers. Hex and octal give values above 999 at digit lengths seen in training.
res["B"] = {}
ranges = {
    16: [("seen 256-999 (3 digits)", 256, 999), ("UNSEEN 1000-2047 (3 digits)", 1000, 2047), ("UNSEEN a 2048-3071, b 256-999", 2048, 3071)],
    8: [("seen 512-999 (4 digits)", 512, 999), ("UNSEEN 1000-2047 (4 digits)", 1000, 2047), ("UNSEEN a 2048-3071, b 512-999", 2048, 3071)],
    2: [("seen 512-999 (10 digits)", 512, 999), ("UNSEEN 1000-1023 (10 digits)", 1000, 1023)],
    10: [("seen 100-999 (3 digits)", 100, 999), ("UNSEEN 1000-2047 (4 digits: new length)", 1000, 2047)],
}
for b, rs in ranges.items():
    res["B"][b] = {}
    for name, lo, hi in rs:
        rng = random.Random(b + lo)
        ps = []
        while len(ps) < N:
            x, y = rng.randint(lo, hi), rng.randint(lo, hi)
            if lo == 2048:  # one operand beyond training; the other inside it, at the same digit length
                y = rng.randint(256 if b == 16 else 512, 999)
            # keep the sum's digit count inside what training covered, so length is not the variable
            if b in (8, 16) and x + y >= 4096:
                continue
            ps.append(Problem(x, y, b, sample_shots(rng, (x, y))))
        out = greedy(model, ps)
        ans = [p.answer for p in ps]
        res["B"][b][name] = {"exact": wilson(sum(o == a for o, a in zip(out, ans)), N), "digit_acc": digit_acc(out, ans, 5)}
        print("B", b, name, res["B"][b][name]["exact"][0], [round(x, 3) if x is not None else None for x in res["B"][b][name]["digit_acc"]])


# --------------------------------------------------------------------------------------------- C
def rule_answer(sa: str, sb: str, t: int) -> str | None:
    """The answer if the test-line digit strings are read and added in base t (None if invalid)."""
    try:
        return to_answer(int(sa, t) + int(sb, t), t)
    except ValueError:
        return None


res["C"] = {}
for u in [3, 4, 5, 6, 7, 9, 11, 12, 13, 14, 15]:
    rng = random.Random(u)
    ps = [Problem(*(lambda x, y: (x, y))(rng.randint(0, 999), rng.randint(0, 999)), u) for _ in range(N)]
    for p in ps:
        p.shots = sample_shots(rng, (p.a, p.b))
    out = greedy(model, ps)
    rules: dict = {}
    for t in [u, *BASES]:
        if t in rules:
            continue
        hits, valid = 0, 0
        first_hits, first_valid = 0, 0
        for p, o in zip(ps, out):
            r = rule_answer(to_base(p.a, u), to_base(p.b, u), t)
            if r is None:
                continue
            valid += 1
            hits += o == r
            first_valid += 1
            first_hits += o[:1] == r[:1]
        rules[t] = {
            "exact": hits / valid if valid else None,
            "first_digit": first_hits / first_valid if first_valid else None,
            "n_valid": valid,
        }
    # chance-level references: how often do two different rules agree on the first digit anyway?
    res["C"][u] = {"rules": rules, "digit_acc_true": digit_acc(out, [p.answer for p in ps], 8)}
    print("C", u, {t: (round(v["exact"], 3) if v["exact"] is not None else None, round(v["first_digit"], 3) if v["first_digit"] is not None else None) for t, v in rules.items()})

# Wrap-point: for the first digit, what does the model emit as a function of the raw sum a0+b0?
wrap: dict = {}
for u in [8, 9, 10, 11, 12, 16]:
    tab = np.zeros((31, 23), dtype=np.int64)
    rng = random.Random(1000 + u)
    ps = []
    for _ in range(N * 2):
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        ps.append(Problem(a, b, u, sample_shots(rng, (a, b))))
    out = greedy(model, ps, max_new=1)
    for p, o in zip(ps, out):
        s = p.a % u + p.b % u
        if o:
            tab[s, model.vocab[o[0]]] += 1
    wrap[u] = {
        int(s): {model.itos[int(t)]: int(tab[s, t]) for t in np.argsort(-tab[s])[:3] if tab[s, t] > 0}
        for s in range(2 * u - 1)
    }
res["C_wrap"] = wrap

# --------------------------------------------------------------------------------------------- D
# Conflict: test-line tokens valid in several bases, context in one base. Whose rule does it follow?
res["D"] = {}
rng = random.Random(77)
# D1: digits 0-7 only (valid in bases 8, 10, 16): context alone decides.
ps_by_ctx: dict = {}
for c in [8, 10, 16]:
    ps = []
    while len(ps) < N:
        x, y = rng.randint(0, 511), rng.randint(0, 511)  # 3 octal digits -> tokens 0-7
        p = Problem(x, y, 8, sample_shots(rng, (x, y)), ctx_base=c)
        ps.append(p)
    out = greedy(model, ps)
    r = {}
    for t in [8, 10, 16]:
        r[t] = sum(o == rule_answer(to_base(p.a, 8), to_base(p.b, 8), t) for p, o in zip(ps, out)) / N
    res["D"].setdefault("tokens_0_7", {})[c] = r
    print("D tokens0-7 ctx", c, r)
# D2: hex-looking operands (contain A-F) with decimal/octal/binary context: operands force base?
for c in [2, 8, 10, 16]:
    ps = []
    while len(ps) < N:
        x, y = rng.randint(0, 999), rng.randint(0, 999)
        sx, sy = to_base(x, 16), to_base(y, 16)
        if not (set(sx + sy) & set("ABCDEF")):
            continue
        ps.append(Problem(x, y, 16, sample_shots(rng, (x, y)), ctx_base=c))
    out = greedy(model, ps)
    hex_rule = sum(o == to_answer(p.a + p.b, 16) for p, o in zip(ps, out)) / N
    # "translation": read the hex value, write it in the context's base
    translate = sum(o == to_answer(p.a + p.b, c) for p, o in zip(ps, out)) / N
    # digits emitted restricted to context alphabet?
    in_alpha = sum(all(ch in DIGITS[:c] for ch in o) for o in out) / N
    res["D"].setdefault("hex_operands", {})[c] = {"hex_rule": hex_rule, "translate_to_ctx": translate, "output_in_ctx_alphabet": in_alpha}
    print("D hex operands ctx", c, res["D"]["hex_operands"][c])
# D3: decimal operands with digits 8/9, under binary/octal context: which rule?
for c in [2, 8, 10, 16]:
    ps = []
    while len(ps) < N:
        x, y = rng.randint(0, 999), rng.randint(0, 999)
        if not (set(str(x) + str(y)) & set("89")):
            continue
        ps.append(Problem(x, y, 10, sample_shots(rng, (x, y)), ctx_base=c))
    out = greedy(model, ps)
    dec = sum(o == to_answer(p.a + p.b, 10) for p, o in zip(ps, out)) / N
    hexr = sum(o == rule_answer(str(p.a), str(p.b), 16) for p, o in zip(ps, out)) / N
    res["D"].setdefault("decimal_operands", {})[c] = {"decimal_rule": dec, "hex_rule": hexr, "translate_to_ctx": sum(o == to_answer(p.a + p.b, c) for p, o in zip(ps, out)) / N}
    print("D dec operands ctx", c, res["D"]["decimal_operands"][c])

# --------------------------------------------------------------------------------------------- E
res["E"] = {}
for b in BASES:
    res["E"][b] = {}
    for k in range(6):
        rng = random.Random(b * 100 + k)
        ps = []
        for _ in range(N):
            x, y = rng.randint(0, 999), rng.randint(0, 999)
            ps.append(Problem(x, y, b, sample_shots(rng, (x, y), k=k)))
        res["E"][b][k] = acc(ps)
    print("E", b, {k: round(v["acc"], 3) for k, v in res["E"][b].items()})

# --------------------------------------------------------------------------------------------- F
res["F"] = {}
for b in BASES:
    rng = random.Random(b + 5)
    bychain: dict = {}
    ps = sample_problems(20000, b, seed=50)
    out = greedy(model, ps)
    for p, o in zip(ps, out):
        sb = schoolbook(p.a, p.b, b)
        # longest run of consecutive carry-outs
        best, cur = 0, 0
        for c in sb["c_out"]:
            cur = cur + 1 if c else 0
            best = max(best, cur)
        d = bychain.setdefault(best, [0, 0])
        d[0] += o == p.answer
        d[1] += 1
    res["F"][b] = {k: wilson(*v) + (v[1],) for k, v in sorted(bychain.items())}
    print("F", b, {k: (round(v[0], 3), v[3]) for k, v in res["F"][b].items()})


# --------------------------------------------------------------------------------------------- G
@torch.no_grad()
def sensitivity(base: int, n: int = 2000, kmax: int = 2) -> dict:
    """How often changing one input digit changes the model's prediction at digit k.

    Teacher-forced on the ORIGINAL problem's answer prefix so that only the chosen token changes.
    Rows: query digit k. Columns: a_j, b_j (j = 0..kmax), and previously emitted digit o_{k-1}.
    Also reports the rate at which the corruption changes the TRUE schoolbook digit, as the reference.
    """
    rng = random.Random(base * 3)
    ps = [p for p in sample_problems(n * 3, base, seed=7) if len(to_base(p.a, base)) > kmax + (base == 2) and len(to_base(p.b, base)) > kmax + (base == 2)][:n]
    table_a = np.zeros((kmax, kmax + 1))
    table_b = np.zeros((kmax, kmax + 1))
    truth_a = np.zeros((kmax, kmax + 1))
    table_prev = np.zeros(kmax)
    for k in range(kmax):
        # baseline
        def predict(pp: list[Problem], prefixes: list[str]) -> np.ndarray:
            preds = []
            for chunk_idx in range(0, len(pp), 1024):
                chunk = pp[chunk_idx : chunk_idx + 1024]
                pref = prefixes[chunk_idx : chunk_idx + 1024]
                seqs = [[model.vocab[c] for c in q.prompt + x] for q, x in zip(chunk, pref)]
                m = max(map(len, seqs))
                ids = torch.full((len(seqs), m), model.pad, dtype=torch.long)
                for i, s in enumerate(seqs):
                    ids[i, : len(s)] = torch.tensor(s)
                logits, _ = model.forward(ids.to(model.device))
                idx = torch.tensor([len(s) - 1 for s in seqs], device=model.device)
                preds.append(logits[torch.arange(len(seqs)), idx].argmax(-1).cpu().numpy())
            return np.concatenate(preds)

        prefixes = [p.answer[:k] for p in ps]
        base_pred = predict(ps, prefixes)
        for j in range(kmax + 1):
            for which, tab in (("a", table_a), ("b", table_b)):
                # change digit j of one operand to a different valid digit, keeping digit length
                changed, truth = [], []
                for p in ps:
                    v = p.a if which == "a" else p.b
                    d = (v // base**j) % base
                    alt = rng.choice([x for x in range(base) if x != d])
                    nv = v + (alt - d) * base**j
                    while nv < base ** (len(to_base(v, base)) - 1):  # would shorten the number
                        alt = rng.choice([x for x in range(base) if x != d])
                        nv = v + (alt - d) * base**j
                        if base == 2:
                            nv = v
                            break
                    q = Problem(nv, p.b, base, p.shots) if which == "a" else Problem(p.a, nv, base, p.shots)
                    changed.append(q)
                    truth.append(schoolbook(q.a, q.b, base)["o"][k] != schoolbook(p.a, p.b, base)["o"][k])
                new_pred = predict(changed, prefixes)
                tab[k, j] = (new_pred != base_pred).mean()
                if which == "a":
                    truth_a[k, j] = np.mean(truth)
        if k > 0:
            # corrupt only the previously emitted digit; a schoolbook reader of a_{k-1}, b_{k-1}
            # needs it only if carry is inferred from what was just written.
            corrupt = []
            for p in ps:
                pref = p.answer[: k - 1]
                old = p.answer[k - 1]
                alt = rng.choice([c for c in DIGITS[:base] if c != old])
                corrupt.append(pref + alt)
            table_prev[k] = (predict(ps, corrupt) != base_pred).mean()
    return {"a": table_a, "b": table_b, "truth_a": truth_a, "prev_output": table_prev, "n": len(ps)}


res["G"] = {}
for b in BASES:
    res["G"][b] = sensitivity(b, kmax={2: 4, 8: 3, 10: 2, 16: 2}[b])
    print("G", b)
    print(np.round(res["G"][b]["a"], 3))
    print("truth", np.round(res["G"][b]["truth_a"], 3))
    print("prev", np.round(res["G"][b]["prev_output"], 3))
save_json("e01_behaviour", res)
