"""F03 (forward passes): how is the base identified, and where does the signal travel?

Test line uses digits 0-7 with units digits summing to >= 8, so rules 8, 10 and 16 give different first digits.
Context is octal. Editing context digit tokens keeps prompts aligned, so runs can be patched position by position.
A. Single-token curve: replace one context digit with d = 0..F. Which rule does the first digit follow?
B. Dose and location: number of edits; edit in operand vs answer of a shot.
C. Region patching at layer 0's output, between clean octal and edited runs, in both directions.
D. Base 10 and base 16 contexts: remove evidence (no 8, 9, A-F anywhere) to see the default.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}; N = 1500


def rule(a0, b0, t):
    return model.vocab[DIGITS[(a0 + b0) % t]]


def make(n, seed, ctx_base=8):
    rng = random.Random(seed); out = []
    while len(out) < n:
        ax, bx = rng.randint(0, 7), rng.randint(0, 7)
        if ax + bx < 8:
            continue
        a, b = rng.randint(0, 63) * 8 + ax, rng.randint(0, 63) * 8 + bx
        lines = [render(*s, ctx_base) for s in sample_shots(rng, (a, b))]
        ctx = "\n".join(lines)
        test = f"{to_base(a, 8)}+{to_base(b, 8)}="
        out.append(dict(ctx=ctx, test=test, a0=ax, b0=bx))
    return out


def ids_of(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]
    L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)


def predict(strs, hooks_fn=None, bs=512):
    out = []
    for i in range(0, len(strs), bs):
        x, q = ids_of(strs[i:i + bs])
        h = hooks_fn(i, i + bs, x) if hooks_fn else None
        lg, _ = model.forward(x, hooks=h)
        out.append(lg[torch.arange(len(x), device=DEVICE), q].argmax(-1).cpu().numpy())
    return np.concatenate(out)


def score(pred, P):
    return {t: float(np.mean([p == rule(x["a0"], x["b0"], t) for p, x in zip(pred, P)])) for t in (8, 10, 16)}


def digit_positions(ctx):
    pos, seg = [], None
    i = 0
    for j, ch in enumerate(ctx):
        if ch in DIGITS:
            pos.append(j)
    return pos


def edit(ctx, d, rng, n=1, where=None):
    c = list(ctx)
    pos = []
    in_ans = False
    for j, ch in enumerate(c):
        if ch == "=": in_ans = True
        elif ch == "\n": in_ans = False
        elif ch != "+" and (where is None or (where == "answer") == in_ans):
            pos.append(j)
    for j in rng.sample(pos, min(n, len(pos))):
        c[j] = d
    return "".join(c)


P = make(N, 1)
clean = [x["ctx"] + "\n" + x["test"] for x in P]
res["clean_octal"] = score(predict(clean), P)
print("clean octal context", res["clean_octal"], flush=True)
res["A"] = {}
for d in DIGITS:
    rng = random.Random(ord(d))
    res["A"][d] = score(predict([edit(x["ctx"], d, rng) + "\n" + x["test"] for x in P]), P)
    print("A one token", d, {k: round(v, 3) for k, v in res["A"][d].items()}, flush=True)
res["B"] = {}
for d in "9AF":
    for n in (1, 2, 4, 8):
        rng = random.Random(n)
        res["B"][f"{d}/n{n}"] = score(predict([edit(x["ctx"], d, rng, n) + "\n" + x["test"] for x in P]), P)
    for wh in ("operand", "answer"):
        rng = random.Random(7)
        res["B"][f"{d}/{wh}"] = score(predict([edit(x["ctx"], d, rng, 1, wh) + "\n" + x["test"] for x in P]), P)
    print("B", d, {k.split('/')[1]: {t: round(v, 2) for t, v in res['B'][k].items()} for k in res["B"] if k.startswith(d)}, flush=True)

res["D"] = {}
for t, name in ((10, "decimal"), (16, "hex")):
    Q = make(N, 2, ctx_base=t)
    res["D"][f"{name} context"] = score(predict([x["ctx"] + "\n" + x["test"] for x in Q]), Q)
    # same integers rendered in that base, but dropping shots containing the evidence
    keep = []
    rng = random.Random(3)
    while len(keep) < N // 2:
        ax, bx = rng.randint(0, 7), rng.randint(0, 7)
        if ax + bx < 8: continue
        lines = []
        while len(lines) < 5:
            x, y = rng.randint(0, 999), rng.randint(0, 999)
            l = render(x, y, t)
            if set(l) <= set("01234567+="): lines.append(l)
        a, b = rng.randint(0, 63) * 8 + ax, rng.randint(0, 63) * 8 + bx
        keep.append(dict(ctx="\n".join(lines), test=f"{to_base(a, 8)}+{to_base(b, 8)}=", a0=ax, b0=bx))
    res["D"][f"{name} context, only digits 0-7 shown"] = score(predict([x["ctx"] + "\n" + x["test"] for x in keep]), keep)
print("D", res["D"], flush=True)


# ---- C. region patching on single-token pairs
def regions(s):
    ctx_len = s.rfind("\n") + 1
    eq = len(s) - 1
    labels = {"ctx_newline": [], "ctx_edit_line": [], "ctx_other": [], "test_operands": [], "query": [eq]}
    for i in range(ctx_len):
        if s[i] == "\n": labels["ctx_newline"].append(i)
    for i in range(ctx_len, eq): labels["test_operands"].append(i)
    return labels


res["C"] = {}
for d, target in (("9", 10), ("A", 16), ("F", 16)):
    rng = random.Random(11)
    edited = [edit(x["ctx"], d, rng) + "\n" + x["test"] for x in P]
    pc, pe = predict(clean), predict(edited)
    r8 = np.array([rule(x["a0"], x["b0"], 8) for x in P]); rt = np.array([rule(x["a0"], x["b0"], target) for x in P])
    flip = np.where((pc == r8) & (pe == rt) & (r8 != rt))[0][:800]
    if len(flip) < 50:
        print("C skip", d, len(flip)); continue
    C = [clean[i] for i in flip]; E = [edited[i] for i in flip]
    rec = {}
    for layer in ("resid_pre1", "resid_pre0"):
        # capture both
        def cap(S):
            out = []
            for i in range(0, len(S), 256):
                x, q = ids_of(S[i:i + 256]); _, c = model.forward(x, capture={layer}); out.append(c[layer])
            return out
        CC, EE = cap(C), cap(E)
        lab = [regions(s) for s in C]
        names = ["ctx_newline", "test_operands", "query", "ctx_all", "all_test", "everything"]
        for nm in names:
            for direction in ("clean_gets_edited", "edited_gets_clean"):
                S = C if direction == "clean_gets_edited" else E
                src = EE if direction == "clean_gets_edited" else CC
                want = rt[flip] if direction == "clean_gets_edited" else r8[flip]
                preds = []
                for bi, i in enumerate(range(0, len(S), 256)):
                    x, q = ids_of(S[i:i + 256])
                    mask = torch.zeros(x.shape, dtype=torch.bool, device=DEVICE)
                    for row in range(x.shape[0]):
                        l = lab[i + row]; ctx_end = S[i + row].rfind("\n") + 1; eq = len(S[i + row]) - 1
                        if nm == "ctx_all": mask[row, :ctx_end] = True
                        elif nm == "all_test": mask[row, ctx_end:eq + 1] = True
                        elif nm == "everything": mask[row, :eq + 1] = True
                        else: mask[row, l[nm]] = True
                    def hook(t, m=mask, s=src[bi]):
                        t = t.clone(); t[m] = s[m]; return t
                    lg, _ = model.forward(x, hooks={layer: hook})
                    preds.append(lg[torch.arange(x.shape[0], device=DEVICE), q].argmax(-1).cpu().numpy())
                pr = np.concatenate(preds)
                rec[f"{layer}/{nm}/{direction}"] = float((pr == want).mean())
    res["C"][d] = {"n": len(flip), "patches": rec}
    print("C", d, "n", len(flip), {k: round(v, 2) for k, v in rec.items() if "resid_pre1" in k}, flush=True)
save_json("f03_base_switch", res)
print("DONE")
