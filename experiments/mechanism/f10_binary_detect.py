"""F10 (forward passes): what separates binary from octal? And is base an ordered axis?

Test line: digits 0/1 only, units 1+1 (rule 2 gives 0, rule 8 gives 2).
A. Context in a true base B = 2..7: which rule does the model follow?
B. Binary context with a fraction p of digits replaced by digit d.
C. Test-line evidence: binary context, test operands contain one digit d.
D. Which L0 head carries it: base-4 context (often octal-like) vs binary context; attention of each head by digit value.
E. Context lengths: shots of short binary numbers vs long, since binary strings are long.
F. Steering 8 -> 16 on trials with s >= 10, where rules 8, 10 and 16 all differ.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, DIGITS, Model, render, sample_shots, save_json, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}; tok = np.array([model.vocab[c] for c in DIGITS])
def tens(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]
    L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
    return x.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
def predict(strs, bs=256):
    out = []
    for i in range(0, len(strs), bs):
        x, q = tens(strs[i:i + bs]); lg, _ = model.forward(x)
        out.append(lg[torch.arange(len(x), device=DEVICE), q].argmax(-1).cpu().numpy())
    return np.concatenate(out)
def test_line(rng, digits_len=(3, 8)):
    a = "1" + "".join(rng.choice("01") for _ in range(rng.randint(*digits_len) - 2)) + "1"
    b = "1" + "".join(rng.choice("01") for _ in range(rng.randint(*digits_len) - 2)) + "1"
    return a, b
def rules(pr):
    return {"rule2 (0)": float((pr == model.vocab["0"]).mean()), "rule>=3 (2)": float((pr == model.vocab["2"]).mean())}
N = 1200
res["A"] = {}
for B in range(2, 9):
    rng = random.Random(B); S = []
    for _ in range(N):
        a, b = test_line(rng)
        S.append("\n".join(render(*sh, B) for sh in sample_shots(rng, (-1, -1))) + f"\n{a}+{b}=")
    res["A"][B] = rules(predict(S)); print("A ctx base", B, res["A"][B], flush=True)
res["B"] = {}
for d in "234567":
    for p in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0):
        rng = random.Random(int(d) * 100 + int(p * 100)); S = []
        for _ in range(N // 2):
            a, b = test_line(rng)
            ctx = list("\n".join(render(*sh, 2) for sh in sample_shots(rng, (-1, -1))))
            for j, ch in enumerate(ctx):
                if ch in "01" and rng.random() < p: ctx[j] = d
            S.append("".join(ctx) + f"\n{a}+{b}=")
        res["B"][f"{d}/{p}"] = rules(predict(S))
    print("B digit", d, {p: round(res["B"][f"{d}/{p}"]["rule2 (0)"], 2) for p in (0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0)}, flush=True)
res["C"] = {}
for d in "2375":
    rng = random.Random(ord(d)); S = []
    for _ in range(N):
        a, b = test_line(rng); a = a[0] + d + a[2:] if len(a) > 2 else a
        S.append("\n".join(render(*sh, 2) for sh in sample_shots(rng, (-1, -1))) + f"\n{a}+{b}=")
    res["C"][d] = rules(predict(S)); print("C test-line digit", d, res["C"][d], flush=True)
# E: only short binary numbers in context; or digits 0/1 with octal-length numbers
res["E"] = {}
for name, hi in (("short (<8)", 7), ("mid (<64)", 63), ("full", 999)):
    rng = random.Random(hi); S = []
    for _ in range(N):
        a, b = test_line(rng)
        S.append("\n".join(render(*sh, 2) for sh in sample_shots(rng, (-1, -1), hi=hi)) + f"\n{a}+{b}=")
    res["E"][name] = rules(predict(S)); print("E binary ctx numbers", name, res["E"][name], flush=True)
# octal context using only digits 0,1
rng = random.Random(5); S = []
while len(S) < N:
    lines = []
    while len(lines) < 5:
        l = render(rng.randint(0, 999), rng.randint(0, 999), 8)
        if set(l) <= set("01+="): lines.append(l)
    a, b = test_line(rng); S.append("\n".join(lines) + f"\n{a}+{b}=")
res["E"]["octal ctx, only digits 0/1 shown"] = rules(predict(S)); print("E", res["E"]["octal ctx, only digits 0/1 shown"], flush=True)
# D attention by digit value in L0H* from query
res["D"] = {}
for B in (2, 3, 4, 5, 6, 7, 8):
    rng = random.Random(B * 3); S = []
    for _ in range(600):
        a, b = test_line(rng); S.append("\n".join(render(*sh, B) for sh in sample_shots(rng, (-1, -1))) + f"\n{a}+{b}=")
    tot = np.zeros((4, 8)); cnt = np.zeros(8); other = np.zeros(4); n = 0
    for i in range(0, len(S), 128):
        ch = S[i:i + 128]; x, q = tens(ch); _, c = model.forward(x, capture={"pattern0"})
        pat = c["pattern0"]
        for j, s in enumerate(ch):
            row = pat[j, :, q[j]].cpu().numpy(); start = s.rfind("\n") + 1
            for t in range(start):
                if s[t] in DIGITS[:8]:
                    tot[:, DIGITS.index(s[t])] += row[:, t]; cnt[DIGITS.index(s[t])] += 1
            other += row[:, start:].sum(1); n += 1
    res["D"][B] = {"per_token": tot / np.maximum(cnt, 1), "mass_on_test_line": other / n}
    print("D ctx base", B, "L0H1 per token x1000", np.round(res["D"][B]["per_token"][1] * 1000, 1).tolist(), "test-line mass by head", np.round(other / n, 2).tolist(), flush=True)
# F steering 8 -> 16 with s >= 10
rng = random.Random(77); C, E, S_ = [], [], []
while len(C) < 1200:
    ax, bx = rng.randint(3, 7), rng.randint(3, 7)
    if ax + bx < 10: continue
    a = rng.randint(1, 120) * 8 + ax; b = rng.randint(1, 120) * 8 + bx
    ctx = "\n".join(render(*sh, 8) for sh in sample_shots(rng, (a, b)))
    t = f"{to_base(a, 8)}+{to_base(b, 8)}="
    c2 = list(ctx); pos = [j for j, ch in enumerate(c2) if ch not in "+=\n"]
    for j in rng.sample(pos, 2): c2[j] = "A"
    C.append(ctx + "\n" + t); E.append("".join(c2) + "\n" + t); S_.append(ax + bx)
s = np.array(S_)
def masks(x, q, strs):
    idx = torch.arange(x.shape[1], device=DEVICE)[None]; st = torch.tensor([z.rfind("\n") + 1 for z in strs], device=DEVICE); pl = torch.tensor([z.rfind("+") for z in strs], device=DEVICE)
    return (idx >= st[:, None]) & (idx < q[:, None]) & (idx != pl[:, None]), idx == q[:, None]
dq, dop = [], []
for i in range(0, 600, 256):
    xc, q = tens(C[i:i + 256]); xe, _ = tens(E[i:i + 256]); op, qm = masks(xc, q, C[i:i + 256])
    _, cc = model.forward(xc, capture={"resid_pre1"}); _, ce = model.forward(xe, capture={"resid_pre1"})
    d = ce["resid_pre1"] - cc["resid_pre1"]; dq.append(d[qm]); dop.append(d[op])
vq, vop = torch.cat(dq).mean(0), torch.cat(dop).mean(0)
res["F"] = {}
for al in (0, .25, .5, .6, .7, .8, .9, 1, 1.25, 1.5):
    out = []
    for i in range(600, len(C), 256):
        x, q = tens(C[i:i + 256]); op, qm = masks(x, q, C[i:i + 256])
        def fn(t, op=op, qm=qm, al=al):
            t = t.clone(); t[qm] += al * vq; t[op] += al * vop; return t
        lg, _ = model.forward(x, hooks={"resid_pre1": fn}); out.append(lg[torch.arange(len(x), device=DEVICE), q].argmax(-1).cpu().numpy())
    pr = np.concatenate(out); sv = s[600:]
    res["F"][al] = {f"rule{r}": float((pr == tok[sv % r]).mean()) for r in (8, 10, 16)}
    print("F alpha", al, {k: round(v, 2) for k, v in res["F"][al].items()}, flush=True)
save_json("f10_binary_detect", res); print("DONE")
