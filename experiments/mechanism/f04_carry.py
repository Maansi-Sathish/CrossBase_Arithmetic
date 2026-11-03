"""F04 (forward passes): where does the carry come from?

Teacher-forced prediction at digit k after corrupting exactly one source, on trials where sources disagree:
 W. Written digit: change o_{k-1} so that "infer carry from what I wrote" implies the opposite carry. Operands unchanged.
 O. Operand digit: change a_{k-1} so the true operand carry flips. The written answer is left as it was.
 D. Deep chain (k=2, digit 1 sums to base-1, so the carry into 2 depends on digit 0):
    D1 change a_0 so the carry from digit 0 flips (needs a look back two digits).
    D2 change the written o_1 so that it implies the other carry.
Each reports: stayed with the original digit, followed the alternative carry, something else.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, sample_shots, render, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}


def dig(n, b, k): return (n // b**k) % b
def setdig(n, b, k, d): return n + (d - dig(n, b, k)) * b**k
def carry_in(a, b_, base, k):
    M = base**k; return int(a % M + b_ % M >= M)


def predict(strs, bs=512):
    out = []
    for i in range(0, len(strs), bs):
        seqs = [[model.vocab[c] for c in s] for s in strs[i:i + bs]]
        L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        q = torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
        lg, _ = model.forward(x.to(DEVICE)); out.append(lg[torch.arange(len(seqs), device=DEVICE), q].argmax(-1).cpu().numpy())
    return np.concatenate(out)


def prompt(a, b_, base, shots, written):
    return "\n".join(render(*s, base) for s in shots) + f"\n{to_base(a, base)}+{to_base(b_, base)}=" + written


def tally(rows, base):
    """rows: (corrupted prompt, stay digit, alternative digit)."""
    pr = predict([r[0] for r in rows])
    stay = np.array([model.vocab[DIGITS[r[1]]] for r in rows]); alt = np.array([model.vocab[DIGITS[r[2]]] for r in rows])
    return {"n": len(rows), "stayed": float((pr == stay).mean()), "followed_alternative": float((pr == alt).mean()), "other": float(((pr != stay) & (pr != alt)).mean())}


for base in BASES:
    rng = random.Random(base * 17); hi = 999
    W, O, D1, D2, base_ok = [], [], [], [], []
    tries = 0
    k = 1
    while (len(W) < 2500 or len(O) < 2500) and tries < 300000:
        tries += 1
        a, b_ = rng.randint(base, hi), rng.randint(base, hi)
        shots = sample_shots(rng, (a, b_))
        if len(to_answer(a + b_, base)) <= k: continue
        c = carry_in(a, b_, base, k); curr = dig(a, base, k) + dig(b_, base, k)
        o_true = (curr + c) % base; o_alt = (curr + 1 - c) % base
        ans = to_answer(a + b_, base)
        if len(W) < 2500:
            # A written digit consistent with the opposite carry, under the rule carry = (written < a_0 + b_0 mod ...).
            a0, b0 = dig(a, base, 0), dig(b_, base, 0)
            # written digit is (a0+b0) mod base if truthful. Inference from it: carry iff written < a0.
            cands = [w for w in range(base) if w != (a0 + b0) % base and int(w < a0) != c]
            if cands:
                w = rng.choice(cands)
                W.append((prompt(a, b_, base, shots, DIGITS[w]), o_true, o_alt))
        if len(O) < 2500:
            a0, b0 = dig(a, base, 0), dig(b_, base, 0)
            cands = [d for d in range(base) if d != a0 and int(d + b0 >= base) != c]
            if cands and a >= base:
                na = setdig(a, base, 0, rng.choice(cands))
                O.append((prompt(na, b_, base, shots, ans[:k]), o_true, o_alt))
    res[base] = {"W_written_digit": tally(W, base) if W else None, "O_operand_digit": tally(O, base) if O else None}
    tries = 0
    k = 2
    while (len(D1) < 1500 or len(D2) < 1500) and tries < 500000:
        tries += 1
        a, b_ = rng.randint(base**2 if base**2 <= hi else base, hi), rng.randint(base**2 if base**2 <= hi else base, hi)
        if dig(a, base, 1) + dig(b_, base, 1) != base - 1: continue
        if len(to_answer(a + b_, base)) <= k: continue
        shots = sample_shots(rng, (a, b_))
        c = carry_in(a, b_, base, 2); curr = dig(a, base, 2) + dig(b_, base, 2)
        o_true = (curr + c) % base; o_alt = (curr + 1 - c) % base
        ans = to_answer(a + b_, base)
        a0, b0 = dig(a, base, 0), dig(b_, base, 0)
        if len(D1) < 1500:
            cands = [d for d in range(base) if int(d + b0 >= base) != c]
            if cands:
                na = setdig(a, base, 0, rng.choice(cands))
                D1.append((prompt(na, b_, base, shots, ans[:2]), o_true, o_alt))
        if len(D2) < 1500:
            # With digit-1 sum = base-1, written o_1 = 0 means a carry arrived; base-1 means none.
            w1 = (base - 1) if c else 0
            D2.append((prompt(a, b_, base, shots, ans[0] + DIGITS[w1]), o_true, o_alt))
    res[base]["D1_operand_two_back"] = tally(D1, base) if D1 else None
    res[base]["D2_written_chain"] = tally(D2, base) if D2 else None
    print(base, {a2: ({k2: (round(v, 3) if isinstance(v, float) else v) for k2, v in r.items()} if r else None) for a2, r in res[base].items()}, flush=True)
    save_json("f04_carry", res)
print("DONE")
