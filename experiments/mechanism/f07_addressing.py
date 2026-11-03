"""F07 (forward passes): how does a query find digit k? Pointer test using position embeddings.

Add PE[p+d] - PE[p] (d = -1, +1) at selected positions only, then measure where the reading head attends.
If the query computes target = pos('+') - 1 - k from position information:
  shifting '+' should move attention on a by the same d.
If keys are matched by absolute position:
  shifting operand digits should also move attention. If keys know only their index within the number, it should not.
Reports attention mass at offsets from the correct digit (offset -1 = one place to the left, more significant),
and whether the output digit follows the digit that attention moved to.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); res = {}
PE = model.PE
# (base, k): head reading a, head reading b
READERS = {(2, 0): (1, 0), (2, 1): (1, 0), (2, 2): (1, 0), (8, 0): (1, 0), (8, 1): (3, 3), (8, 2): (0, 0), (10, 0): (1, 0), (10, 1): (3, 3), (10, 2): (0, 0), (16, 0): (1, 0), (16, 1): (3, 3), (16, 2): (0, 0)}
TARGETS = ["plus", "eq", "query_token", "written_all", "a_digits", "b_digits", "operands", "test_line_all", "context_all"]


def build(base, k, n, seed):
    rng = random.Random(seed); out = []
    margin = 2
    while len(out) < n:
        a, b_ = rng.randint(0, 999), rng.randint(0, 999)
        sa, sb = to_base(a, base), to_base(b_, base)
        if len(sa) < k + margin or len(sb) < k + margin:
            continue
        ctx = "\n".join(render(*s, base) for s in sample_shots(rng, (a, b_)))
        ans = to_answer(a + b_, base)
        if len(ans) <= k:
            continue
        s = f"{ctx}\n{sa}+{sb}={ans[:k]}"
        ctx_len = len(ctx) + 1
        out.append(dict(s=s, a=a, b=b_, plus=ctx_len + len(sa), eq=ctx_len + len(sa) + 1 + len(sb), q=len(s) - 1, start=ctx_len, base=base))
    return out


def run(P, k, target, d, ra, rb):
    bs = 256
    ma, mb, n = np.zeros(5), np.zeros(5), 0
    corr_true = follow_left = 0
    for i in range(0, len(P), bs):
        chunk = P[i:i + bs]
        seqs = [[model.vocab[c] for c in p["s"]] for p in chunk]
        L = max(map(len, seqs)); x = torch.full((len(seqs), L), model.pad, dtype=torch.long)
        for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
        x = x.to(DEVICE)
        mask = torch.zeros(x.shape, dtype=torch.bool, device=DEVICE)
        for j, p in enumerate(chunk):
            if target == "plus": mask[j, p["plus"]] = True
            elif target == "eq": mask[j, p["eq"]] = True
            elif target == "query_token": mask[j, p["q"]] = True
            elif target == "written_all": mask[j, p["eq"] + 1:p["q"] + 1] = True
            elif target == "a_digits": mask[j, p["start"]:p["plus"]] = True
            elif target == "b_digits": mask[j, p["plus"] + 1:p["eq"]] = True
            elif target == "operands": mask[j, p["start"]:p["eq"]] = True
            elif target == "test_line_all": mask[j, p["start"]:p["q"] + 1] = True
            elif target == "context_all": mask[j, :p["start"]] = True
        pos = torch.arange(L, device=DEVICE)[None].expand(x.shape)
        new = (pos + d).clamp(0, 255)
        delta = (PE[new] - PE[pos]) * mask[..., None]
        lg, c = model.forward(x, hooks={"resid_pre0": (lambda t, delta=delta: t + delta)} if d else None, capture={"pattern1"})
        pat = c["pattern1"]
        J = torch.arange(len(chunk), device=DEVICE)
        q = torch.tensor([p["q"] for p in chunk], device=DEVICE)
        pl = torch.tensor([p["plus"] for p in chunk], device=DEVICE); eq = torch.tensor([p["eq"] for p in chunk], device=DEVICE); st = torch.tensor([p["start"] for p in chunk], device=DEVICE)
        for off in range(-2, 3):
            pa = pl - 1 - k + off; pb = eq - 1 - k + off
            va = (pa >= st) & (pa < pl); vb = (pb > pl) & (pb < eq)
            ma[off + 2] += (pat[J, ra, q, pa.clamp(0, L - 1)] * va).sum().item()
            mb[off + 2] += (pat[J, rb, q, pb.clamp(0, L - 1)] * vb).sum().item()
        pr = lg[J, q].argmax(-1).cpu().numpy()
        b = chunk[0]["base"]; M = b**k
        A = np.array([p["a"] for p in chunk]); B = np.array([p["b"] for p in chunk])
        c0 = ((A % M) + (B % M) >= M).astype(int) if k else np.zeros(len(A), int)
        true = ((A // M) % b + (B // M) % b + c0) % b
        left_ = ((A // (M * b)) % b + (B // M) % b + c0) % b
        tok = np.array([model.vocab[ch] for ch in DIGITS])
        corr_true += (pr == tok[true]).sum()
        follow_left += ((pr == tok[left_]) & (left_ != true)).sum()
        n += len(chunk)
    return {"a_offset_mass": (ma / n).tolist(), "b_offset_mass": (mb / n).tolist(), "output_true": float(corr_true / n), "output_uses_a_one_left": float(follow_left / n)}


for (base, k), (ra, rb) in READERS.items():
    P = build(base, k, 500, base * 10 + k)
    row = {"clean": run(P, k, "plus", 0, ra, rb)}
    for t in TARGETS:
        for d in (-1, 1):
            row[f"{t}/{d:+d}"] = run(P, k, t, d, ra, rb)
    res[f"{base}/k{k}"] = row
    print(f"\nb{base} k{k} readers a:H{ra} b:H{rb}. Columns: mass on a at offsets -2..2 | b at offsets | output correct")
    def fmt(r): return " ".join(f"{x:.2f}" for x in r["a_offset_mass"]) + " | " + " ".join(f"{x:.2f}" for x in r["b_offset_mass"]) + f" | {r['output_true']:.2f}"
    print("  clean".ljust(26), fmt(row["clean"]))
    for key, r in row.items():
        if key != "clean":
            print(("  " + key).ljust(26), fmt(r), flush=True)
    save_json("f07_addressing", res)
print("DONE")
