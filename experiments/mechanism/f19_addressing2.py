"""F19 (forward passes): the rest of addressing.

A. Digit 0, off-distribution test lines. "x+y+z=" and "xy=" (no plus). Does the a-reader look before the FIRST plus, the LAST plus, or just latest digit?
B. Is the pointer in the query or the keys? Shift '+' position embedding, then patch layer 0's output only at the query, or only at operand tokens.
C. Which L0 head tells the query where '+' and '=' are? Patch one L0 head's output at the query from the shifted run.
D. What switches head roles between digits? Top digit-index neurons, their firing by k, and zeroing them while recording which heads read the current digits.
"""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import BASES, DEVICE, DIGITS, Model, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); PE = model.PE; res = {"A": {}, "B": {}, "C": {}, "D": {}}
tok = np.array([model.vocab[c] for c in DIGITS])
def ids(strs):
    seqs = [[model.vocab[c] for c in s] for s in strs]; n = max(map(len, seqs)); x = torch.full((len(seqs), n), model.pad, dtype=torch.long)
    for j, s in enumerate(seqs): x[j, :len(s)] = torch.tensor(s)
    return x.to(DEVICE), torch.tensor([len(s) - 1 for s in seqs], device=DEVICE)
# A
for b in BASES:
    rng = random.Random(b + 3); rows = []
    lo = {2: 3, 8: 2, 10: 2, 16: 2}[b]; L3 = {2: 5, 8: 3, 10: 3, 16: 3}[b]
    def num():
        return rng.choice(DIGITS[1:b]) + "".join(rng.choice(DIGITS[:b]) for _ in range(rng.randint(lo - 1, L3 - 1)))
    S3, S2, seg = [], [], []
    for _ in range(800):
        x, y, z = num(), num(), num(); ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (-1, -1))); c = len(ctx) + 1
        S3.append(f"{ctx}\n{x}+{y}+{z}="); seg.append((c + len(x) - 1, c + len(x) + 1 + len(y) - 1, c + len(x) + 1 + len(y) + 1 + len(z) - 1, int(x[-1], b), int(y[-1], b), int(z[-1], b)))
    pats, pr = [], []
    for i in range(0, 800, 200):
        x_, q = ids(S3[i:i + 200]); lg, c = model.forward(x_, capture={"pattern1"}); J = torch.arange(len(x_), device=DEVICE)
        sg = torch.as_tensor(np.array(seg[i:i + 200]), device=DEVICE)
        pats.append(torch.stack([c["pattern1"][J, :, q, sg[:, s]] for s in range(3)], 2).cpu().numpy()); pr.append(lg[J, q].argmax(-1).cpu().numpy())
    P = np.concatenate(pats); pr = np.concatenate(pr); seg_ = np.array(seg); J = np.arange(800)
    mass = {f"H{h}": [float(P[:, h, s].mean()) for s in range(3)] for h in (0, 1, 3)}
    pairs = {"x+y": seg_[:, 3] + seg_[:, 4], "y+z": seg_[:, 4] + seg_[:, 5], "x+z": seg_[:, 3] + seg_[:, 5]}
    out = {k2: float((pr == tok[v % b]).mean()) for k2, v in pairs.items()}
    res["A"][f"{b}/double_plus"] = {"attention_on_last_digit_of_segment_1_2_3": mass, "output_matches": out}
    print(f"A b{b} x+y+z=: mass on end of (x,y,z) {({h: np.round(v, 2).tolist() for h, v in mass.items()})}; output matches {({k2: round(v, 2) for k2, v in out.items()})}", flush=True)
    # no plus
    S0, ends = [], []
    for _ in range(800):
        x, y = num(), num(); ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (-1, -1))); c = len(ctx) + 1
        S0.append(f"{ctx}\n{x}{y}="); ends.append((c + len(x) + len(y) - 1, c + len(x) + len(y) - 2))
    pats = []
    for i in range(0, 800, 200):
        x_, q = ids(S0[i:i + 200]); _, c = model.forward(x_, capture={"pattern1"}); J2 = torch.arange(len(x_), device=DEVICE); en = torch.as_tensor(np.array(ends[i:i + 200]), device=DEVICE); pats.append(torch.stack([c["pattern1"][J2, :, q, en[:, s]] for s in range(2)], 2).cpu().numpy())
    P = np.concatenate(pats)
    res["A"][f"{b}/no_plus"] = {f"H{h}": [float(P[:, h, 0].mean()), float(P[:, h, 1].mean())] for h in (0, 1, 3)}
    print(f"A b{b} no plus: mass on (last digit, second last)", {k2: np.round(v, 2).tolist() for k2, v in res["A"][f"{b}/no_plus"].items()}, flush=True)
# B and C
READ = {(10, 1): (3, 3), (10, 2): (0, 0), (16, 1): (3, 3), (8, 1): (3, 3), (2, 2): (1, 0), (2, 3): (1, 0)}
for (b, k), (ra, rb) in READ.items():
    rng = random.Random(b * 5 + k); P = []
    while len(P) < 400:
        a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b); ans = to_answer(a + c_, b)
        if len(sa) <= k or len(sb) <= k or len(ans) <= k: continue
        ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_))); c = len(ctx) + 1
        P.append((f"{ctx}\n{sa}+{sb}={ans[:k]}", c, c + len(sa), c + len(sa) + 1 + len(sb)))
    def meanpos(pat, h, q, lo, hi, J, n):
        row = pat[J, h, q]; idx = torch.arange(n, device=DEVICE)[None]; m = (idx >= lo[:, None]) & (idx < hi[:, None]); return (row * m * idx).sum(1) / (row * m).sum(1).clamp_min(1e-6)
    out = {}
    for tgt in ("plus", "eq"):
        tot = {}
        for i in range(0, 400, 100):
            ch = P[i:i + 100]; x, q = ids([p[0] for p in ch]); J = torch.arange(len(ch), device=DEVICE); n = x.shape[1]; idx = torch.arange(n, device=DEVICE)[None]
            st, pl, eq = [torch.tensor([p[t] for p in ch], device=DEVICE) for t in (1, 2, 3)]
            m = idx == (pl if tgt == "plus" else eq)[:, None]; pos = idx.expand(x.shape); delta = (PE[(pos - 1).clamp_min(0)] - PE[pos]) * m[..., None]
            names = {"z0", "resid_pre1", "pattern1", "mlp_post0"}
            _, c0 = model.forward(x, capture=names); _, c1 = model.forward(x, hooks={"resid_pre0": lambda t, delta=delta: t + delta}, capture=names)
            base_a = meanpos(c0["pattern1"], ra, q, st, pl, J, n); base_b = meanpos(c0["pattern1"], rb, q, pl + 1, eq, J, n)
            def rec(key, hooks):
                _, cc = model.forward(x, hooks=hooks, capture={"pattern1"})
                d = tot.setdefault(key, [0., 0.]); d[0] += (meanpos(cc["pattern1"], ra, q, st, pl, J, n) - base_a).sum().item(); d[1] += (meanpos(cc["pattern1"], rb, q, pl + 1, eq, J, n) - base_b).sum().item()
            rec("full shift", {"resid_pre0": lambda t, delta=delta: t + delta})
            qm = idx == q[:, None]; opm = (idx >= st[:, None]) & (idx < eq[:, None])
            for reg, mk in (("layer 0 output at query only", qm), ("layer 0 output at operands/+/= only", (idx >= st[:, None]) & (idx <= eq[:, None]) & ~qm)):
                def hk(t, mk=mk, s=c1["resid_pre1"]):
                    t = t.clone(); t[mk] = s[mk]; return t
                rec(reg, {"resid_pre1": hk})
            for h in range(4):
                def hk(t, s=c1["z0"], h=h):
                    t = t.clone(); t[J, q, h] = s[J, q, h]; return t
                rec(f"L0H{h} at query", {"z0": hk})
            def hk(t, s=c1["mlp_post0"]):
                t = t.clone(); t[J, q] = s[J, q]; return t
            rec("L0 neurons at query", {"mlp_post0": hk})
        out[tgt] = {a2: (v[0] / 400, v[1] / 400) for a2, v in tot.items()}
    res["B"][f"{b}/k{k}"] = out
    print(f"B/C b{b} k{k}: shift of attended position, a and b, for a -1 shift of the token's position tag")
    for tgt in out:
        print("   ", tgt, {a2: (round(v[0], 2), round(v[1], 2)) for a2, v in out[tgt].items()}, flush=True)
    save_json("f19_addressing2", res)
# D role switch
for b in (8, 10, 16):
    rows = {}
    rng = random.Random(b); cases = {}
    for k in (0, 1, 2):
        P = []
        while len(P) < 500:
            a, c_ = rng.randint(0, 999), rng.randint(0, 999); sa, sb = to_base(a, b), to_base(c_, b); ans = to_answer(a + c_, b)
            if len(sa) <= k or len(sb) <= k or len(ans) <= k: continue
            ctx = "\n".join(render(*s, b) for s in sample_shots(rng, (a, c_))); c = len(ctx) + 1
            P.append((f"{ctx}\n{sa}+{sb}={ans[:k]}", c + len(sa), c + len(sa) + 1 + len(sb)))
        cases[k] = P
    # find k-coded L0 neurons
    M = {}
    for k, P in cases.items():
        acc = []
        for i in range(0, 500, 125):
            x, q = ids([p[0] for p in P[i:i + 125]]); _, c = model.forward(x, capture={"mlp_post0"}); acc.append(c["mlp_post0"][torch.arange(len(x), device=DEVICE), q])
        M[k] = torch.cat(acc).mean(0).cpu().numpy()
    spread = np.stack([M[k] for k in (0, 1, 2)]); top = np.argsort(-(spread.max(0) - spread.min(0)))[:8]
    rows["top_k_neurons"] = {int(n): np.round(spread[:, n], 2).tolist() for n in top}
    print(f"D b{b} top digit-index L0 neurons (firing at digits 0,1,2):", rows["top_k_neurons"], flush=True)
    def reads(P, k, hooks=None):
        out = np.zeros((4, 2)); pred = 0
        for i in range(0, 500, 125):
            ch = P[i:i + 125]; x, q = ids([p[0] for p in ch]); J = torch.arange(len(ch), device=DEVICE); _, c = model.forward(x, hooks=hooks, capture={"pattern1"})
            pl, eq = [torch.tensor([p[t] for p in ch], device=DEVICE) for t in (1, 2)]
            for h in range(4):
                out[h, 0] += (c["pattern1"][J, h, q, pl - 1 - k] + c["pattern1"][J, h, q, eq - 1 - k]).sum().item()
                if k: out[h, 1] += (c["pattern1"][J, h, q, pl - k] + c["pattern1"][J, h, q, eq - k]).sum().item()
        return out / 500
    rows["reads"] = {}
    for k in (0, 1, 2):
        clean = reads(cases[k], k); rows["reads"][f"k{k}/clean"] = clean.tolist()
        print(f"   k{k} clean: mass on current digits by head {np.round(clean[:, 0], 2).tolist()}, on previous digits {np.round(clean[:, 1], 2).tolist()}", flush=True)
        for n in top[:3]:
            n = int(n)
            def hook(t, n=n):
                t = t.clone(); t[..., n] = 0.; return t
            r = reads(cases[k], k, {"mlp_post0": hook}); rows["reads"][f"k{k}/zero_{n}"] = r.tolist()
            print(f"      zero L0 neuron {n}: current {np.round(r[:, 0], 2).tolist()}, previous {np.round(r[:, 1], 2).tolist()}", flush=True)
    res["D"][b] = rows
save_json("f19_addressing2", res); print("DONE")
