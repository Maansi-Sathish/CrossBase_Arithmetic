"""E2: pull compact per-position activation tables out of the team's bucket captures.

The saved tensors are views into per-prompt storages covering EVERY token position, so although the
schema names only 7 positions, the full sequence is recoverable. We keep all positions on the test
line and all answer query positions, then check them against a fresh forward pass.

Files are read with weights_only=True, so nothing in them can execute code.

Output per base under results/tables/b{base}/:
  rows.npz: idx, a, b, kind, j, pos, correct
  {resid0,resid1,mlp0,mlp1,z0,z1}.npy: float32 [rows, dim]; z is flattened head-by-head
Kinds: 0 = digit j of a, 1 = digit j of b, 2 = '+', 3 = query q_j ('=' is q_0, q_j reads answer digit j-1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, BUCKET, Batch, Model, Problem, save_json

OUT = BUCKET.parent / "tables"
A_DIGIT, B_DIGIT, PLUS, QUERY = 0, 1, 2, 3


def full(t: torch.Tensor, n: int, d: int) -> torch.Tensor:
    """Recover the [n, d] per-position tensor a captured single-position view points into."""
    return t.new_empty(0).set_(t.untyped_storage(), 0, (n, d))


def parse_shots(prompt: str, base: int) -> list[tuple[int, int]]:
    """Recover the few-shot integer pairs from a prompt."""
    out = []
    for line in prompt.split("\n")[:-1]:
        lhs = line.split("=")[0]
        x, y = lhs.split("+")
        out.append((int(x, base), int(y, base)))
    return out


model = Model()
summary = {}
all_pairs = {}
for base in BASES:
    path = BUCKET / f"base{base}.pt"
    print("loading", path, flush=True)
    data = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    rows = data["baseline"]
    out_dir = OUT / f"b{base}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. count rows so the memmaps can be sized up front
    specs = []
    n_rows = 0
    for idx, r in enumerate(rows):
        m = r["metadata"]
        la, lb = None, None
        prompt = r["prompt"]
        line = prompt.split("\n")[-1]
        sa, sb = line[:-1].split("+")
        la, lb = len(sa), len(sb)
        p_eq = len(prompt) - 1
        p_plus = p_eq - lb - 1
        positions = []
        for j in range(la):
            positions.append((A_DIGIT, j, p_plus - 1 - j))
        for j in range(lb):
            positions.append((B_DIGIT, j, p_eq - 1 - j))
        positions.append((PLUS, 0, p_plus))
        # query positions: as many as the model actually generated (ending at the token before EOS)
        n_tokens = len(r["result"]["output_ids"])
        n_query = n_tokens - 1 - p_eq  # position n_tokens-1 is the last token; its prediction is unused
        for j in range(max(n_query, 0)):
            positions.append((QUERY, j, p_eq + j))
        specs.append((idx, m["a"], m["b"], positions, n_tokens))
        n_rows += len(positions)
    print(base, "rows", n_rows, flush=True)

    arrs = {
        "resid0": np.lib.format.open_memmap(out_dir / "resid0.npy", "w+", np.float32, (n_rows, 64)),
        "resid1": np.lib.format.open_memmap(out_dir / "resid1.npy", "w+", np.float32, (n_rows, 64)),
        "mlp0": np.lib.format.open_memmap(out_dir / "mlp0.npy", "w+", np.float32, (n_rows, 256)),
        "mlp1": np.lib.format.open_memmap(out_dir / "mlp1.npy", "w+", np.float32, (n_rows, 256)),
        "z0": np.lib.format.open_memmap(out_dir / "z0.npy", "w+", np.float32, (n_rows, 64)),
        "z1": np.lib.format.open_memmap(out_dir / "z1.npy", "w+", np.float32, (n_rows, 64)),
    }
    meta = {k: np.zeros(n_rows, dtype=np.int32) for k in ["idx", "a", "b", "kind", "j", "pos", "correct"]}

    # 2. copy out. Use the eq_sign view, whose storage covers the whole sequence.
    i = 0
    correct_total = 0
    for (idx, a, b, positions, n_tokens), r in zip(specs, rows):
        g = r["result"]["geometry"]["eq_sign"]
        res0 = full(g[0]["residual"], n_tokens, 64).numpy()
        res1 = full(g[1]["residual"], n_tokens, 64).numpy()
        m0 = full(g[0]["mlp"], n_tokens, 256).numpy()
        m1 = full(g[1]["mlp"], n_tokens, 256).numpy()
        z0 = full(g[0]["heads"], n_tokens, 64).numpy()
        z1 = full(g[1]["heads"], n_tokens, 64).numpy()
        ok = int(r["result"]["answer"]["token"] == r["metadata"]["answer"])
        correct_total += ok
        ps = np.array([p[2] for p in positions])
        n = len(ps)
        arrs["resid0"][i : i + n] = res0[ps]
        arrs["resid1"][i : i + n] = res1[ps]
        arrs["mlp0"][i : i + n] = m0[ps]
        arrs["mlp1"][i : i + n] = m1[ps]
        arrs["z0"][i : i + n] = z0[ps]
        arrs["z1"][i : i + n] = z1[ps]
        meta["idx"][i : i + n] = idx
        meta["a"][i : i + n] = a
        meta["b"][i : i + n] = b
        meta["kind"][i : i + n] = [p[0] for p in positions]
        meta["j"][i : i + n] = [p[1] for p in positions]
        meta["pos"][i : i + n] = ps
        meta["correct"][i : i + n] = ok
        i += n
    for v in arrs.values():
        v.flush()
    np.savez(out_dir / "rows.npz", **meta)

    # 3. verify against a fresh forward pass on the same prompts
    sample = list(range(0, 10000, 37))[:256]
    probs = []
    seqs = []
    for s in sample:
        seqs.append(rows[s]["result"]["output_ids"].tolist())
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), model.pad, dtype=torch.long)
    for k, s in enumerate(seqs):
        ids[k, : len(s)] = torch.tensor(s)
    with torch.no_grad():
        _, cache = model.forward(ids.to(model.device), capture={"resid_post0", "resid_post1", "mlp_post1", "z0"})
    diffs = []
    for k, s in enumerate(sample):
        g = rows[s]["result"]["geometry"]["eq_sign"]
        L = len(seqs[k])
        diffs.append(float((full(g[1]["residual"], L, 64).to(model.device) - cache["resid_post1"][k, :L]).abs().max()))
        diffs.append(float((full(g[1]["mlp"], L, 256).to(model.device) - cache["mlp_post1"][k, :L]).abs().max()))
        diffs.append(float((full(g[0]["heads"], L, 64).to(model.device) - cache["z0"][k, :L].flatten(1)).abs().max()))
    summary[base] = {
        "rows": n_rows,
        "bucket_accuracy": correct_total / len(rows),
        "max_abs_diff_vs_fresh_forward": max(diffs),
        "num_prompts": len(rows),
    }
    all_pairs[base] = np.array([[s[1], s[2]] for s in specs])
    # the prompts, for cross-base matching
    np.save(out_dir / "pairs.npy", all_pairs[base])
    shots = np.array([parse_shots(r["prompt"], base) for r in rows], dtype=np.int32)
    np.save(out_dir / "shots.npy", shots)
    print(base, summary[base], flush=True)
    del data, rows, arrs

summary["matched_pairs_across_bases"] = {
    str(b): float((all_pairs[b] == all_pairs[10]).all(1).mean()) for b in BASES
}
summary["matched_shots_across_bases"] = {
    str(b): float((np.load(OUT / f"b{b}" / "shots.npy") == np.load(OUT / "b10" / "shots.npy")).all((1, 2)).mean())
    for b in BASES
}
print(summary)
save_json("e02_extract", summary)
