"""Generate a synthetic training dataset and save it (locally by default, or push to the HF Hub).

Task: addition in four numeral bases (binary, octal, decimal, hexadecimal) simultaneously,
with a shared character-level tokenizer. Each prompt is k=8 few-shot solved examples followed
by an unsolved test problem. No base-indicator token is included — the model must infer the
base from the digit characters in the few-shot context.

Leakage-safety: operand pairs are split into disjoint train/val pools before any prompts are
built. Val questions always use val-pool operands; few-shot examples always draw from the
train pool, so a val pair never appears in any training prompt.

Performance: uses a pre-rendered lookup table + NumPy vectorised index sampling instead of
per-row Python loops — generates 1M rows in ~15-20s instead of ~10 min.
"""

import argparse
import time
import numpy as np
from datasets import Dataset, DatasetDict


# --------------------------------------------------------------------------- #
# Base conversion helpers
# --------------------------------------------------------------------------- #

def to_base(n: int, base: int) -> str:
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        rem = n % base
        digits.append("0123456789ABCDEF"[rem])
        n //= base
    return "".join(reversed(digits))


def has_carry(a: int, b: int, base: int) -> bool:
    carry = 0
    while a > 0 or b > 0 or carry > 0:
        col = (a % base) + (b % base) + carry
        carry = col // base
        if carry > 0:
            return True
        a //= base
        b //= base
    return False


# --------------------------------------------------------------------------- #
# Pre-render lookup table  (built once, ~40k entries, takes ~0.09 s)
# --------------------------------------------------------------------------- #

BASES = [2, 8, 10, 16]
FEW_SHOT = 8
MIN_CARRY_SHOTS = 2
MAX_OPERAND = 100   # operands drawn from [0, 99] inclusive


def _build_lookup() -> dict[tuple[int, int, int], str]:
    """Pre-render every (a, b, base) solved example string.

    Returns a dict mapping (a, b, base) -> "A + B = C" string.
    Building this once and indexing into it is ~100x faster than
    calling to_base() inside the row-generation loop.
    """
    t = time.time()
    table: dict[tuple[int, int, int], str] = {}
    for base in BASES:
        for a in range(MAX_OPERAND):
            for b in range(MAX_OPERAND):
                table[(a, b, base)] = (
                    f"{to_base(a, base)} + {to_base(b, base)} = {to_base(a + b, base)}"
                )
    print(f"  Lookup table: {len(table):,} entries built in {time.time()-t:.2f}s")
    return table


def _build_carry_masks(
    pairs: list[tuple[int, int]]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """For each base, split `pairs` into carry / non-carry index arrays.

    Returns {base: (carry_indices, non_carry_indices)} where indices index into `pairs`.
    """
    masks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for base in BASES:
        carry_idx     = np.array([i for i, (a, b) in enumerate(pairs) if     has_carry(a, b, base)], dtype=np.int32)
        non_carry_idx = np.array([i for i, (a, b) in enumerate(pairs) if not has_carry(a, b, base)], dtype=np.int32)
        masks[base] = (carry_idx, non_carry_idx)
    return masks


# --------------------------------------------------------------------------- #
# Fast vectorised split generator
# --------------------------------------------------------------------------- #

def _generate_split(
    total: int,
    q_pool: list[tuple[int, int]],
    train_pairs: list[tuple[int, int]],
    train_carry_masks: dict[int, tuple[np.ndarray, np.ndarray]],
    lookup: dict[tuple[int, int, int], str],
    split_name: str,
    split_seed: int,
    few_shot: int = FEW_SHOT,
    min_carry_shots: int = MIN_CARRY_SHOTS,
) -> dict:
    """Generate `total` rows, distributed evenly across BASES.

    Key speed improvements over the naive loop:
    - All random index sampling is done in one NumPy call per base.
    - String assembly uses list comprehensions over pre-rendered strings
      rather than calling to_base() per row.
    - dict-of-lists is built by extending pre-allocated lists, avoiding
      repeated dict merging.
    """
    rng = np.random.default_rng(split_seed)
    rows_per_base = total // len(BASES)

    ids, prompts, answers, bases_col, as_col, bs_col = [], [], [], [], [], []

    for base in BASES:
        carry_idx, non_carry_idx = train_carry_masks[base]

        n_carry     = min(min_carry_shots, len(carry_idx))
        n_non_carry = few_shot - n_carry

        # -------------------------------------------------------------- #
        # Sample test-problem operand pairs  (rows_per_base draws)
        # -------------------------------------------------------------- #
        q_indices = rng.integers(0, len(q_pool), size=rows_per_base)

        # -------------------------------------------------------------- #
        # Sample few-shot indices in one batch:
        #   shape (rows_per_base, few_shot)
        # Carry slots: first n_carry columns
        # Non-carry slots: next n_non_carry columns
        # -------------------------------------------------------------- #
        fs_carry = rng.integers(0, max(1, len(carry_idx)),
                                size=(rows_per_base, n_carry))          # (N, n_carry)
        fs_non   = rng.integers(0, max(1, len(non_carry_idx)),
                                size=(rows_per_base, n_non_carry))      # (N, n_non_carry)

        # Map local carry/non-carry indices back to pair indices in train_pairs.
        if len(carry_idx) > 0:
            fs_carry_mapped = carry_idx[fs_carry]           # (N, n_carry)
        else:
            fs_carry_mapped = np.zeros((rows_per_base, 0), dtype=np.int32)

        if len(non_carry_idx) > 0:
            fs_non_mapped = non_carry_idx[fs_non]           # (N, n_non_carry)
        else:
            fs_non_mapped = np.zeros((rows_per_base, 0), dtype=np.int32)

        # Concatenate and shuffle columns so carry examples aren't always first.
        fs_all = np.concatenate([fs_carry_mapped, fs_non_mapped], axis=1)  # (N, few_shot)
        # Shuffle each row independently by argsort of random noise.
        shuffle_keys = rng.random(fs_all.shape)
        order = np.argsort(shuffle_keys, axis=1)
        fs_all = fs_all[np.arange(rows_per_base)[:, None], order]

        # -------------------------------------------------------------- #
        # Assemble prompt strings (list comprehension, no Python loops
        # over individual characters).
        # -------------------------------------------------------------- #
        for i in range(rows_per_base):
            a, b = q_pool[q_indices[i]]

            # 8 solved few-shot examples joined by " , "
            fs_strs = [lookup[(train_pairs[fs_all[i, k]][0],
                               train_pairs[fs_all[i, k]][1],
                               base)]
                       for k in range(few_shot)]
            solved = " , ".join(fs_strs)

            # Unsolved test problem
            test_q = f"{to_base(a, base)} + {to_base(b, base)} ="
            prompt = f"{solved} , {test_q}"
            answer = to_base(a + b, base)

            ids.append(f"{split_name}-b{base}-{i}")
            prompts.append(prompt)
            answers.append(answer)
            bases_col.append(base)
            as_col.append(a)
            bs_col.append(b)

    # Interleave bases by shuffling in-place.
    perm = rng.permutation(len(ids)).tolist()
    return {
        "_id":    [ids[i]      for i in perm],
        "prompt": [prompts[i]  for i in perm],
        "answer": [answers[i]  for i in perm],
        "base":   [bases_col[i] for i in perm],
        "a":      [as_col[i]   for i in perm],
        "b":      [bs_col[i]   for i in perm],
    }


# --------------------------------------------------------------------------- #
# Sanity checks
# --------------------------------------------------------------------------- #

def _sanity_check(
    train_ds: Dataset,
    val_ds: Dataset,
    train_pairs: list[tuple[int, int]],
    val_pairs: list[tuple[int, int]],
) -> None:
    print("\nRunning sanity checks…")
    train_pair_set = set(train_pairs)
    val_pair_set   = set(val_pairs)

    assert train_pair_set.isdisjoint(val_pair_set), "Train/val operand pools overlap — leakage!"

    rng0 = np.random.default_rng(0)
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        sample_idx = rng0.integers(len(ds), size=min(500, len(ds)))
        for idx in sample_idx:
            row = ds[int(idx)]
            expected = to_base(row["a"] + row["b"], row["base"])
            assert row["answer"] == expected, (
                f"[{split_name}] row {idx}: expected {expected!r} got {row['answer']!r}"
            )
        print(f"  [{split_name}] 500 answer spot-checks passed.")

    for row in val_ds.select(range(min(200, len(val_ds)))):
        assert (row["a"], row["b"]) in val_pair_set, \
            f"Val row has train-pool pair ({row['a']}, {row['b']}) — leakage!"
    print("  Val operand containment check passed.")

    from collections import Counter
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        counts = Counter(ds["base"])
        for base in BASES:
            n = counts[base]
            expected = len(ds) // len(BASES)
            assert abs(n - expected) <= len(BASES), \
                f"[{split_name}] base {base}: count {n} too far from {expected}"
        print(f"  [{split_name}] base distribution: {dict(sorted(counts.items()))}")

    VOCAB = set("0123456789ABCDEF+=, ")
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        for row in ds.select(range(min(200, len(ds)))):
            bad = set(row["prompt"]) - VOCAB
            assert not bad, f"Out-of-vocab chars {bad!r} in prompt: {row['prompt']!r}"
            assert row["prompt"].endswith("="), f"Prompt doesn't end with '=': {row['prompt']!r}"
    print("  Vocab + format checks passed.")
    print("All sanity checks passed.\n")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def build_dataset(
    train_size: int,
    val_size: int,
    seed: int,
    val_holdout: float,
) -> DatasetDict:
    """Build the train/validation DatasetDict for multi-base addition.

    Args:
        train_size:   Total training rows (split evenly across 4 bases).
        val_size:     Total validation rows (split evenly across 4 bases).
        seed:         Master random seed.
        val_holdout:  Fraction of (a, b) pairs reserved exclusively for val questions.

    Returns:
        DatasetDict with 'train' and 'validation' splits. Columns:
            prompt  - full few-shot prompt ending at '='
            answer  - correct answer string in the appropriate base
            base    - integer base (2, 8, 10, or 16)
            a, b    - operands as decimal integers
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)

    # 1. Build operand pool and split into disjoint train / val halves.
    all_pairs = [(a, b) for a in range(MAX_OPERAND) for b in range(MAX_OPERAND)]
    indices   = np.arange(len(all_pairs))
    rng.shuffle(indices)

    split_idx   = int(len(indices) * (1 - val_holdout))
    train_pairs = [all_pairs[i] for i in indices[:split_idx]]
    val_pairs   = [all_pairs[i] for i in indices[split_idx:]]
    print(f"Operand pool: {len(all_pairs):,} | train: {len(train_pairs):,} | val: {len(val_pairs):,}")

    # 2. Pre-build lookup table and carry masks (done once, shared across splits).
    print("Pre-building lookup table and carry masks…")
    lookup            = _build_lookup()
    train_carry_masks = _build_carry_masks(train_pairs)

    # 3. Generate splits.
    print(f"Generating train split ({train_size:,} rows)…")
    t1 = time.time()
    train_data = _generate_split(train_size, train_pairs, train_pairs,
                                  train_carry_masks, lookup, "train", seed)
    print(f"  Done in {time.time()-t1:.1f}s")

    print(f"Generating validation split ({val_size:,} rows)…")
    t2 = time.time()
    val_data = _generate_split(val_size, val_pairs, train_pairs,
                                train_carry_masks, lookup, "validation", seed + 1)
    print(f"  Done in {time.time()-t2:.1f}s")

    train_ds = Dataset.from_dict(train_data)
    val_ds   = Dataset.from_dict(val_data)

    _sanity_check(train_ds, val_ds, train_pairs, val_pairs)

    print(f"Total time: {time.time()-t0:.1f}s")
    return DatasetDict({"train": train_ds, "validation": val_ds})


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic multi-base addition dataset and save it locally or to the Hub"
    )
    parser.add_argument("--hub-name",   type=str,   default=None)
    parser.add_argument("--output-dir", type=str,   default="./artifacts/dataset")
    parser.add_argument("--train-size", type=int,   default=1_000_000)
    parser.add_argument("--val-size",   type=int,   default=100_000)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--val-holdout",type=float, default=0.1)
    parser.add_argument("--shard-size", type=str,   default="500MB")
    args = parser.parse_args()

    print(f"Generating dataset | train={args.train_size:,} val={args.val_size:,} "
          f"seed={args.seed} val_holdout={args.val_holdout}\n")

    dataset = build_dataset(
        train_size=args.train_size,
        val_size=args.val_size,
        seed=args.seed,
        val_holdout=args.val_holdout,
    )

    print(dataset)
    print("\nSample rows:")
    for i in range(4):
        row = dataset["train"][i]
        print(f"  base={row['base']:2d}  a={row['a']:3d}  b={row['b']:3d}  answer={row['answer']}")
        print(f"         {row['prompt'][:120]}")

    if args.hub_name:
        dataset.push_to_hub(args.hub_name, max_shard_size=args.shard_size)
        print(f"\nPushed dataset to hub: {args.hub_name}")
    else:
        dataset.save_to_disk(args.output_dir)
        print(f"\nSaved dataset to {args.output_dir}")