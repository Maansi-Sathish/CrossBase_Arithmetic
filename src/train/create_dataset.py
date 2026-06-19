"""Generate a synthetic training dataset and save it (locally by default, or push to the HF Hub).

Task: addition in four numeral bases (binary, octal, decimal, hexadecimal) simultaneously,
with a shared character-level tokenizer. Each prompt is FEW_SHOT solved examples followed
by an unsolved test problem. No base-indicator token is included — the model must infer the
base from the digit characters in the few-shot context.

This version implements the exhaustive-pair design (not resampling-with-replacement):
  - For each base, all (max_operand+1)^2 (a, b) pairs over [0, max_operand] x [0, max_operand]
    are enumerated.
  - Each base's pairs are split (1 - val_holdout) / val_holdout -> train / val,
    independently per base.
  - Every (a, b) pair becomes exactly one row (no duplicates, no oversampling).

Answer convention (Lee et al.): operands are written forward (normal digit order), but the
answer is written in REVERSED digit order, unpadded. E.g. decimal 46+52=98: the operands stay
forward, but the answer "98" is reversed to "89", so the training row is "46+52=89". Do not
reverse the operands, only the answer -- see to_base_answer below for the precise rule.

Whitespace: no spaces around operators (tokenizer now strips them). Few-shot examples are
joined with "\\n", matching the tokenizer's vocabulary (comma is no longer the separator).

Leakage-safety:
  - Each base's pairs are split into disjoint train/val pools before any prompts are built.
  - Few-shot examples for BOTH train and val rows are drawn only from that base's TRAIN pool,
    so a val pair never appears anywhere in a training context.
  - Few-shot sampling is WITHOUT replacement and EXCLUDES the test pair itself, so the test
    problem is never inadvertently shown solved in its own prompt.
"""

import argparse
import time
import numpy as np
from datasets import Dataset, DatasetDict


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

BASES = [2, 8, 10, 16]
FEW_SHOT = 5
PAD_WIDTH = 0  # 0 = no padding (current default). Set >0 to zero-pad operands to this width.


# --------------------------------------------------------------------------- #
# Base conversion helpers
# --------------------------------------------------------------------------- #

def to_base(n: int, base: int) -> str:
    """Convert a non-negative integer to its string representation in the given base.

    Uses uppercase A-F for hex digits. If PAD_WIDTH > 0, left-pads with zeros to that
    width (operand padding toggle -- off by default).
    """
    if n == 0:
        digits_str = "0"
    else:
        digits = []
        while n > 0:
            digits.append("0123456789ABCDEF"[n % base])
            n //= base
        digits_str = "".join(reversed(digits))

    if PAD_WIDTH > 0:
        digits_str = digits_str.zfill(PAD_WIDTH)

    return digits_str


def to_base_answer(n: int, base: int) -> str:
    """Answer rendering: reversed digit order, never padded.

    Per the Lee et al. convention: the model is trained to emit the answer's digits
    least-significant-first. This strips any padding (answers are never padded,
    regardless of PAD_WIDTH) and reverses the forward digit string.
    """
    fwd = to_base(n, base).lstrip("0") or "0"  # strip any padding before reversing
    return fwd[::-1]


# --------------------------------------------------------------------------- #
# Prompt rendering
# --------------------------------------------------------------------------- #

def render_example(a: int, b: int, base: int) -> str:
    """Render one solved addition example in the given base.

    Format: "A+B=C" (no spaces). Operands are forward digit order; answer is the
    reversed-digit-order convention (to_base_answer).
    """
    return f"{to_base(a, base)}+{to_base(b, base)}={to_base_answer(a + b, base)}"


def build_prompt(
    test_a: int,
    test_b: int,
    base: int,
    fs_pairs: list[tuple[int, int]],
    rng: np.random.Generator,
    few_shot: int = FEW_SHOT,
) -> str:
    """Build one few-shot prompt ending at '='. Optimized for large datasets."""
    # 1. Pick indices uniformly from the whole pool
    chosen_indices = rng.choice(len(fs_pairs), size=few_shot, replace=False)
    
    fs_strs = []
    for idx in chosen_indices:
        pair = fs_pairs[idx]
        
        # 2. If we accidentally picked the test pair itself, swap it on the fly!
        if pair == (test_a, test_b):
            # Use an alternative index that we know isn't in chosen_indices
            # (e.g., adding 1 or wrapped around the list length)
            alt_idx = (idx + 1) % len(fs_pairs)
            while alt_idx in chosen_indices:
                alt_idx = (alt_idx + 1) % len(fs_pairs)
            pair = fs_pairs[alt_idx]
            
        fs_strs.append(render_example(pair[0], pair[1], base))

    solved = "\n".join(fs_strs)
    test_q = f"{to_base(test_a, base)}+{to_base(test_b, base)}="
    return f"{solved}\n{test_q}"


# --------------------------------------------------------------------------- #
# Core dataset builder
# --------------------------------------------------------------------------- #

def build_dataset(
    seed: int = 42,
    val_holdout: float = 0.5,
    max_operand: int = 99,
) -> DatasetDict:
    """Build the train/validation DatasetDict for multi-base addition.

    Exhaustive design: every (a, b) pair in [0, max_operand] x [0, max_operand] is used
    exactly once, per base. Each base's pairs are split (1-val_holdout)/val_holdout into
    train/val, using seed + idx as that base's split seed (idx = BASES.index(base)).

    Args:
        seed:        Master random seed. Each base derives its own split/few-shot seed as
                     seed + idx, where idx is the base's position in BASES.
        val_holdout: Fraction of pairs per base held out for validation (default 0.5).
        max_operand: Operands are drawn from [0, max_operand] inclusive (default 99).

    Returns:
        DatasetDict with "train" and "validation" splits. Columns:
            prompt  - the full few-shot prompt ending at '='
            answer  - the correct answer string, reversed-digit, in the appropriate base
            base    - integer base (2, 8, 10, or 16)
            a       - first operand as a decimal integer  b  - second operand as a decimal integer
    """
    t0 = time.time()

    # All (a, b) pairs for operands in [0, max_operand].
    all_pairs = [(a, b) for a in range(max_operand + 1) for b in range(max_operand + 1)]
    n_total = len(all_pairs)
    n_val = int(n_total * val_holdout)
    n_train = n_total - n_val

    train_rows: dict[str, list] = {"_id": [], "prompt": [], "answer": [], "base": [], "a": [], "b": []}
    val_rows: dict[str, list] = {"_id": [], "prompt": [], "answer": [], "base": [], "a": [], "b": []}

    # Keep per-base pools for the leakage sanity check at the end.
    base_train_pairs: dict[int, list[tuple[int, int]]] = {}
    base_val_pairs: dict[int, list[tuple[int, int]]] = {}

    for idx, base in enumerate(BASES):
        base_seed = seed + idx
        rng = np.random.default_rng(base_seed)

        # 1. Split this base's pairs into train/val (disjoint).
        indices = np.arange(n_total)
        rng.shuffle(indices)
        train_pairs = [all_pairs[i] for i in indices[:n_train]]
        val_pairs   = [all_pairs[i] for i in indices[n_train:]]

        base_train_pairs[base] = train_pairs
        base_val_pairs[base]   = val_pairs

        # 2. Each pair becomes exactly one row. Few-shot context always drawn from
        #    this base's TRAIN pool (for both train rows and val rows).
        for i, (a, b) in enumerate(train_pairs):
            prompt = build_prompt(a, b, base, train_pairs, rng, few_shot=FEW_SHOT)
            train_rows["_id"].append(f"train-b{base}-{i}")
            train_rows["prompt"].append(prompt)
            train_rows["answer"].append(to_base_answer(a + b, base))
            train_rows["base"].append(base)
            train_rows["a"].append(a)
            train_rows["b"].append(b)

        for i, (a, b) in enumerate(val_pairs):
            prompt = build_prompt(a, b, base, train_pairs, rng, few_shot=FEW_SHOT)
            val_rows["_id"].append(f"val-b{base}-{i}")
            val_rows["prompt"].append(prompt)
            val_rows["answer"].append(to_base_answer(a + b, base))
            val_rows["base"].append(base)
            val_rows["a"].append(a)
            val_rows["b"].append(b)

        print(f"  base={base:2d} (seed={base_seed}): {len(train_pairs):,} train + {len(val_pairs):,} val rows")

    # Shuffle row order within each split so bases are interleaved (not block-ordered).
    shuffle_rng = np.random.default_rng(seed)
    train_perm = shuffle_rng.permutation(len(train_rows["_id"])).tolist()
    val_perm   = shuffle_rng.permutation(len(val_rows["_id"])).tolist()

    train_data = {k: [v[i] for i in train_perm] for k, v in train_rows.items()}
    val_data   = {k: [v[i] for i in val_perm]   for k, v in val_rows.items()}

    train_ds = Dataset.from_dict(train_data)
    val_ds   = Dataset.from_dict(val_data)

    _sanity_check(train_ds, val_ds, base_train_pairs, base_val_pairs, n_train, n_val)

    print(f"Total time: {time.time() - t0:.1f}s")
    return DatasetDict({"train": train_ds, "validation": val_ds})


def _sanity_check(
    train_ds: Dataset,
    val_ds: Dataset,
    base_train_pairs: dict[int, list[tuple[int, int]]],
    base_val_pairs: dict[int, list[tuple[int, int]]],
    n_train: int,
    n_val: int,
) -> None:
    """Fast correctness checks on the generated dataset."""
    print("\nRunning sanity checks…")

    # Per-base train/val pair pools are disjoint.
    for base in BASES:
        train_set = set(base_train_pairs[base])
        val_set   = set(base_val_pairs[base])
        assert train_set.isdisjoint(val_set), f"base {base}: train/val pools overlap — leakage!"
        assert len(train_set) == n_train, f"base {base}: expected {n_train} train pairs, got {len(train_set)}"
        assert len(val_set)   == n_val,   f"base {base}: expected {n_val} val pairs, got {len(val_set)}"
    print("  Per-base train/val pool disjointness + size checks passed.")

    # Exact-row-count checks.
    assert len(train_ds) == len(BASES) * n_train, f"Expected {len(BASES) * n_train} train rows, got {len(train_ds)}"
    assert len(val_ds)   == len(BASES) * n_val,   f"Expected {len(BASES) * n_val} val rows, got {len(val_ds)}"
    print(f"  Row counts: train={len(train_ds):,}, val={len(val_ds):,}")

    # Every (a, b) pair per base appears exactly once across train ∪ val.
    for base in BASES:
        train_pairs_in_ds = {(r["a"], r["b"]) for r in train_ds if r["base"] == base}
        val_pairs_in_ds   = {(r["a"], r["b"]) for r in val_ds   if r["base"] == base}
        assert train_pairs_in_ds == set(base_train_pairs[base]), f"base {base}: train rows don't match train pool"
        assert val_pairs_in_ds   == set(base_val_pairs[base]),   f"base {base}: val rows don't match val pool"
    print("  Exhaustive pair coverage checks passed (every pair used exactly once).")

    # Answer correctness: reversed digit order, unpadded, matches a + b.
    rng0 = np.random.default_rng(0)
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        sample_idx = rng0.integers(len(ds), size=min(500, len(ds)))
        for idx in sample_idx:
            row = ds[int(idx)]
            expected = to_base_answer(row["a"] + row["b"], row["base"])
            assert row["answer"] == expected, (
                f"[{split_name}] row {idx}: a={row['a']} b={row['b']} base={row['base']} "
                f"-> expected {expected!r}, got {row['answer']!r}"
            )
        print(f"  [{split_name}] 500 reversed-answer spot-checks passed.")

    # Val rows' few-shot context never contains a val-pool pair (leakage check).
    # Parse each few-shot line back to (a, b) and confirm it's in the train pool.
    for row in val_ds.select(range(min(1000, len(val_ds)))):
        base = row["base"]
        lines = row["prompt"].split("\n")
        fewshot_lines = lines[:-1]  # last line is the unsolved test problem
        train_set = set(base_train_pairs[base])
        for line in fewshot_lines:
            lhs, _, _ans = line.partition("=")
            a_str, _, b_str = lhs.partition("+")
            a_val = int(a_str, base)
            b_val = int(b_str, base)
            assert (a_val, b_val) in train_set, (
                f"Val row leakage: few-shot pair ({a_val},{b_val}) base {base} "
                f"not in train pool — prompt: {row['prompt']!r}"
            )
    print("  Val few-shot leakage check passed (1000 rows).")

    # Confirm that no few-shot line in a prompt matches the test pair itself.
    # This validates the without-replacement + test-exclusion fix.
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        for row in ds.select(range(min(500, len(ds)))):
            base = row["base"]
            lines = row["prompt"].split("\n")
            fewshot_lines = lines[:-1]
            test_line_prefix = lines[-1].rstrip("=")  # "A+B"
            for line in fewshot_lines:
                lhs = line.partition("=")[0]
                assert lhs != test_line_prefix, (
                    f"[{split_name}] Test pair appears solved in its own few-shot context: {row['prompt']!r}"
                )
    print("  Test-pair self-leakage check passed (500 rows per split).")

    # Few-shot examples within a single prompt are all distinct (no-replacement check).
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        for row in ds.select(range(min(500, len(ds)))):
            lines = row["prompt"].split("\n")
            fewshot_lines = lines[:-1]
            assert len(fewshot_lines) == len(set(fewshot_lines)), (
                f"[{split_name}] Duplicate few-shot lines in prompt: {row['prompt']!r}"
            )
    print("  Without-replacement few-shot uniqueness check passed (500 rows per split).")

    # Prompt format: ends in '=' with nothing after, no spaces anywhere.
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        for row in ds.select(range(min(200, len(ds)))):
            assert row["prompt"].endswith("="), f"[{split_name}] prompt doesn't end with '=': {row['prompt']!r}"
            assert " " not in row["prompt"],    f"[{split_name}] prompt contains a space: {row['prompt']!r}"
    print("  Prompt format (ends at '=', no spaces) checks passed.")

    # Few-shot count: FEW_SHOT solved lines + 1 unsolved test line.
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        for row in ds.select(range(min(100, len(ds)))):
            n_lines = row["prompt"].count("\n") + 1
            assert n_lines == FEW_SHOT + 1, (
                f"[{split_name}] expected {FEW_SHOT + 1} lines, got {n_lines}: {row['prompt']!r}"
            )
    print(f"  Few-shot count (={FEW_SHOT}) checks passed.")

    # Vocabulary coverage (matches the new tokenizer vocab).
    VOCAB = set("0123456789ABCDEF+=\n")
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        for row in ds.select(range(min(200, len(ds)))):
            bad = set(row["prompt"]) - VOCAB
            assert not bad, f"[{split_name}] out-of-vocab chars {bad!r}: {row['prompt']!r}"
    print("  Vocabulary coverage checks passed.")

    print("All sanity checks passed.\n")


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a synthetic multi-base addition dataset and save it locally or to the Hub"
    )
    parser.add_argument("--hub-name",    type=str,   default=None,
                        help="HF Hub dataset repo to push to (e.g. your-username/your-dataset-name). "
                             "If omitted, saved locally to --output-dir instead.")
    parser.add_argument("--output-dir",  type=str,   default="./artifacts/dataset")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--val-holdout", type=float, default=0.5,
                        help="Fraction of pairs per base held out for validation (default 0.5).")
    parser.add_argument("--max-operand", type=int,   default=99,
                        help="Operands drawn from [0, max_operand] inclusive, per base.")
    parser.add_argument("--shard-size",  type=str,   default="500MB")
    args = parser.parse_args()

    print(f"Generating dataset | seed={args.seed} | val_holdout={args.val_holdout} "
          f"| max_operand={args.max_operand} | bases={BASES} | few_shot={FEW_SHOT} | pad_width={PAD_WIDTH}\n")

    dataset = build_dataset(
        seed=args.seed,
        val_holdout=args.val_holdout,
        max_operand=args.max_operand,
    )

    print(dataset)
    print("\nSample rows:")
    for i in range(4):
        row = dataset["train"][i]
        print(f"  base={row['base']:2d}  a={row['a']:3d}  b={row['b']:3d}  answer={row['answer']}")
        print(f"  {row['prompt'][:200]}\n")

    if args.hub_name:
        dataset.push_to_hub(args.hub_name, max_shard_size=args.shard_size)
        print(f"\nPushed dataset to hub: {args.hub_name}")
    else:
        dataset.save_to_disk(args.output_dir)
        print(f"\nSaved dataset to {args.output_dir}  (pass --hub-name to push to the Hub instead)")