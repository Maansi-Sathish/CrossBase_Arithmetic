"""Generate a synthetic training dataset and save it (locally by default, or push to the HF Hub).

Task: addition in four numeral bases (binary, octal, decimal, hexadecimal) simultaneously,
with a shared character-level tokenizer. Each prompt is FEW_SHOT solved examples followed
by an unsolved test problem. No base-indicator token is included — the model must infer the
base from the digit characters in the few-shot context.

This version implements the exhaustive-pair design (not resampling-with-replacement):
  - For each base, all 10,000 (a, b) pairs over [0, 99] x [0, 99] are enumerated.
  - Each base's pairs are split 50/50 -> 5,000 train + 5,000 val, independently per base.
  - Every (a, b) pair becomes exactly one row (no duplicates, no oversampling).
  - Total: 4 bases x 5,000 = 20,000 train rows, 20,000 val rows.

Answer convention (Lee et al.): operands are written forward (normal digit order), but the
answer is written in REVERSED digit order, unpadded. E.g. decimal 46+52=98: the operands stay
forward, but the answer "98" is reversed to "89", so the training row is "46+52=89". Do not
reverse the operands, only the answer -- see to_base_answer below for the precise rule.

Whitespace: no spaces around operators (tokenizer now strips them). Few-shot examples are
joined with "\n", matching the tokenizer's vocabulary (comma is no longer the separator).

Leakage-safety: each base's pairs are split into disjoint train/val pools before any prompts
are built. Few-shot examples for BOTH train and val rows are drawn only from that base's
train pool, so a val pair never appears anywhere in a training context.
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
MAX_OPERAND = 100  # operands drawn from [0, 99] inclusive


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
    """Build one few-shot prompt ending at '=' (model must produce the answer).

    Few-shot pool is `fs_pairs` (always the base's TRAIN pool, even for val rows).
    No carry-related sampling -- few-shot examples are drawn uniformly at random.

    Returns a string like:
        "A+B=C\nD+E=F\n...\nX+Y="
    where the final test pair is unsolved and the prompt ends immediately after '='.
    """
    idxs = rng.integers(0, len(fs_pairs), size=few_shot)
    fs_strs = [render_example(fs_pairs[i][0], fs_pairs[i][1], base) for i in idxs]
    solved = "\n".join(fs_strs)
    test_q = f"{to_base(test_a, base)}+{to_base(test_b, base)}="
    return f"{solved}\n{test_q}"


# --------------------------------------------------------------------------- #
# Core dataset builder
# --------------------------------------------------------------------------- #

def build_dataset(seed: int) -> DatasetDict:
    """Build the train/validation DatasetDict for multi-base addition.

    Exhaustive design: every (a, b) pair in [0, 99] x [0, 99] is used exactly once,
    per base. Each base's 10,000 pairs are split 50/50 into train/val (5,000 each),
    using seed + idx as that base's split seed (idx = BASES.index(base)).

    Args:
        seed: Master random seed. Each base derives its own split/few-shot seed as
              seed + idx, where idx is the base's position in BASES.

    Returns:
        DatasetDict with "train" and "validation" splits. Columns:
            prompt  - the full few-shot prompt ending at '='
            answer  - the correct answer string, reversed-digit, in the appropriate base
            base    - integer base (2, 8, 10, or 16)
            a       - first operand as a decimal integer
            b       - second operand as a decimal integer
    """
    t0 = time.time()

    all_pairs = [(a, b) for a in range(MAX_OPERAND) for b in range(MAX_OPERAND)]  # 10,000 pairs

    train_rows: dict[str, list] = {"_id": [], "prompt": [], "answer": [], "base": [], "a": [], "b": []}
    val_rows: dict[str, list] = {"_id": [], "prompt": [], "answer": [], "base": [], "a": [], "b": []}

    # Keep per-base pools around for the leakage sanity check at the end.
    base_train_pairs: dict[int, list[tuple[int, int]]] = {}
    base_val_pairs: dict[int, list[tuple[int, int]]] = {}

    for idx, base in enumerate(BASES):
        base_seed = seed + idx
        rng = np.random.default_rng(base_seed)

        # 1. Split this base's 10,000 pairs 50/50 into train/val.
        indices = np.arange(len(all_pairs))
        rng.shuffle(indices)
        half = len(indices) // 2
        train_pairs = [all_pairs[i] for i in indices[:half]]
        val_pairs = [all_pairs[i] for i in indices[half:]]

        base_train_pairs[base] = train_pairs
        base_val_pairs[base] = val_pairs

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
    val_perm = shuffle_rng.permutation(len(val_rows["_id"])).tolist()

    train_data = {k: [v[i] for i in train_perm] for k, v in train_rows.items()}
    val_data = {k: [v[i] for i in val_perm] for k, v in val_rows.items()}

    train_ds = Dataset.from_dict(train_data)
    val_ds = Dataset.from_dict(val_data)

    _sanity_check(train_ds, val_ds, base_train_pairs, base_val_pairs)

    print(f"Total time: {time.time() - t0:.1f}s")
    return DatasetDict({"train": train_ds, "validation": val_ds})


def _sanity_check(
    train_ds: Dataset,
    val_ds: Dataset,
    base_train_pairs: dict[int, list[tuple[int, int]]],
    base_val_pairs: dict[int, list[tuple[int, int]]],
) -> None:
    """Fast correctness checks on the generated dataset."""
    print("\nRunning sanity checks…")

    # Per-base train/val pair pools are disjoint.
    for base in BASES:
        train_set = set(base_train_pairs[base])
        val_set = set(base_val_pairs[base])
        assert train_set.isdisjoint(val_set), f"base {base}: train/val pools overlap — leakage!"
        assert len(train_set) == 5000, f"base {base}: expected 5000 train pairs, got {len(train_set)}"
        assert len(val_set) == 5000, f"base {base}: expected 5000 val pairs, got {len(val_set)}"
    print("  Per-base train/val pool disjointness + size checks passed.")

    # Exact-row-count checks.
    assert len(train_ds) == len(BASES) * 5000, f"Expected {len(BASES) * 5000} train rows, got {len(train_ds)}"
    assert len(val_ds) == len(BASES) * 5000, f"Expected {len(BASES) * 5000} val rows, got {len(val_ds)}"
    print(f"  Row counts: train={len(train_ds):,}, val={len(val_ds):,}")

    # Every (a, b) pair per base appears exactly once across train ∪ val.
    for base in BASES:
        train_pairs_in_ds = {(r["a"], r["b"]) for r in train_ds if r["base"] == base}
        val_pairs_in_ds = {(r["a"], r["b"]) for r in val_ds if r["base"] == base}
        assert train_pairs_in_ds == set(base_train_pairs[base]), f"base {base}: train rows don't match train pool"
        assert val_pairs_in_ds == set(base_val_pairs[base]), f"base {base}: val rows don't match val pool"
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
    # We can't directly inspect which pairs were drawn for few-shot from the saved
    # prompt alone in general, but we CAN check that none of the rendered few-shot
    # example substrings correspond to an answer that only a val-pool pair could have
    # produced where the (a,b) match a val pair AND the rendered example text appears.
    # A simpler, sufficient check: re-render with the same logic is not reproducible
    # post-hoc (rng state), so instead verify structurally that val rows' few-shot
    # lines, when parsed back to (a, b, base), are all members of that base's train pool.
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

    # Prompt format: ends in '=' with nothing after, no spaces anywhere.
    for split_name, ds in [("train", train_ds), ("validation", val_ds)]:
        for row in ds.select(range(min(200, len(ds)))):
            assert row["prompt"].endswith("="), f"[{split_name}] prompt doesn't end with '=': {row['prompt']!r}"
            assert " " not in row["prompt"], f"[{split_name}] prompt contains a space: {row['prompt']!r}"
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
    parser = argparse.ArgumentParser(description="Generate the multi-base addition dataset and save it locally or to the Hub")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HF Hub dataset repo to push to (e.g. your-username/your-dataset-name). "
        "If omitted, the dataset is saved locally to --output-dir instead (no login needed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/dataset",
        help="Local directory to save the dataset to when --hub-name is not given.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Master seed. Each base derives its split/few-shot seed as seed + idx (idx = position in BASES).",
    )
    parser.add_argument("--shard-size", type=str, default="500MB", help="Max shard size per Parquet file on the Hub")
    args = parser.parse_args()

    print(f"Generating dataset | seed={args.seed} | bases={BASES} | few_shot={FEW_SHOT} | pad_width={PAD_WIDTH}\n")

    dataset = build_dataset(seed=args.seed)

    print(dataset)
    print("\nSample rows:")
    for i in range(4):
        row = dataset["train"][i]
        print(f"  base={row['base']:2d}  a={row['a']:3d}  b={row['b']:3d}  answer={row['answer']}")
        print(f"         {row['prompt']}")

    if args.hub_name:
        dataset.push_to_hub(args.hub_name, max_shard_size=args.shard_size)
        print(f"\nPushed dataset to hub: {args.hub_name}")
    else:
        dataset.save_to_disk(args.output_dir)
        print(f"\nSaved dataset to {args.output_dir}  (pass --hub-name to push to the Hub instead)")