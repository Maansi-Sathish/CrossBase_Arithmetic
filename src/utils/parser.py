"""Command-line arguments for the inference / activation-extraction entry point (src/main.py)."""

import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add inference command-line arguments to the parser."""

    # --- Generic run arguments ---
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="HuggingFace repo or local path to the model.",
    )
    parser.add_argument(
        "--layers", "-l",
        nargs="+",
        type=int,
        required=False,
        help="List of layer indices to analyze. Defaults to all decoder layers.",
    )
    parser.add_argument(
        "--num-prompts", "-p",
        default=1000,
        type=int,
        help="Number of prompts to evaluate on. Defaults to 1000.",
    )
    parser.add_argument(
        "--max-new-tokens", "-mnt",
        type=int,
        required=False,
        default=16,
        help="Maximum number of new tokens to generate. Defaults to 16.",
    )
    parser.add_argument(
        "--output", "-out",
        type=str,
        help="Full output path for the results file. If omitted, auto-generated.",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        required=False,
        default=42,
        help="Random seed for reproducibility. Defaults to 42.",
    )

    # --- Core mechinterp toggles ---
    parser.add_argument(
        "--capture-geometry",
        action="store_true",
        help="Capture detailed per-position activations (MLP neurons, attention heads, "
             "residual stream). Omit for lightweight runs such as large ablation sweeps.",
    )
    parser.add_argument(
        "--intervention",
        type=str,
        default=None,
        help="Path to an intervention spec JSON (neurons/heads to ablate or patch). "
             "When provided, an intervention pass is run after the baseline.",
    )

    # --- Task-specific argument ---
    parser.add_argument(
        "--base", "-b",
        type=int,
        required=True,
        choices=[2, 8, 10, 16],
        help="Which numeral base to run prompts for. One of 2, 8, 10, 16. "
             "Run separately for each base to compare circuits across bases.",
    )