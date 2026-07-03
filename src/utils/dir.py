"""Build the output path/filename for saved inference results."""

import argparse
from pathlib import Path


def generate_output_path(args: argparse.Namespace) -> str:
    """Generate the output path for a run's results.

    If args.output is set, it is honored directly. Otherwise a filename is
    built from the run parameters so different runs never overwrite each other.

    Filename encodes:
        m    = model name (slashes replaced with --)
        p    = number of prompts
        b    = base used
        int  = whether intervention was run
        geo  = whether geometry was captured

    Args:
        args: The parsed command-line arguments.

    Returns:
        str: The output path for the run's results.
    """
    if args.output is not None:
        return args.output
    results_dir = Path("results")

    # Model name — replace / with -- so HF repo names don't create subdirectories
    model_name = str(args.model_path).replace("/", "--")

    # Single base — passed as --base 2, --base 8 etc.
    base = getattr(args, "base", "unknown")

    # Whether intervention (ablation) was run
    intervention = getattr(args, "intervention", None) is not None

    # Whether geometry (per-position activations) was captured
    geo = getattr(args, "capture_geometry", False)

    components = [
        f"[m={model_name}]",
        f"[p={args.num_prompts}]",
        f"[b={base}]",
        f"[int={intervention}]",
        f"[geo={geo}]",
    ]

    return str(results_dir / ("_".join(components) + ".pt"))