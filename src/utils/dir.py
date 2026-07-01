"""Build the output path/filename for saved inference results.

`generate_output_path` has a sensible default (model name + #prompts), so the pipeline
runs as-is. Customize it to encode more of your run's parameters into the filename so
different runs don't overwrite each other. A small worked example is in comments.
"""

import argparse


def generate_output_path(args: argparse.Namespace) -> str:
    """Generate the output path for a run's results.

    If `args.output` is set, it is honored directly. Otherwise, a filename is built
    from the run parameters in `args`.

    Filename encodes:
        m     = model name (slashes replaced with --)
        p     = number of prompts
        b     = bases used (e.g. 2-8-10-16 or just 2 if base_filter is set)
        mo    = max operand
        fs    = few shot examples
        int   = whether an intervention was run
        geo   = whether geometry was captured

    This means every unique combination of run parameters gets its own file,
    so runs never overwrite each other.

    Args:
        args (argparse.Namespace): The parsed command-line arguments.

    Returns:
        str: The output path for the run's results.
    """
    if args.output is not None:
        return args.output

    # Model name — replace / with -- so HF repo names don't create subdirectories
    # e.g. CrossBaseArithmetic/arithmetic-model-L2_H8_D128 -> CrossBaseArithmetic--arithmetic-model-L2_H8_D128
    model_name = str(args.model_path).replace("/", "--")

    # Which bases are active — use base_filter if set, otherwise join all bases
    # e.g. base_filter=2 -> "2", bases=[2,8,10,16] -> "2-8-10-16"
    base_filter = getattr(args, "base_filter", None)
    bases = getattr(args, "bases", [2, 8, 10, 16])
    if base_filter is not None:
        base_str = str(base_filter)
    else:
        base_str = "-".join(str(b) for b in sorted(bases))

    # Max operand — tells you the difficulty/range of the run
    max_operand = getattr(args, "max_operand", 999)

    # Few shot count — affects how much context the model sees
    few_shot = getattr(args, "few_shot_examples", 5)

    # Whether intervention (ablation) was run
    intervention = getattr(args, "intervention", None) is not None

    # Whether geometry (per-position activations) was captured
    geo = getattr(args, "capture_geometry", False)

    # Build filename from all parameters
    components = [
        f"[m={model_name}]",
        f"[p={args.num_prompts}]",
        f"[b={base_str}]",
        f"[mo={max_operand}]",
        f"[fs={few_shot}]",
        f"[int={intervention}]",
        f"[geo={geo}]",
    ]

    return "_".join(components) + ".pt"