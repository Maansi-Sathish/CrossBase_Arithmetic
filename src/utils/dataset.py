"""Prompt dataset for inference: builds the prompts that main.py runs the model on.

Prompts follow exactly the same format the model was trained on (see
src/train/create_dataset.py): few_shot_examples solved equations, then the
test question, answered in reversed-digit order (LSD first) via to_base_answer.
No base-indicator token — the model infers the base from the digit characters
in the few-shot context, exactly as during training.
"""

import random


# --------------------------------------------------------------------------- #
# Base conversion helpers — copied exactly from src/train/create_dataset.py
# so prompt format is byte-for-byte identical to training data
# --------------------------------------------------------------------------- #

def to_base(n: int, base: int) -> str:
    """Convert a non-negative integer to its string representation in the given base.
    Uses uppercase A-F for hex. No padding (PAD_WIDTH=0 matches training default).
    """
    if n == 0:
        return "0"
    digits = []
    while n > 0:
        digits.append("0123456789ABCDEF"[n % base])
        n //= base
    return "".join(reversed(digits))


def to_base_answer(n: int, base: int) -> str:
    """Answer in reversed digit order, unpadded — the Lee et al. / training convention.
    Strips any leading zeros before reversing so '042' doesn't become '240'.
    """
    fwd = to_base(n, base).lstrip("0") or "0"
    return fwd[::-1]  # reverse: LSD first


def render_example(a: int, b: int, base: int) -> str:
    """One solved shot: 'A+B=reversed_answer' — no spaces, matches training format."""
    return f"{to_base(a, base)}+{to_base(b, base)}={to_base_answer(a + b, base)}"


# --------------------------------------------------------------------------- #
# PromptDataset
# --------------------------------------------------------------------------- #

class PromptDataset:
    """A collection of prompts to run activation-extraction inference on.

    `inference.run` iterates over `self.prompts`, a list of {"prompt": str, "metadata": dict}
    entries. "prompt" is the string fed to the model; "metadata" carries ground truth
    (base, a, b, answer) so is_correct in main.py can score the output.
    """

    def __init__(self) -> None:
        """Initialize an empty dataset."""
        self.prompts: list[dict] = []

    def __len__(self) -> int:
        """Return the number of prompts in the dataset."""
        return len(self.prompts)

    @classmethod
    def generate_prompts(
        cls,
        num_prompts: int,
        base: int,
        max_operand: int = 999,
        few_shot_examples: int = 5,
    ) -> "PromptDataset":
        """Build prompts for a single base, matching training format exactly.

        We always run N prompts for one base at a time — base is fixed at
        runtime via --base CLI argument. To compare circuits across bases,
        run the script separately for each base.

        Args:
            num_prompts:       How many prompts to generate (from --num-prompts).
            base:              Which base to generate prompts for (from --base).
                               One of 2, 8, 10, 16.
            max_operand:       Operands sampled from [0, max_operand].
                               Hardcoded to 999 matching training.
            few_shot_examples: Number of solved examples shown before the question.
                               Hardcoded to 5 matching training.

        Returns:
            A PromptDataset whose .prompts is a list of num_prompts
            {"prompt": str, "metadata": dict} entries, all in the given base.
        """
        instance = cls()

        for _ in range(num_prompts):
            # Sample the test pair (a, b)
            a = random.randint(0, max_operand)
            b = random.randint(0, max_operand)

            # Build few-shot examples — no duplicates, test pair excluded
            # Mirrors create_dataset.py's without-replacement sampling logic
            seen = {(a, b)}
            shots = []
            for _ in range(few_shot_examples):
                while True:
                    fa = random.randint(0, max_operand)
                    fb = random.randint(0, max_operand)
                    if (fa, fb) not in seen:
                        break
                seen.add((fa, fb))
                shots.append(render_example(fa, fb, base))

            # Full prompt: shots joined by \n, then test question ending at =
            question = f"{to_base(a, base)}+{to_base(b, base)}="
            prompt = "\n".join(shots) + "\n" + question

            # Answer in reversed-digit order — what the model outputs
            answer = to_base_answer(a + b, base)

            instance.prompts.append({
                "prompt": prompt,
                "metadata": {
                    "base": base,
                    "a": a,
                    "b": b,
                    "answer": answer,
                },
            })

        return instance