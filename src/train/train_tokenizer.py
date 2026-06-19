"""Build a character-level tokenizer and save it (locally by default, or push to the HF Hub).

This builds a deterministic, training-free WordLevel tokenizer that maps each
character in `VOCAB_CHARS` to its own token. This pattern is useful for small
from-scratch models on synthetic tasks where the input alphabet is small and fixed.

TODO (you must implement this for your task):
  1. `VOCAB_CHARS`  -- every character that can appear in a prompt or answer.
  2. `SAMPLE_TEXTS` -- a few representative strings for the round-trip sanity
     checks at the bottom of this script.

By default the tokenizer is saved locally to --output-dir; pass --hub-name to push it to
the HF Hub instead.

See the commented-out example below for a small self-contained illustration
(a single-digit arithmetic task).
"""

import argparse

from tokenizers import Regex, Tokenizer
from tokenizers.decoders import Fuse
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast

# --------------------------------------------------------------------------- #
# TODO: fill these two lists in for your task.
#
#   VOCAB_CHARS:  EVERY character that can appear in any prompt or answer your model
#                 will ever see -- digits, operators, letters, spaces, newline, etc.
#                 The tokenizer gives each one its own token id. Anything you forget to
#                 list becomes the <unk> token, so be exhaustive. Use sorted(...) so the
#                 vocabulary order is deterministic across machines.
#
#   SAMPLE_TEXTS: a handful of representative strings (using only VOCAB_CHARS). The block
#                 at the bottom of this file encodes then decodes each one and checks it
#                 round-trips exactly -- a quick sanity check that your vocab is complete.
# --------------------------------------------------------------------------- #
#
# Example (a single-digit addition task with prompts like "7+5="):
#
#     VOCAB_CHARS = sorted("0123456789+=\n")
#     SAMPLE_TEXTS = [
#         "2+3=5\n",
#         "7+5=12\n",
#         "1+1=2\n4+0=4\n2+6=8\n6+4=10\n5+6=11\n8+1=9\n9+8=17\n3+3=",
#     ]
#
# Our task: addition problems across four numeral bases (binary, octal,
# decimal, hexadecimal) sharing one vocabulary. There is no per-base marker
# token -- the model has to infer the base purely from which digit
# characters it sees, so every base's digits must already be in this set.
# Digits 0-9 cover binary/octal/decimal; A-F additionally cover hex.
# '+' and '=' are the operator/result markers, ',' separates few-shot
# examples within a prompt (comma-delimited style), '\n' separates them
# in the newline-delimited style used during training, and ' ' is kept
# for any prompt that exercises explicit spacing.

VOCAB_CHARS: list[str] = sorted("0123456789ABCDEF+=,\n ")
SAMPLE_TEXTS: list[str] = [
    "13+11=24",                               # decimal
    "1101+1011=11000",                        # binary
    "15+13=30",                               # octal
    "D+B=18",                                 # hex
    "9F+7C=11B",                              # hex: exercises 9, 7, C
    "A+6=10",                                 # hex: exercises A
    "E+1=F",                                  # hex: exercises E, F
    "13 + 11 = 24",                           # exercises space
    "13+11=24,15+13=30",                      # exercises comma
    # few-shot style: \n-delimited, unanswered test problem at the end
    "13+11=24\n25+14=39\n47+32=79\n8+91=99\n56+23=79\n10+11=21\n33+44=77\n62+15=77\n18+20=",   # decimal
    "1101+1011=11000\n11001+1110=100111\n101111+100000=1001111\n1000+1011=10011\n110+101=1011\n11111+1=100000\n10101+1010=11111\n1001+110=1111\n11010+101=",  # binary
    "15+13=30\n31+16=47\n57+40=117\n10+13=23\n70+27=117\n12+55=67\n43+54=117\n76+11=107\n24+20=",  # octal
    "D+B=18\n19+E=27\n5F+4C=AB\n32+1F=51\n7A+6=80\nCA+F=D9\nC3+2D=F0\n48+39=81\n2B+14=",        # hex
]
# --------------------------------------------------------------------------- #

_SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]
_VOCAB = {tok: i for i, tok in enumerate(_SPECIAL_TOKENS + VOCAB_CHARS)}


def build_tokenizer() -> PreTrainedTokenizerFast:
    """Build a deterministic character-level tokenizer over VOCAB_CHARS.

    A tokenizer maps text <-> integer token IDs. Here we use the simplest possible scheme:
    one token per character (a WordLevel vocab plus a Split pre-tokenizer that isolates every
    single character). This is ideal for small synthetic tasks because it's transparent and has
    a tiny vocabulary -- no subword merges to reason about. The Fuse decoder just glues the
    characters back together when decoding.
    """
    tok_obj = Tokenizer(WordLevel(vocab=_VOCAB, unk_token="<unk>"))
    tok_obj.pre_tokenizer = Split(pattern=Regex(r"[\s\S]"), behavior="isolated")
    tok_obj.decoder = Fuse()
    # Wrap in a PreTrainedTokenizerFast so it behaves like any HuggingFace tokenizer (padding,
    # special tokens, save/push_to_hub, etc.).
    return PreTrainedTokenizerFast(
        tokenizer_object=tok_obj,
        bos_token="<bos>",  # beginning-of-sequence
        eos_token="<eos>",  # end-of-sequence (the model learns to emit this to stop)
        pad_token="<pad>",  # fills short sequences so a batch is rectangular
        unk_token="<unk>",  # stands in for any character not in VOCAB_CHARS
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a character-level tokenizer and save it locally or to the Hub")
    parser.add_argument(
        "--hub-name",
        type=str,
        default=None,
        help="HF Hub repo to push the tokenizer to (e.g. your-username/your-project-tokenizer). "
        "If omitted, the tokenizer is saved locally to --output-dir instead (no login needed).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./artifacts/tokenizer",
        help="Local directory to save the tokenizer to when --hub-name is not given.",
    )
    args = parser.parse_args()

    if not VOCAB_CHARS or not SAMPLE_TEXTS:
        raise ValueError(
            "VOCAB_CHARS and SAMPLE_TEXTS are not set. "
            "Fill these in for your task -- see the module docstring for an example."
        )

    tokenizer = build_tokenizer()
    vocab = tokenizer.get_vocab()

    # Vocabulary structure
    print(f"Vocabulary ({len(tokenizer)} tokens):")
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        print(f"  {idx:2d}: {repr(token)}")

    expected_vocab_size = len(_SPECIAL_TOKENS) + len(VOCAB_CHARS)
    assert len(tokenizer) == expected_vocab_size, f"Expected {expected_vocab_size} tokens, got {len(tokenizer)}"

    # Special token IDs match _SPECIAL_TOKENS insertion order
    assert tokenizer.pad_token_id == 0, f"pad_token_id={tokenizer.pad_token_id}, expected 0"
    assert tokenizer.bos_token_id == 1, f"bos_token_id={tokenizer.bos_token_id}, expected 1"
    assert tokenizer.eos_token_id == 2, f"eos_token_id={tokenizer.eos_token_id}, expected 2"
    assert vocab["<unk>"] == 3, f"unk id={vocab['<unk>']}, expected 3"

    # pad and eos must be distinct, otherwise DataCollatorForLanguageModeling masks out EOS during training
    assert tokenizer.pad_token_id != tokenizer.eos_token_id, "pad_token_id must differ from eos_token_id"

    # All token IDs are within [0, vocab_size)
    assert max(vocab.values()) == len(tokenizer) - 1, "Token IDs are not contiguous"

    # Every vocab character has its own entry and no two characters share an ID
    char_ids = [vocab[c] for c in VOCAB_CHARS]
    assert len(char_ids) == len(set(char_ids)), "Duplicate IDs among vocab characters"

    print("\nVocabulary assertions passed.")

    # Round-trip sanity checks on SAMPLE_TEXTS. For a character-level
    # tokenizer, each token should be exactly one input character.
    print("\nSanity checks:")
    for text in SAMPLE_TEXTS:
        ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(ids)
        print(f"  {text!r:16s} -> {tokens}")

        assert tokens == list(text), (
            f"Tokenization mismatch for {text!r}:\n  expected: {list(text)}\n  got:      {tokens}"
        )
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        assert decoded == text, f"Round-trip mismatch for {text!r}:\n  expected: {text!r}\n  got:      {decoded!r}"
        assert tokenizer.unk_token_id not in ids, f"Unknown token in encoding of {text!r}: ids={ids}"
        assert all(0 <= i < len(tokenizer) for i in ids), f"Out-of-bounds token ID in encoding of {text!r}: ids={ids}"

    print("\nAll tokenization assertions passed.")

    if args.hub_name:
        tokenizer.push_to_hub(args.hub_name)
        print(f"\nPushed tokenizer to hub: {args.hub_name}")
    else:
        tokenizer.save_pretrained(args.output_dir)
        print(f"\nSaved tokenizer to {args.output_dir}  (pass --hub-name to push to the Hub instead)")