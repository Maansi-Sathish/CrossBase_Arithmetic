"""Minimal tests for find_answer_span and find_positions_of_interest.

Run from the project root:
    python test_inference_functions.py
"""

import sys
import torch

sys.path.insert(0, "src")
from inference import find_answer_span, find_positions_of_interest
from model import load_model

MODEL_PATH = "CrossBaseArithmetic/arithmetic-model-L2_H4_D64"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from '{MODEL_PATH}'...")
model = load_model(MODEL_PATH, device)

# ------------------------------------------------------------------ #
# Test find_answer_span
# ------------------------------------------------------------------ #
print("\n=== find_answer_span ===")
cases = [
    ("binary",               "11000",  "11000"),
    ("octal",                "30",     "30"),
    ("decimal",              "89",     "89"),
    ("decimal leading zero", "011",    "011"),
    ("hex letters",          "AB",     "AB"),
    ("hex mixed",            "F7",     "F7"),
    ("leading newline",      "\n89",   "89"),
    ("leading space",        " 89",    "89"),
    ("empty",                "",       None),
    ("garbage",              "hello",  None),
    ("lowercase",            "ab",     None),
]

for desc, inp, expected in cases:
    result = find_answer_span(inp)
    got = result[0] if result else None
    status = "OK" if got == expected else "FAIL"
    print(f"  [{status}] {desc:25s} input={inp!r:10s} -> {got!r} (expected {expected!r})")

# ------------------------------------------------------------------ #
# Test find_positions_of_interest
# ------------------------------------------------------------------ #
print("\n=== find_positions_of_interest ===")
prompts = {
    2:  ("1101+1011=11000\n11001+1110=100111\n101111+100000=1001111\n1000+1011=10011\n110+101=1011\n11010+101=",
         "11010", "101"),
    8:  ("15+13=03\n31+16=74\n57+40=711\n10+13=32\n70+27=711\n24+20=",
         "24", "20"),
    10: ("13+11=42\n25+14=93\n47+32=97\n8+91=99\n56+23=97\n46+52=",
         "46", "52"),
    16: ("D+B=81\n19+E=72\n5F+4C=BA\n32+1F=15\n7A+6=08\n2B+14=",
         "2B", "14"),
}

for base, (prompt, op_a, op_b) in prompts.items():
    pos = find_positions_of_interest(model, prompt)
    toks = model.to_str_tokens(prompt, prepend_bos=False)
    expected = {
        "first_digit_a": op_a[0],
        "last_digit_a":  op_a[-1],
        "operator":      "+",
        "first_digit_b": op_b[0],
        "last_digit_b":  op_b[-1],
        "eq_sign":       "=",
    }
    results = {k: toks[pos[k]] if pos.get(k) is not None else None for k in expected}
    ok = all(results[k] == expected[k] for k in expected)
    print(f"  [{'OK' if ok else 'FAIL'}] base {base:2d}: {results}")

print("\nAll tests done!")