"""Download the team's activation captures from the HF bucket (Xet transfer)."""
import os
import sys

from huggingface_hub import download_bucket_files

root = sys.argv[1]
for b in [10, 16, 8, 2]:
    if not os.path.exists(f"{root}/base{b}.pt") or os.path.getsize(f"{root}/base{b}.pt") == 0:
        download_bucket_files("CrossBaseArithmetic/results", files=[(f"base{b}.pt", f"{root}/base{b}.pt")])
    print("done", b, flush=True)
print("ALLDONE", flush=True)
