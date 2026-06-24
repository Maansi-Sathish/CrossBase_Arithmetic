import os
import json
import subprocess
import sys

COMPLETED_FILE = "completed_runs.json"

def load_completed():
    if os.path.exists(COMPLETED_FILE):
        with open(COMPLETED_FILE) as f:
            return set(json.load(f))
    return set()

def mark_completed(run_name):
    completed = load_completed()
    completed.add(run_name)
    with open(COMPLETED_FILE, "w") as f:
        json.dump(list(completed), f)

# Full grid
LAYERS = [1, 2, 3, 4]
HEADS  = [2, 4, 8, 16]
DIMS   = [32, 64, 128, 256]

completed = load_completed()

for layers in LAYERS:
    for heads in HEADS:
        for dim in DIMS:
            if dim % heads != 0:
                print(f"Skipping L{layers}_H{heads}_D{dim} — not divisible")
                continue

            run_name = f"L{layers}_H{heads}_D{dim}"

            if run_name in completed:
                print(f"Already done: {run_name}, skipping.")
                continue

            print(f"\nStarting: {run_name}")
            result = subprocess.run([
                "python", "train_model.py",
                "--layers", str(layers),
                "--heads", str(heads),
                "--dim", str(dim),
            ])

            if result.returncode == 0:
                mark_completed(run_name)
                print(f"Completed: {run_name}")
            else:
                print(f"FAILED: {run_name} — stopping sweep")
                sys.exit(1)