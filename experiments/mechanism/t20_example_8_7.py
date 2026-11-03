"""T20: the worked example 8 + 7 in each base. Greedy output, and at each step: top tokens, L1 head attention targets,
the carry evidence, and the stop margin."""
import random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, torch
from core import DEVICE, Model, Problem, render, sample_shots, save_json, to_answer, to_base
torch.set_grad_enabled(False)
model = Model(); inv = {v: k for k, v in model.vocab.items()}; res = {}
for b in (2, 8, 10, 16):
    outs = []
    for seed in range(20):
        p = Problem(8, 7, b, sample_shots(random.Random(seed), (8, 7)))
        s = p.prompt; steps = []
        for step in range(6):
            seqs = [model.vocab[x] for x in s]; X = torch.tensor([seqs], device=DEVICE); lg, c = model.forward(X, capture={"pattern1", "pattern0"})
            l = lg[0, -1]; top = torch.topk(l, 3); nxt = inv[int(top.indices[0])]
            st = s.rfind("\n") + 1; line = s[st:]
            att = {}
            for h in range(4):
                r = c["pattern1"][0, h, -1, st:]; j = int(r.argmax()); att[f"L1H{h}"] = (line[j] if line[j] != "\n" else "\\n", j, round(float(r[j]), 2))
            steps.append({"text so far": line, "top": [(inv[int(i)], round(float(v), 1)) for v, i in zip(top.values, top.indices)], "L1 head strongest target (token, index in line, weight)": att})
            if nxt == "<eos>" or nxt == "\n": break
            s += nxt
        outs.append((s[s.rfind("=") + 1:], steps))
    answers = [o[0] for o in outs]; correct = to_answer(15, b)
    res[b] = {"prompt line": f"{to_base(8, b)}+{to_base(7, b)}=", "correct (reversed digits)": correct, "share of 20 contexts correct": sum(a == correct for a in answers) / 20, "example steps": outs[0][1]}
    print(f"base {b}: {res[b]['prompt line']} -> model {answers[0]!r}, correct {correct!r}, 20 contexts: {res[b]['share of 20 contexts correct']}")
    for st_ in outs[0][1]: print("   ", st_["text so far"], st_["top"], st_["L1 head strongest target (token, index in line, weight)"])
print("example context:", repr(Problem(8, 7, 10, sample_shots(random.Random(0), (8, 7))).prompt[:60]))
save_json("t20_example_8_7", res); print("DONE")
