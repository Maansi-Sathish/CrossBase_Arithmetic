# Base Adder Trace

An interactive page for students: type two numbers (0–999) and a base, and the trained `arithmetic-model-L2_H4_D64` adds them in the browser. You then step through how it does it: finding the base, fetching each column's digits, the carry, the neurons' vote, the scores and stopping.

- `base-adder-trace.html`: the built page. Open it directly in a browser.
- `page.html`: layout, text and the step views.
- `engine.js`: the model's forward pass in JavaScript, plus the trace and attribution helpers.
- `build.py`: exports the weights and assembles the page (`python -I experiments/mechanism/webtool/build.py` from the repository root).

Every value is computed live from the weights. The carry threshold (about half the base) and the stop windows are rules from `../REPORT.md`, shown as summaries.
