"""Build the Base Adder Trace page: export the model's weights and inline them, with the engine, into one HTML file.

Run from the repository root:  python -I experiments/mechanism/webtool/build.py
Output: experiments/mechanism/webtool/base-adder-trace.html (open it in a browser; no server needed).
The page runs the model's forward pass in JavaScript (engine.js); it matched PyTorch's logits to within 5e-5.
"""
import base64, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import Model  # noqa: E402

m = Model(device="cpu")
tensors = {k: {"shape": list(v.shape), "b64": base64.b64encode(v.float().contiguous().numpy().astype("<f4").tobytes()).decode()} for k, v in m.sd.items()}
weights = "const MODEL_WEIGHTS = " + json.dumps({"vocab": m.vocab, "tensors": tensors}) + ";\n"
page = (HERE / "page.html").read_text().replace("/*__ENGINE__*/", (HERE / "engine.js").read_text()).replace("/*__WEIGHTS__*/", weights)
(HERE / "base-adder-trace.html").write_text("<!doctype html>\n" + page)
print("wrote", HERE / "base-adder-trace.html")
