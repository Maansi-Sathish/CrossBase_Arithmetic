"""Shared infrastructure for the mechanism experiments.

Contains:
  * ``Model``    -- a minimal, batched, hookable re-implementation of the trained GPT-2. Verified
                    against the HuggingFace model in ``e00_setup.py``. Much faster than the
                    per-prompt TransformerBridge loop, and it exposes attention patterns directly.
  * ``Problem``  -- one addition problem with the high-level variables a schoolbook algorithm uses.
  * ``make_batch`` -- tokenises problems and records the semantic position of every test-line token.

Terminology used everywhere:
  * digit index k counts from the least-significant digit: a_0 is the units digit of a.
  * the answer is written LSD first, so the model emits digit k at query position q_k = p_eq + k
    (q_0 is the '=' token; q_1 is the answer's first emitted digit; ...).
  * at q_k a schoolbook algorithm needs a_k, b_k and carry-in c_k; it computes the raw digit sum
    s_k = a_k + b_k + c_k, emits o_k = s_k mod base and passes on c_{k+1} = (s_k >= base).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

MODEL_ID = "CrossBaseArithmetic/arithmetic-model-L2_H4_D64"
DATASET_ID = "CrossBaseArithmetic/multi-base-addition-999"
BASES = [2, 8, 10, 16]
DIGITS = "0123456789ABCDEF"
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
FIGS = ROOT / "figures"
BUCKET = ROOT.parent.parent / "results" / "bucket"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fixed categorical order for bases (dataviz reference palette slots 1-4). Colour follows the base.
BASE_COLOR = {2: "#2a78d6", 8: "#eb6834", 10: "#1baf7a", 16: "#eda100"}
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"


# --------------------------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------------------------- #
def gelu_new(x: torch.Tensor) -> torch.Tensor:
    """GPT-2's tanh approximation of GELU (HF ``gelu_new``)."""
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def _fetch(model_id: str, name: str) -> str:
    """A model file from a local directory if model_id is one, otherwise from the Hub."""
    if Path(model_id).is_dir():
        return str(Path(model_id) / name)
    return hf_hub_download(model_id, name)


class Model:
    """Functional GPT-2 forward pass with named capture and intervention points.

    Hook names:
      resid_pre{L}, resid_mid{L}, resid_post{L}   [batch, pos, d_model]
      attn_out{L}, mlp_out{L}                     [batch, pos, d_model]
      mlp_post{L}                                 [batch, pos, d_mlp]    (the "neurons")
      z{L}                                        [batch, pos, head, d_head]
      pattern{L}                                  [batch, head, pos, pos]
    """

    def __init__(self, sd: dict[str, torch.Tensor] | None = None, device: str = DEVICE, model_id: str = MODEL_ID) -> None:
        """Load the trained weights unless a state dict is supplied."""
        if sd is None:
            sd = load_file(_fetch(model_id, "model.safetensors"))
        cfg = json.load(open(_fetch(model_id, "config.json")))
        self.device = device
        self.sd = {k.replace("transformer.", ""): v.to(device) for k, v in sd.items()}
        self.d_model = self.sd["wte.weight"].shape[1]
        self.n_layers = cfg["n_layer"]
        self.n_heads = cfg["n_head"]
        self.d_head = self.d_model // self.n_heads
        self.WE = self.sd["wte.weight"]
        self.PE = self.sd["wpe.weight"]
        vocab = json.load(open(_fetch(model_id, "tokenizer.json")))["model"]["vocab"]
        self.vocab: dict[str, int] = vocab
        self.itos = {v: k for k, v in vocab.items()}
        self.eos = vocab["<eos>"]
        self.pad = vocab["<pad>"]

    def _ln(self, name: str, x: torch.Tensor) -> torch.Tensor:
        w, b = self.sd[f"{name}.weight"], self.sd[f"{name}.bias"]
        return F.layer_norm(x, (self.d_model,), w, b, eps=1e-5)

    def forward(
        self,
        ids: torch.Tensor,
        capture: set[str] | None = None,
        hooks: dict | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Run the model. ``hooks`` maps hook name -> fn(tensor) -> tensor (applied in-line)."""
        capture = capture or set()
        hooks = hooks or {}
        cache: dict[str, torch.Tensor] = {}

        def hp(name: str, t: torch.Tensor) -> torch.Tensor:
            if name in hooks:
                t = hooks[name](t)
            if name in capture:
                cache[name] = t
            return t

        n = ids.shape[1]
        x = self.WE[ids] + self.PE[torch.arange(n, device=ids.device)]
        mask = torch.tril(torch.ones(n, n, device=ids.device, dtype=torch.bool))
        for L in range(self.n_layers):
            x = hp(f"resid_pre{L}", x)
            h = self._ln(f"h.{L}.ln_1", x)
            qkv = h @ self.sd[f"h.{L}.attn.c_attn.weight"] + self.sd[f"h.{L}.attn.c_attn.bias"]
            q, k, v = qkv.split(self.d_model, dim=-1)
            bs = ids.shape[0]
            q = q.view(bs, n, self.n_heads, self.d_head).transpose(1, 2)
            k = k.view(bs, n, self.n_heads, self.d_head).transpose(1, 2)
            v = v.view(bs, n, self.n_heads, self.d_head).transpose(1, 2)
            scores = q @ k.transpose(-1, -2) / math.sqrt(self.d_head)
            scores = scores.masked_fill(~mask, float("-inf"))
            pat = hp(f"pattern{L}", scores.softmax(-1))
            z = (pat @ v).transpose(1, 2)  # [b, pos, head, d_head]
            z = hp(f"z{L}", z)
            attn = z.reshape(bs, n, self.d_model)
            attn = attn @ self.sd[f"h.{L}.attn.c_proj.weight"] + self.sd[f"h.{L}.attn.c_proj.bias"]
            attn = hp(f"attn_out{L}", attn)
            x = hp(f"resid_mid{L}", x + attn)
            m = self._ln(f"h.{L}.ln_2", x)
            post = gelu_new(m @ self.sd[f"h.{L}.mlp.c_fc.weight"] + self.sd[f"h.{L}.mlp.c_fc.bias"])
            post = hp(f"mlp_post{L}", post)
            mo = post @ self.sd[f"h.{L}.mlp.c_proj.weight"] + self.sd[f"h.{L}.mlp.c_proj.bias"]
            mo = hp(f"mlp_out{L}", mo)
            x = hp(f"resid_post{L}", x + mo)
        logits = self.unembed(x)
        return logits, cache

    def unembed(self, resid: torch.Tensor) -> torch.Tensor:
        """Apply final LayerNorm and the tied unembedding (also used for the logit lens)."""
        return self._ln("ln_f", resid) @ self.WE.T


def random_init_model(seed: int = 0) -> Model:
    """A control model: same architecture and init distribution, never trained."""
    m = Model()
    g = torch.Generator(device="cpu").manual_seed(seed)
    sd = {}
    for k, v in m.sd.items():
        if "ln" in k and k.endswith("weight"):
            sd[k] = torch.ones_like(v)
        elif k.endswith("bias"):
            sd[k] = torch.zeros_like(v)
        else:
            sd[k] = (torch.randn(v.shape, generator=g) * 0.02).to(v.device)
    r = Model.__new__(Model)
    r.__dict__.update(m.__dict__)
    r.sd = sd
    r.WE, r.PE = sd["wte.weight"], sd["wpe.weight"]
    return r


# --------------------------------------------------------------------------------------------- #
# Problems
# --------------------------------------------------------------------------------------------- #
def to_base(n: int, base: int) -> str:
    """Forward digit string of n in the given base (uppercase hex; same as the training data)."""
    if n == 0:
        return "0"
    out = []
    while n:
        out.append(DIGITS[n % base])
        n //= base
    return "".join(reversed(out))


def to_answer(n: int, base: int) -> str:
    """The answer convention: LSD first, unpadded."""
    return to_base(n, base)[::-1]


def digits_lsd(n: int, base: int, width: int) -> list[int]:
    """Digits of n, least significant first, zero-extended to ``width``."""
    out = []
    for _ in range(width):
        out.append(n % base)
        n //= base
    return out


def schoolbook(a: int, b: int, base: int, width: int = 12) -> dict[str, list[int]]:
    """All per-digit high-level variables of schoolbook addition, LSD first."""
    da, db = digits_lsd(a, base, width), digits_lsd(b, base, width)
    c, cs, ss, os, nc = 0, [], [], [], []
    for k in range(width):
        s = da[k] + db[k] + c
        cs.append(c)
        ss.append(s)
        os.append(s % base)
        c = int(s >= base)
        nc.append(c)
    return {"a": da, "b": db, "c_in": cs, "s": ss, "o": os, "c_out": nc}


def render(a: int, b: int, base: int) -> str:
    """One solved example, as it appears in a prompt."""
    return f"{to_base(a, base)}+{to_base(b, base)}={to_answer(a + b, base)}"


@dataclass
class Problem:
    """A prompt and its ground truth. ``base`` is the base of the test line; ``ctx`` the context's."""

    a: int
    b: int
    base: int
    shots: list[tuple[int, int]] = field(default_factory=list)
    ctx_base: int | None = None
    answer_override: str | None = None
    answer_base: int | None = None

    @property
    def context_base(self) -> int:
        """Base used to render the few-shot examples."""
        return self.ctx_base if self.ctx_base is not None else self.base

    @property
    def prompt(self) -> str:
        """Full prompt ending in '='."""
        lines = [render(x, y, self.context_base) for x, y in self.shots]
        return "\n".join(lines + [f"{to_base(self.a, self.base)}+{to_base(self.b, self.base)}="])

    @property
    def answer(self) -> str:
        """The answer string the test line should produce."""
        if self.answer_override is not None:
            return self.answer_override
        return to_answer(self.a + self.b, self.answer_base or self.base)


def sample_shots(rng: random.Random, exclude: tuple[int, int], k: int = 5, hi: int = 999) -> list[tuple[int, int]]:
    """Few-shot pairs: uniform on [0, hi]^2, distinct, not equal to the test pair."""
    seen = {exclude}
    out = []
    while len(out) < k:
        p = (rng.randint(0, hi), rng.randint(0, hi))
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def sample_problems(n: int, base: int, seed: int = 0, hi: int = 999, fixed_shots: bool = False) -> list[Problem]:
    """Random problems in one base. ``fixed_shots`` uses one canonical context for every prompt."""
    rng = random.Random(seed * 1009 + base)
    canon = sample_shots(random.Random(12345 + base), (-1, -1))
    out = []
    for _ in range(n):
        a, b = rng.randint(0, hi), rng.randint(0, hi)
        out.append(Problem(a, b, base, canon if fixed_shots else sample_shots(rng, (a, b))))
    return out


# --------------------------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------------------------- #
class Batch:
    """Right-padded token ids (prompt + answer + eos) plus per-problem semantic positions.

    Right padding is safe because attention is causal and positions are absolute: padding never
    influences activations at or before the last real token.
    """

    def __init__(self, model: Model, problems: list[Problem], with_answer: bool = True) -> None:
        """Tokenise. With ``with_answer`` the correct answer and EOS are appended (teacher forcing)."""
        self.problems = problems
        vocab = model.vocab
        seqs, self.p_eq, self.p_plus, self.p_astart, self.len_a, self.len_b, self.len_ans = [], [], [], [], [], [], []
        for p in problems:
            prompt = p.prompt
            ans = p.answer if with_answer else ""
            ids = [vocab[c] for c in prompt] + [vocab[c] for c in ans] + ([model.eos] if with_answer else [])
            seqs.append(ids)
            self.p_eq.append(len(prompt) - 1)
            sa, sb = to_base(p.a, p.base), to_base(p.b, p.base)
            self.p_plus.append(len(prompt) - 2 - len(sb))
            self.p_astart.append(len(prompt) - 2 - len(sb) - len(sa))
            self.len_a.append(len(sa))
            self.len_b.append(len(sb))
            self.len_ans.append(len(p.answer))
        n = max(len(s) for s in seqs)
        self.ids = torch.full((len(seqs), n), model.pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            self.ids[i, : len(s)] = torch.tensor(s)
        self.ids = self.ids.to(model.device)
        self.lengths = [len(s) for s in seqs]
        self.p_eq = np.array(self.p_eq)
        self.p_plus = np.array(self.p_plus)
        self.p_astart = np.array(self.p_astart)
        self.len_a = np.array(self.len_a)
        self.len_b = np.array(self.len_b)
        self.len_ans = np.array(self.len_ans)


def iter_batches(problems: list[Problem], bs: int = 512):
    """Yield chunks of problems, roughly grouped so padding stays small."""
    for i in range(0, len(problems), bs):
        yield problems[i : i + bs]


@torch.no_grad()
def greedy(
    model: Model,
    problems: list[Problem],
    max_new: int = 16,
    bs: int = 512,
    hooks_fn=None,
) -> list[str]:
    """Greedy generation (batched; each prompt tracks its own length). Returns text before EOS."""
    out: list[str] = []
    for chunk in iter_batches(problems, bs):
        seqs = [[model.vocab[c] for c in p.prompt] for p in chunk]
        done = [False] * len(seqs)
        for step in range(max_new):
            n = max(len(s) for s in seqs)
            ids = torch.full((len(seqs), n), model.pad, dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, : len(s)] = torch.tensor(s)
            ids = ids.to(model.device)
            hooks = hooks_fn(chunk, step, seqs) if hooks_fn else None
            logits, _ = model.forward(ids, hooks=hooks)
            idx = torch.tensor([len(s) - 1 for s in seqs], device=model.device)
            nxt = logits[torch.arange(len(seqs)), idx].argmax(-1).tolist()
            for i, t in enumerate(nxt):
                if not done[i]:
                    if t == model.eos:
                        done[i] = True
                    else:
                        seqs[i].append(t)
            if all(done):
                break
        for s, p in zip(seqs, chunk):
            out.append("".join(model.itos[t] for t in s[len(p.prompt) :]))
    return out


# --------------------------------------------------------------------------------------------- #
# Stats helpers
# --------------------------------------------------------------------------------------------- #
def ci95(x: np.ndarray | list[float]) -> tuple[float, float]:
    """Mean and normal-approximation 95% half-width."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(1.96 * x.std(ddof=1) / np.sqrt(max(len(x), 1))) if len(x) > 1 else 0.0


def wilson(k: int, n: int) -> tuple[float, float, float]:
    """Wilson 95% interval for a proportion: (p, lo, hi)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    z = 1.96
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def save_json(name: str, obj: object) -> None:
    """Write results to data/<name>.json."""
    DATA.mkdir(exist_ok=True, parents=True)

    def default(o: object) -> object:
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))

    (DATA / f"{name}.json").write_text(json.dumps(obj, indent=1, default=default))


def style() -> None:
    """Recessive axes and grid, consistent across figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK2,
            "axes.titlesize": 10,
            "axes.titlecolor": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "axes.axisbelow": True,
            "xtick.color": INK2,
            "ytick.color": INK2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfb",
        }
    )
