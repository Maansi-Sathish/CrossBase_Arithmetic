"""Probe utilities: loaders for the extracted tables, control features, and fast GPU probes.

Controls used throughout, because decodability alone proves little:
  * input baseline: the same probe on one-hot aligned digits of the test line. Anything it decodes
    needs no computation by the model.
  * random-init baseline: the same probe on activations of an untrained model with the same prompts.
    Anything it decodes comes from the architecture and tokens, not from training.
  * integer hold-out: test integers are never seen at training time, so a probe can't just memorise
    an integer -> label table.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
from core import BASES, BUCKET, DEVICE, Model, Problem, digits_lsd, random_init_model, schoolbook

TABLES = BUCKET.parent / "tables"
A_DIGIT, B_DIGIT, PLUS, QUERY = 0, 1, 2, 3


class Table:
    """Memory-mapped activations for one base. Rows are (prompt, position)."""

    def __init__(self, base: int) -> None:
        """Open the arrays for ``base``."""
        self.base = base
        self.dir = TABLES / f"b{base}"
        m = np.load(self.dir / "rows.npz")
        self.meta = {k: m[k] for k in m.files}
        self.arr = {k: np.load(self.dir / f"{k}.npy", mmap_mode="r") for k in ["resid0", "resid1", "mlp0", "mlp1", "z0", "z1"]}
        self.pairs = np.load(self.dir / "pairs.npy")
        self.shots = np.load(self.dir / "shots.npy")

    def select(self, kind: int, j: int) -> np.ndarray:
        """Row indices for one semantic position, e.g. (QUERY, 0) is '='."""
        return np.where((self.meta["kind"] == kind) & (self.meta["j"] == j))[0]

    def problems(self, idx: np.ndarray) -> list[Problem]:
        """The prompts for the given prompt indices."""
        return [
            Problem(int(self.pairs[i, 0]), int(self.pairs[i, 1]), self.base, [tuple(map(int, s)) for s in self.shots[i]])
            for i in idx
        ]

    def get(self, name: str, rows: np.ndarray) -> np.ndarray:
        """Materialise a feature matrix."""
        return np.asarray(self.arr[name][rows], dtype=np.float32)


def input_onehot(a: np.ndarray, b: np.ndarray, base: int, width: int = 11) -> np.ndarray:
    """Input baseline: one-hot digit at every LSD-aligned slot, with an 'absent' symbol."""
    n = len(a)
    out = np.zeros((n, 2 * width * (base + 1)), dtype=np.float32)
    for side, v in enumerate((a, b)):
        for i, x in enumerate(v.tolist()):
            ds = digits_lsd(int(x), base, width)
            length = 0
            y = int(x)
            while y:
                y //= base
                length += 1
            length = max(length, 1)
            for k in range(width):
                sym = ds[k] if k < length else base
                out[i, side * width * (base + 1) + k * (base + 1) + sym] = 1
    return out


@torch.no_grad()
def random_features(base: int, problems: list[Problem], kind: int, j: int, name: str = "resid_post1", seed: int = 0) -> np.ndarray:
    """Activations of an untrained model at a semantic position, teacher-forced with the true answer."""
    from core import Batch, iter_batches

    m = random_init_model(seed)
    out = []
    for chunk in iter_batches(problems, 256):
        bt = Batch(m, chunk)
        _, cache = m.forward(bt.ids, capture={name})
        if kind == QUERY:
            pos = bt.p_eq + j
        elif kind == A_DIGIT:
            pos = bt.p_plus - 1 - j
        elif kind == B_DIGIT:
            pos = bt.p_eq - 1 - j
        else:
            pos = bt.p_plus
        pos_t = torch.tensor(np.minimum(pos, bt.ids.shape[1] - 1), device=m.device)
        out.append(cache[name][torch.arange(len(chunk), device=m.device), pos_t].cpu().numpy())
    return np.concatenate(out)


def integer_split(values: np.ndarray, frac: float = 0.25, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Train and test masks where test integers never occur in training."""
    uniq = np.unique(values)
    rng = np.random.default_rng(seed)
    test_vals = rng.choice(uniq, int(len(uniq) * frac), replace=False)
    te = np.isin(values, test_vals)
    return ~te, te


class Ridge:
    """Closed-form ridge with lambda chosen on an inner split, standardised features, on the GPU."""

    def __init__(self, x: np.ndarray, device: str = DEVICE) -> None:
        """Pre-compute the statistics for a training matrix."""
        self.device = device
        X = torch.as_tensor(x, device=device, dtype=torch.float64)
        self.mu = X.mean(0)
        self.sd = X.std(0).clamp_min(1e-6)
        self.X = (X - self.mu) / self.sd

    def _z(self, x: np.ndarray) -> torch.Tensor:
        return (torch.as_tensor(x, device=self.device, dtype=torch.float64) - self.mu) / self.sd

    def fit(self, y: np.ndarray, lams: tuple[float, ...] = (1e-2, 1.0, 1e2, 1e4)) -> torch.Tensor:
        """Solve for multi-output targets. Returns [d, outputs] coefficients (intercept is y mean)."""
        Y = torch.as_tensor(y, device=self.device, dtype=torch.float64)
        if Y.ndim == 1:
            Y = Y[:, None]
        self.ymu = Y.mean(0)
        Yc = Y - self.ymu
        n = self.X.shape[0]
        perm = torch.randperm(n, device=self.device, generator=torch.Generator(device=self.device).manual_seed(0))
        cut = int(0.8 * n)
        tr, va = perm[:cut], perm[cut:]
        best, best_l = float("inf"), lams[0]
        d = self.X.shape[1]
        Xtr = self.X[tr]
        XtX = Xtr.T @ Xtr
        XtY = Xtr.T @ Yc[tr]
        eye = torch.eye(d, device=self.device, dtype=torch.float64)
        if len(lams) > 1:
            for lam in lams:
                coef = torch.linalg.solve(XtX + lam * eye, XtY)
                err = ((self.X[va] @ coef - Yc[va]) ** 2).mean().item()
                if err < best:
                    best, best_l = err, lam
        else:
            best_l = lams[0]
        self.lam = best_l
        self.coef = torch.linalg.solve(self.X.T @ self.X + best_l * eye, self.X.T @ Yc)
        return self.coef

    def predict(self, x: np.ndarray) -> torch.Tensor:
        """Predict with the last fitted coefficients."""
        return self._z(x) @ self.coef + self.ymu


def r2(pred: torch.Tensor, y: np.ndarray, reference_mean: torch.Tensor | None = None) -> float:
    """Pooled R^2 over outputs. Negative values mean worse than predicting the test mean."""
    Y = torch.as_tensor(y, device=pred.device, dtype=torch.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    mean = Y.mean(0) if reference_mean is None else reference_mean
    return float(1 - ((pred - Y) ** 2).sum() / ((Y - mean) ** 2).sum().clamp_min(1e-12))


def onehot(y: np.ndarray, n: int) -> np.ndarray:
    """One-hot encode integer labels."""
    out = np.zeros((len(y), n), dtype=np.float64)
    out[np.arange(len(y)), y] = 1
    return out


def classify(xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, n_classes: int) -> dict:
    """Linear classifier (ridge on one-hot). Reports accuracy, chance level, and a binomial CI."""
    model = Ridge(xtr)
    model.fit(onehot(ytr, n_classes))
    pred = model.predict(xte).argmax(1).cpu().numpy()
    acc = float((pred == yte).mean())
    majority = float(np.bincount(yte, minlength=n_classes).max() / len(yte))
    return {"acc": acc, "chance": majority, "ci": 1.96 * float(np.sqrt(acc * (1 - acc) / len(yte))), "n_test": len(yte)}


def mlp_classify(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xte: np.ndarray,
    yte: np.ndarray,
    n_classes: int,
    hidden: int = 512,
    epochs: int = 300,
    seed: int = 0,
) -> dict:
    """Two-hidden-layer MLP probe with early stopping on an inner held-out split."""
    torch.manual_seed(seed)
    X = torch.as_tensor(xtr, device=DEVICE, dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0).clamp_min(1e-6)
    X = (X - mu) / sd
    Y = torch.as_tensor(ytr, device=DEVICE, dtype=torch.long)
    XT = (torch.as_tensor(xte, device=DEVICE, dtype=torch.float32) - mu) / sd
    YT = torch.as_tensor(yte, device=DEVICE, dtype=torch.long)
    n = len(X)
    perm = torch.randperm(n, device=DEVICE)
    cut = int(0.85 * n)
    tr, va = perm[:cut], perm[cut:]
    net = torch.nn.Sequential(
        torch.nn.Linear(X.shape[1], hidden), torch.nn.GELU(), torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Linear(hidden, n_classes)
    ).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    best, best_state, wait = -1.0, None, 0
    bs = 512
    for ep in range(epochs):
        net.train()
        order = tr[torch.randperm(len(tr), device=DEVICE)]
        for i in range(0, len(order), bs):
            b = order[i : i + bs]
            loss = torch.nn.functional.cross_entropy(net(X[b]), Y[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = (net(X[va]).argmax(1) == Y[va]).float().mean().item()
        if v > best:
            best, best_state, wait = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            wait += 1
            if wait > 25:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        acc = (net(XT).argmax(1) == YT).float().mean().item()
    majority = float(np.bincount(yte, minlength=n_classes).max() / len(yte))
    return {"acc": acc, "chance": majority, "ci": 1.96 * float(np.sqrt(acc * (1 - acc) / len(yte))), "n_test": len(yte)}
