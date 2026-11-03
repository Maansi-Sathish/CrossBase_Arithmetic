"""Build report figures from saved results only."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from core import BASE_COLOR, BASES, DATA, FIGS, GRID, INK, INK2, style

style()
FIGS.mkdir(exist_ok=True)
load = lambda n: json.load(open(DATA / f"{n}.json"))
BLUES = matplotlib.colors.LinearSegmentedColormap.from_list("b", ["#f4f7fb", "#2a78d6", "#123f78"])

# 1. attention
z = np.load(DATA / "e03_attention.npz")
slots = load("e03_attention_embed")["slots"]
keep = [i for i, s in enumerate(slots) if s in ("a_k", "b_k", "a_k-1", "b_k-1", "o_k-1", "+", "=", "ctx_digit", "ctx_newline")]
fig, axes = plt.subplots(2, 4, figsize=(10, 4.6), sharex=True, sharey=True)
for r, k in enumerate((0, 1)):
    for c, b in enumerate(BASES):
        ax = axes[r, c]
        im = ax.imshow(z[f"b{b}"][k, 1][:, keep], cmap=BLUES, vmin=0, vmax=1, aspect="auto")
        ax.grid(False)
        ax.set_title(f"Base {b}, digit {k}")
        ax.set_yticks(range(4), [f"L1H{h}" for h in range(4)])
        ax.set_xticks(range(len(keep)), [slots[i] for i in keep], rotation=60, ha="right", fontsize=7)
fig.colorbar(im, ax=axes, shrink=0.8, label="Mean attention")
fig.suptitle("Layer 1 heads read the matching operand digits", color=INK, x=0.45)
fig.savefig(FIGS / "attention_l1.png")
plt.close(fig)

# 2. spectrum
s = load("s04_fourier_helix")
P = np.array(s["periods"])
fig, axes = plt.subplots(2, 2, figsize=(9, 5.2), sharex=True, sharey=True)
for ax, b in zip(axes.ravel(), BASES):
    d = s["A"][str(b)]["=/a+b"]
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(P, d["input_onehot"], color="#9a9890", lw=1.5, label="Input digits")
    ax.plot(P, d["resid1"], color=BASE_COLOR[b], label="Layer 1 residual")
    ax.set_xscale("log", base=2)
    ax.set_ylim(-0.1, 1.02)
    ax.set_title(f"Base {b}")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xticks([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [2, 4, 8, 16, 32, 64, 128, 256, 512, "1024"])
    for T in (5, 10):
        if b == 10:
            ax.axvline(T, color=GRID, lw=0.8, zorder=0)
for ax in axes[1]:
    ax.set_xlabel("Period T of (a+b)")
for ax in axes[:, 0]:
    ax.set_ylabel("Held-out R²")
fig.suptitle("At '=', periodic structure in a+b belongs to the prompt's own base", color=INK)
fig.tight_layout()
fig.savefig(FIGS / "fourier_sum.png")
plt.close(fig)

# 3. head decoding
h = load("s06_heads_base")["A"]
cols = ["a", "b", "c", "s", "o"]
names = {"a": "a_k", "b": "b_k", "c": "carry", "s": "raw sum", "o": "output"}
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for ax, (L, k) in zip(axes, [(1, 0), (1, 1), (0, 1)]):
    M = np.array([[np.array(h[f"L{L}H{i}/k{k}"][v], dtype=float)[2, 2] for v in cols] for i in range(4)])
    im = ax.imshow(M, cmap=BLUES, vmin=0, vmax=1, aspect="auto")
    ax.grid(False)
    ax.set_title(f"Layer {L} heads, answer digit {k}")
    ax.set_xticks(range(5), [names[v] for v in cols], fontsize=8)
    ax.set_yticks(range(4), [f"H{i}" for i in range(4)])
    for i in range(4):
        for j in range(5):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 else INK)
fig.suptitle("Base 10: heads deliver digits and carry, but no head alone holds the sum", color=INK)
fig.tight_layout()
fig.colorbar(im, ax=axes, shrink=0.8, label="Probe accuracy")
fig.savefig(FIGS / "head_decoding.png")
plt.close(fig)

# 4. drop correlation
c = load("s05_classes")["D"]
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9))
for ax, L in zip(axes, ("L0", "L1")):
    M = np.array(c[L]["spearman"])
    im = ax.imshow(M, cmap=BLUES, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_title(f"Layer {L[1]} MLP neurons")
    ax.set_xticks(range(4), BASES)
    ax.set_yticks(range(4), BASES)
    ax.set_xlabel("Base")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 else INK)
axes[0].set_ylabel("Base")
fig.suptitle("Rank correlation of ablation drops across bases", color=INK)
fig.tight_layout()
fig.colorbar(im, ax=axes, shrink=0.8)
fig.savefig(FIGS / "drop_correlation.png")
plt.close(fig)
print("ok")

# 7. clock offset profiles
k = load("f12_clock")["A"]
fig, axes = plt.subplots(1, 3, figsize=(10, 2.9), sharey=True)
for ax, b in zip(axes, (8, 10, 16)):
    g = np.array(k[str(b)]["0"]["g"])
    ax.axhline(0, color=GRID, lw=1)
    ax.bar(range(b), g, color=BASE_COLOR[b], width=0.7)
    ax.set_title(f"Base {b}")
    ax.set_xlabel("Token minus correct digit, mod base")
    ax.set_xticks(range(0, b, 2 if b < 16 else 4))
axes[0].set_ylabel("Mean logit")
fig.suptitle("Mean logit by distance from the correct digit: parity and closeness", color=INK)
fig.tight_layout()
fig.savefig(FIGS / "clock_profile.png")
plt.close(fig)
