"""Concrete worked example: gradient of log-loss vs precision@3 on 8 items.
Generates fig_gradient_example.png (2 panels)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/home/agomez/proyectos/precision-at-k-study"

# 8 installations, sorted by model score (descending)
items = ["A", "B", "C", "D", "E", "F", "G", "H"]
p = np.array([0.92, 0.85, 0.80, 0.78, 0.60, 0.55, 0.30, 0.10])   # scores
y = np.array([1,    0,    0,    1,    1,    0,    1,    0])        # 1 = fraud
K = 3
grad = p - y                     # log-loss pseudo-residual (dL/dz with sigmoid)

RED = "#C0392B"; GREY = "#9AA4B2"; BLUE = "#2471A3"; INK = "#161A20"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.4),
                               gridspec_kw={"width_ratios": [1.05, 1]})

# ---- Panel 1: the ranking and the fixed budget ----
ypos = np.arange(len(items))[::-1]          # A at top
colors = [RED if yi == 1 else GREY for yi in y]
ax1.barh(ypos, p, color=colors, height=0.62, zorder=3)
for yi, pi, it, lab in zip(ypos, p, items, y):
    ax1.text(pi + 0.015, yi, f"{it}", va="center", ha="left",
             fontsize=11, fontweight="bold", color=INK)
    ax1.text(0.02, yi, "fraud" if lab == 1 else "legit", va="center",
             ha="left", fontsize=8.5, color="white", fontweight="bold")
# budget cutoff between rank 3 (C) and 4 (D): ypos of C = 5, D = 4 -> line 4.5
cut = ypos[K] + 0.5
ax1.axhspan(cut, ypos[0] + 0.5, color=RED, alpha=0.06, zorder=0)
ax1.axhline(cut, color=RED, ls="--", lw=1.6, zorder=4)
ax1.text(0.02, cut + 0.10, f"BUDGET: inspect top {K}", ha="left", va="bottom",
         fontsize=9, color=RED, fontweight="bold")
# beneficial finite move: push D above C
ax1.annotate("", xy=(0.815, ypos[2]), xytext=(0.78, ypos[3]),
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.8,
                             connectionstyle="arc3,rad=-0.3"))
ax1.text(0.40, 1.45,
         "raising D past C would help —\nbut that is a finite jump,\nnot a gradient direction",
         fontsize=8.3, color=INK, style="italic")
ax1.set_yticks([]); ax1.set_xlim(0, 1.0); ax1.set_xlabel("model score")
ax1.set_title(f"precision@{K} = 1/3  (only A is fraud inside the budget)",
              fontsize=11)
ax1.spines[["top", "right"]].set_visible(False)

# ---- Panel 2: the gradient each loss provides ----
ax2.axvline(0, color="#C9CFD8", lw=1)
bar_c = [BLUE if g > 0 else RED for g in grad]
ax2.barh(ypos, grad, color=bar_c, height=0.5, alpha=0.85, zorder=3,
         label="log-loss pseudo-residual  p − y")
for yi, g, it in zip(ypos, grad, items):
    ax2.text(g + (0.03 if g >= 0 else -0.03), yi, f"{g:+.2f}",
             va="center", ha="left" if g >= 0 else "right", fontsize=9,
             color=INK, fontweight="bold")
# precision@3 gradient = 0 markers
ax2.scatter(np.zeros_like(ypos), ypos, marker="D", s=55, color=RED,
            edgecolor="white", zorder=5, label="precision@3 gradient = 0")
ax2.set_yticks(ypos); ax2.set_yticklabels(items, fontweight="bold")
ax2.set_xlim(-1.15, 1.25)
ax2.set_xlabel("gradient w.r.t. the score   (◀ push UP / push DOWN ▶)")
ax2.set_title("log-loss gives a direction for every row;\nprecision@3 gives zero",
              fontsize=11)
ax2.spines[["top", "right"]].set_visible(False)
ax2.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

fig.suptitle("One boosting step, two losses — why precision@k has nothing to teach",
             fontsize=13, y=1.0)
fig.tight_layout()
fig.savefig(f"{OUT}/fig_gradient_example.png", dpi=135, bbox_inches="tight")
plt.close(fig)
print("saved fig_gradient_example.png")
