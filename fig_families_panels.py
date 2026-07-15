"""Multi-family (LightGBM / Random Forest / Logistic Regression) versions of the
fine-analysis plots: regret per selection metric vs. (a) the ratio n_pos/K, and
(b) the a-priori proxy min(K, n_pos). Companion to fig_families_by_count.py."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/home/agomez/proyectos/precision-at-k-study"
COL = {"P@K": "#c0392b", "AP": "#27ae60", "logloss": "#7d3c98"}
M3 = ["P@K", "AP", "logloss"]
LAB = {"lightgbm": "LightGBM", "rf": "Random Forest", "logreg": "Logistic Regression"}
N_TITLE = {"lightgbm": 322, "rf": 329, "logreg": 288}

rows = {"lightgbm": [], "rf": [], "logreg": []}
for r in json.load(open(f"{OUT}/results_real3.json")):
    rows["lightgbm"].append(r)
for r in json.load(open(f"{OUT}/results_families.json")):
    if r.get("family") in rows:
        rows[r["family"]].append(r)

def xval(r, key):
    if key == "minKN":
        return min(r["K"], r["n_pos"])
    return r[key]

def binned(info, key, nb=5):
    x = np.array([xval(r, key) for r in info], float)
    e = np.unique(np.quantile(x, np.linspace(0, 1, nb + 1)))
    cen = []; ser = {m: [] for m in M3}; ns = []
    for b in range(len(e) - 1):
        lo, hi = e[b], e[b + 1]
        mask = (x >= lo) & (x < hi if b < len(e) - 2 else x <= hi)
        sub = [info[i] for i in range(len(info)) if mask[i]]
        if len(sub) < 4:
            continue
        cen.append(np.median(x[mask])); ns.append(len(sub))
        for m in M3:
            ser[m].append(np.mean([r[f"nregret_{m}"] for r in sub]))
    return cen, ser, ns

def make(key, fname, suptitle, xlabel, fmt, footnote):
    fams = ["lightgbm", "rf", "logreg"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, fam in zip(axes, fams):
        info = [r for r in rows[fam] if (r["oracle"] - r["meancfg"]) >= 0.02]
        cen, ser, ns = binned(info, key)
        xs = range(len(cen))
        for m in M3:
            ax.plot(xs, ser[m], "o-", color=COL[m], lw=2.2, ms=7, label=m)
        ax.set_xticks(list(xs))
        ax.set_xticklabels([f"{fmt(v)}\nn={n}" for v, n in zip(cen, ns)])
        ax.set_title(f"{LAB[fam]}  (N={N_TITLE[fam]})", fontsize=12)
        ax.set_xlabel(xlabel); ax.grid(alpha=0.25)
    axes[0].set_ylabel("mean normalized regret (lower better)")
    axes[2].legend(title="selection metric", loc="best")
    fig.suptitle(suptitle, fontsize=13, y=1.04)
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=130, bbox_inches="tight")
    plt.close(fig); print("saved", fname)

make("npos_over_K", "figH_families_by_ratio.png",
     "The ratio n_pos/K does NOT order regret — in any family\n"
     "(compare with the clean, monotone count axis in the previous figure)",
     "n_pos / K   (positives per inspection slot)",
     lambda v: f"≈{v:.2g}",
     "How to read it:  x = the ratio n_pos/K (budget relative to positives). Each line = a metric to pick the model; lower = better.  "
     "n under each tick = cases averaged (a \"case\" = one dataset at one budget K; the n's sum to N in each panel title, not 93).  "
     "The lines are non-monotone / tangled: the ratio does not tell you how hard selection is.")

make("minKN", "figI_families_by_apriori.png",
     "The a-priori, model-free min(K, n_pos) orders regret across all three families\n"
     "(what you can compute BEFORE training, and it behaves like the true count)",
     "min(K, n_pos)   (most positives that could land in the budget)",
     lambda v: f"≈{v:.0f}",
     "How to read it:  x = min(budget K, number of positives you have) — knowable before training. Each line = a metric to pick the model; lower = better.  "
     "n under each tick = cases averaged (a \"case\" = one dataset at one budget K; the n's sum to N in each panel title, not 93).  "
     "Regret falls as this grows (monotonically for the stable metrics; a small bump in precision@k), and average precision (green) stays at or below precision@k (red) in every bin.")
