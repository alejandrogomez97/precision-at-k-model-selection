"""Regret per selection metric vs. positives-in-budget, one panel per model family.
Extends the LightGBM 'count governs' view to Random Forest and Logistic Regression."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/home/agomez/proyectos/precision-at-k-study"
COL = {"P@K": "#c0392b", "AP": "#27ae60", "logloss": "#7d3c98"}
M3 = ["P@K", "AP", "logloss"]
LAB = {"lightgbm": "LightGBM", "rf": "Random Forest", "logreg": "Logistic Regression"}

# gather rows per family
rows = {"lightgbm": [], "rf": [], "logreg": []}
for r in json.load(open(f"{OUT}/results_real3.json")):
    rows["lightgbm"].append(r)
for r in json.load(open(f"{OUT}/results_families.json")):
    if r.get("family") in rows:
        rows[r["family"]].append(r)

def binned(info, nb=5):
    x = np.array([r["pos_in_budget"] for r in info], float)
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

fams = ["lightgbm", "rf", "logreg"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
for ax, fam in zip(axes, fams):
    info = [r for r in rows[fam] if (r["oracle"] - r["meancfg"]) >= 0.02]
    cen, ser, ns = binned(info)
    xs = range(len(cen))
    for m in M3:
        ax.plot(xs, ser[m], "o-", color=COL[m], lw=2.2, ms=7, label=m)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f"≈{v:.0f}" for v in cen])
    ax.set_title(f"{LAB[fam]}  (N={len(info)})", fontsize=12)
    ax.set_xlabel("positives inside the budget (count)")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("mean normalized regret (lower better)")
axes[2].legend(title="selection metric", loc="upper right")
fig.suptitle("Same pattern in all three model families: as more positives land in the budget, regret falls —\n"
             "and average precision (green) stays at or below precision@k (red) throughout",
             fontsize=13, y=1.04)
fig.text(0.5, -0.03,
         "How to read it:  → moving right = datasets where MORE true positives fall inside the top-K budget (≈4, ≈9, … positives).  "
         "Each coloured line = a metric used to PICK the model; lower = it picked better models (less regret).  "
         "Every dot averages one equal-size group of datasets (~55–70 dataset×budget cases) — the groups are equal in size, which is why moving right the positives grow but each dot rests on a similar amount of data.",
         ha="center", fontsize=8, color="#555")
fig.tight_layout()
fig.savefig(f"{OUT}/figG_families_by_count.png", dpi=130, bbox_inches="tight")
plt.close(fig)
print("saved figG_families_by_count.png")
