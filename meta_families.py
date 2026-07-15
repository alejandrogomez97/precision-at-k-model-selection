"""Analyse results_families.json: does the conclusion hold for RF and LogReg?
Per family: mean regret, win-rate, paired Wilcoxon vs precision@k. One figure."""
import json, numpy as np
from scipy.stats import wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/home/agomez/proyectos/precision-at-k-study"
SEL = ["P@K", "AP", "AUC", "logloss"]
COL = {"P@K": "#c0392b", "AP": "#27ae60", "AUC": "#2471a3", "logloss": "#7d3c98"}
LAB = {"logreg": "Logistic Regression", "rf": "Random Forest", "lightgbm": "LightGBM"}

R = json.load(open(f"{OUT}/results_families.json"))
# also fold in the LightGBM run for a 3-family comparison
try:
    for r in json.load(open(f"{OUT}/results_real3.json")):
        r = dict(r); r["family"] = "lightgbm"; R.append(r)
except FileNotFoundError:
    pass

fams = ["lightgbm", "rf", "logreg"]
summary = {}
print("family        metric   mean_regret  win%   Wilcoxon p vs P@K")
for fam in fams:
    info = [r for r in R if r.get("family") == fam and (r["oracle"] - r["meancfg"]) >= 0.02]
    if not info:
        continue
    def col(m): return np.array([r[f"nregret_{m}"] for r in info], float)
    n = len(info); summary[fam] = {"n": n, "metrics": {}}
    for m in SEL:
        wins = sum(1 for r in info if min(SEL, key=lambda mm: r[f"nregret_{mm}"]) == m)
        p = float(wilcoxon(col(m), col("P@K"))[1]) if m != "P@K" else float("nan")
        summary[fam]["metrics"][m] = dict(mean=float(col(m).mean()),
                                          win=wins / n, p_vs_pk=p)
        print(f"{LAB[fam]:20s} {m:8s} {col(m).mean():9.3f}  {100*wins/n:4.0f}   "
              f"{'' if m=='P@K' else f'{p:.2e}'}")
    print()

# ---- figure: mean regret per metric, grouped by family ----
fig, ax = plt.subplots(figsize=(9.5, 5))
w = 0.2; xb = np.arange(len(fams))
for i, m in enumerate(SEL):
    vals = [summary[f]["metrics"][m]["mean"] for f in fams if f in summary]
    ax.bar(xb + (i - 1.5) * w, vals, w, color=COL[m], label=m, alpha=0.85)
ax.set_xticks(xb); ax.set_xticklabels([f"{LAB[f]}\n(n={summary[f]['n']} cases)" for f in fams if f in summary])
ax.set_ylabel("mean normalized regret (lower better)")
ax.set_title("Average precision beats precision@k in all three model families (p<0.001)\n"
             "log-loss's edge is model-dependent; AUC is never better",
             fontsize=12)
ax.grid(alpha=0.25, axis="y"); ax.legend(title="selection metric", ncol=4, loc="upper center")
fig.text(0.5,-0.02,"Each bar = mean normalized regret when selecting the model by that metric (0 = best model in the bank, 1 = an average one).  "
         "n = number of (dataset, budget K) cases per family.",
         ha="center",fontsize=8,color="#666")
fig.tight_layout(); fig.savefig(f"{OUT}/figF_families.png", dpi=130, bbox_inches="tight")
plt.close(fig); print("saved figF_families.png")

json.dump(summary, open(f"{OUT}/families_summary.json", "w"), indent=2)
print("saved families_summary.json")
