"""analyze_featreduce.py — barrido fino de reducción por correlación de Spearman.
Lee results/<TAG>__*.json (TAG por env FRTAG, def featreducefine), agrega por umbral y
produce: (1) tabla con significancia (t-test pareado ΔAP vs baseline por umbral),
(2) fig_featreduce_fine.png con dos paneles: features eliminadas (% y nº medio) y AP vs
umbral (media ± SE, baseline y marcados los umbrales con p<0.05)."""
import os, glob, json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
TAG = os.environ.get("FRTAG", "featreducefine")

rows = [r for f in glob.glob(f"{STUDY}/results/{TAG}__*.json") for r in json.load(open(f))]
d = pd.DataFrame(rows)
base = d[d.thr > 1].set_index(["dataset", "seed"])[["ap", "wall_time"]]
d["ap0"] = d.set_index(["dataset", "seed"]).index.map(base.ap)
d["dAP"] = d.ap - d.ap0
ap0 = float(d[d.thr > 1].ap.mean())

thrs = sorted(t for t in d.thr.unique() if t <= 1.0)
rec = []
for thr in thrs:
    s = d[d.thr == thr]
    dl = s.dAP.dropna().values
    p = float(stats.ttest_1samp(dl, 0).pvalue) if len(dl) >= 3 and np.std(dl) > 0 else np.nan
    rec.append({"thr": thr, "n_removed": s.n_removed.mean(), "pct": s.pct_removed.mean(),
                "ap": s.ap.mean(), "ap_se": s.ap.std() / np.sqrt(len(s)),
                "dAP": s.dAP.mean(), "p": p, "n": len(s)})
A = pd.DataFrame(rec)

print(f"baseline (keep all): AP={ap0:.4f}  (n_datasets×seeds por umbral = {A.n.iloc[0]})")
print(f"{'|ρ|>=':>7} {'#rem':>6} {'%rem':>6} {'AP':>7} {'ΔAP':>8} {'p':>7} {'sig':>4}")
for _, r in A.iterrows():
    sig = "***" if r.p < 0.01 else ("**" if r.p < 0.05 else ("*" if r.p < 0.1 else ""))
    print(f"{r.thr:7.3f} {r.n_removed:6.1f} {r.pct:5.1f}% {r.ap:7.4f} {r.dAP:+8.4f} {r.p:7.3f} {sig:>4}")
json.dump({"baseline_ap": ap0, "by_thr": [
    {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v) for k, v in r.items()}
    for r in rec]}, open(f"{STUDY}/summary_featreduce_fine.json", "w"), indent=1)

# ---------- figura ----------
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
# panel 1: features eliminadas (% y nº medio)
a0 = ax[0]; a0b = a0.twinx()
l1, = a0.plot(A.thr, A.pct, "o-", color="#1f77b4", label="% removed")
l2, = a0b.plot(A.thr, A.n_removed, "s--", color="#7f7f7f", label="# removed (mean)")
a0.set_xlabel("|Spearman| threshold (remove pairs with |ρ| ≥ threshold)")
a0.set_ylabel("% features removed", color="#1f77b4"); a0b.set_ylabel("# features removed (mean)", color="#7f7f7f")
a0.set_title("How many features get removed"); a0.grid(alpha=.3)
a0.legend(handles=[l1, l2], fontsize=8, loc="upper right")
# panel 2: AP vs umbral con SE, baseline y significancia
a1 = ax[1]
a1.axhline(ap0, color="k", ls="--", lw=.9, label=f"baseline (keep all) = {ap0:.3f}")
a1.errorbar(A.thr, A.ap, yerr=A.ap_se, fmt="o-", color="#c0392b", capsize=3, label="AP (mean ± SE)")
sig = A[A.p < 0.05]
if len(sig):
    a1.scatter(sig.thr, sig.ap, s=140, facecolors="none", edgecolors="red", linewidths=2,
               label="p < 0.05 vs baseline", zorder=5)
a1.set_xlabel("|Spearman| threshold"); a1.set_ylabel("AP (test)")
a1.set_title("Does removing correlated features help AP?"); a1.legend(fontsize=8); a1.grid(alpha=.3)
plt.suptitle("Feature removal by Spearman correlation — fine sweep 0.90–1.00 (E2 + LightGBM, "
             f"{int(A.n.iloc[0]/2)} datasets × 2 seeds)", fontweight="bold")
plt.tight_layout(); plt.savefig(f"{STUDY}/fig_featreduce_fine.png", dpi=140); plt.close()
print("\nfig -> fig_featreduce_fine.png ; summary -> summary_featreduce_fine.json")
