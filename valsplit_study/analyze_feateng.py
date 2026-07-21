"""analyze_feateng.py — Apartado 10. AP vs % de variables inventadas al azar.

Como cada dataset parte de un AP distinto, se mira sobre todo el CAMBIO relativo
respecto a 0% de basura (ΔAP = AP(level) − AP(0%)) promediado entre datasets, además
del AP absoluto. Genera fig_feateng.png + summary_feateng.json.
"""
import glob, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
import os as _os
EN = _os.environ.get("FIGLANG") == "en"
def L(es, en): return en if EN else es
SUF = "_en" if EN else ""


def main():
    rows = [r for f in glob.glob(f"{STUDY}/results/feateng__*.json") for r in json.load(open(f))]
    if not rows:
        print("feateng sin datos"); return
    d = pd.DataFrame(rows)
    print(f"feateng: {d.dataset.nunique()} datasets, niveles={sorted(d.level.unique())}, "
          f"seeds={sorted(d.seed.unique())}")
    # ΔAP respecto a nivel 0 por (dataset, seed)
    base = d[d.level == 0.0].set_index(["dataset", "seed"]).ap
    d["ap0"] = d.set_index(["dataset", "seed"]).index.map(base)
    d["dAP"] = d.ap - d.ap0

    by = d.groupby("level").agg(ap=("ap", "mean"), dAP=("dAP", "mean"),
                                se=("dAP", lambda x: x.std()/np.sqrt(len(x))),
                                t=("wall_time", "mean"), n=("ap", "size")).reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    ax[0].plot(by.level*100, by.ap, "o-", color="#1f77b4")
    ax[0].set_xlabel(L("% de variables inventadas (sobre las originales)", "% invented features (over the originals)"))
    ax[0].set_ylabel("AP (test)"); ax[0].set_title(L("AP absoluto medio", "Mean absolute AP")); ax[0].grid(alpha=.3)

    ax[1].axhline(0, color="k", lw=.8)
    ax[1].errorbar(by.level*100, by.dAP, yerr=by.se, fmt="o-", color="#c0392b", capsize=3)
    ax[1].fill_between(by.level*100, by.dAP-by.se, by.dAP+by.se, color="#c0392b", alpha=.15)
    ax[1].set_xlabel(L("% de variables inventadas", "% invented features")); ax[1].set_ylabel(L("ΔAP vs 0% basura", "ΔAP vs 0% noise"))
    ax[1].set_title(L("Cambio de AP (pareado)", "AP change (paired)")); ax[1].grid(alpha=.3)

    # por dataset (líneas finas) para ver dispersión
    for ds in d.dataset.unique():
        s = d[d.dataset == ds].groupby("level").dAP.mean()
        ax[2].plot(s.index*100, s.values, "-", alpha=.5, lw=1)
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].set_xlabel(L("% de variables inventadas", "% invented features")); ax[2].set_ylabel("ΔAP vs 0%")
    ax[2].set_title(L("Por dataset", "Per dataset")); ax[2].grid(alpha=.3)
    plt.suptitle(L("Apartado 9 — feature engineering especulativo (variables random)",
                   "Section 9 — speculative feature engineering (random features)"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_feateng{SUF}.png", dpi=130); plt.close()

    # significancia: AP a 100% vs 0% (pareado); y tiempo
    d100 = d[d.level == 1.0].merge(d[d.level == 0.0][["dataset", "seed", "ap"]],
                                   on=["dataset", "seed"], suffixes=("_100", "_0"))
    delta = (d100.ap_100 - d100.ap_0).values
    pval = float(stats.ttest_1samp(delta, 0).pvalue) if len(delta) >= 3 else None
    out = {"n_datasets": int(d.dataset.nunique()),
           "ap_0": float(by[by.level == 0.0].ap.iloc[0]),
           "ap_100": float(by[by.level == 1.0].ap.iloc[0]) if (by.level == 1.0).any() else None,
           "dAP_100_mean": float(delta.mean()) if len(delta) else None,
           "dAP_100_p": pval, "dAP_100_winrate_pos": float((delta > 0).mean()) if len(delta) else None,
           "time_0": float(by[by.level == 0.0].t.iloc[0]),
           "time_100": float(by[by.level == 1.0].t.iloc[0]) if (by.level == 1.0).any() else None,
           "by_level": [{"level": float(r.level), "ap": round(float(r.ap), 4),
                         "dAP": round(float(r.dAP), 4), "time": round(float(r.t), 1)}
                        for _, r in by.iterrows()]}
    json.dump(out, open(f"{STUDY}/summary_feateng.json", "w"), indent=1)
    print(by.round(4).to_string(index=False)); print(json.dumps(out, indent=1)[:600])
    print("fig -> fig_feateng.png")


if __name__ == "__main__":
    main()
