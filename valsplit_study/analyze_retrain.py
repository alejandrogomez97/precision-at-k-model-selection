"""analyze_retrain.py — ¿ayuda el reentrenamiento del modelo final?

Lee results/grid2__*.json (que guarda, por celda y estrategia, el AP en test CON
reentrenamiento y SIN reentrenamiento) y compara las 4 variantes:
  E1-retrain  : refit del ganador en train+val
  E1-noretrain: modelo entrenado solo en train (val solo para elegir)
  E2-retrain  : refit del ganador en dev
  E2-noretrain: ensemble bagged de los k modelos de fold de la CV
Figura fig_retrain.png + summary_retrain.csv.
"""
import glob, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
import os
EN = os.environ.get("FIGLANG") == "en"
def L(es, en): return en if EN else es
SUF = "_en" if EN else ""


def load():
    rows = [r for f in glob.glob(f"{STUDY}/results/grid2__*.json")
            for r in json.load(open(f)) if "error" not in r and "test_ap_by_ap" in r]
    return pd.DataFrame(rows)


def main():
    df = load()
    if not len(df):
        print("grid2 sin datos todavía"); return
    df = df.dropna(subset=["test_ap_by_ap", "test_ap_nr_by_ap"])
    # solo tamaños con suficientes celdas pareadas (evita puntos ruidosos y garantiza
    # que retrain y no-retrain comparten EXACTAMENTE los mismos puntos)
    MIN_CELLS = 10
    per_size = df.groupby("size").size()
    keep = sorted(s for s in per_size.index if per_size[s] >= MIN_CELLS)
    df = df[df["size"].isin(keep)]
    print(f"grid2: {len(df)} celdas-estrategia, {df.dataset.nunique()} datasets, "
          f"seeds={sorted(df.seed.unique())} | tamaños usados (≥{MIN_CELLS} celdas): {keep}")

    # tabla larga: variante -> AP
    recs = []
    for _, r in df.iterrows():
        base = dict(dataset=r.dataset, size=r["size"], seed=r.seed, imb=r.imb_ratio)
        recs.append({**base, "variant": f"{r.strat}-retrain",   "ap": r.test_ap_by_ap})
        recs.append({**base, "variant": f"{r.strat}-noretrain", "ap": r.test_ap_nr_by_ap})
    LF = pd.DataFrame(recs)
    agg = LF.groupby(["variant", "size"]).ap.mean().reset_index()
    agg.to_csv(f"{STUDY}/summary_retrain.csv", index=False)

    # diferencia pareada retrain - noretrain por estrategia
    print("\n=== ¿ayuda reentrenar? (retrain − noretrain, pareado) ===")
    for strat in ("E1", "E2"):
        s = df[df.strat == strat]
        d = (s["test_ap_by_ap"] - s["test_ap_nr_by_ap"]).values
        t = stats.ttest_1samp(d, 0)
        print(f"  {strat}: mean Δ={d.mean():+.4f}  winrate_retrain={(d>0).mean():.3f}  "
              f"t={t.statistic:.2f} p={t.pvalue:.1e}  n={len(d)}")

    # figura
    colors = {"E1-retrain": "#1f77b4", "E1-noretrain": "#7fb2d6",
              "E2-retrain": "#c0392b", "E2-noretrain": "#e08e83"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for v in ["E1-retrain", "E1-noretrain", "E2-retrain", "E2-noretrain"]:
        a = agg[agg.variant == v].sort_values("size")
        ls = "-" if "retrain" == v.split("-")[1] else "--"
        ax[0].plot(a["size"], a.ap, marker="o", ls=ls, color=colors[v], label=v)
    ax[0].set_xscale("log"); ax[0].set_xlabel(L("tamaño dev", "dev size")); ax[0].set_ylabel(L("AP test", "test AP"))
    ax[0].set_title(L("Las 4 variantes", "The 4 variants")); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    for strat, col in (("E1", "#1f77b4"), ("E2", "#c0392b")):
        s = df[df.strat == strat]
        g = s.assign(d=s["test_ap_by_ap"] - s["test_ap_nr_by_ap"]).groupby("size").agg(
            md=("d", "mean"), se=("d", lambda x: x.std()/np.sqrt(len(x)))).reset_index()
        ax[1].errorbar(g["size"], g.md, yerr=g.se, fmt="o-", color=col,
                       capsize=3, label=strat)
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xscale("log")
    ax[1].set_xlabel(L("tamaño dev", "dev size")); ax[1].set_ylabel("AP(retrain) − AP(no-retrain)")
    ax[1].set_title(L("¿Ayuda reentrenar el modelo final?", "Does retraining the final model help?")); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.suptitle(L("Reentrenamiento del modelo final tras la selección",
                   "Retraining the final model after selection"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_retrain{SUF}.png", dpi=130); plt.close()
    print(f"\nfig -> fig_retrain{SUF}.png")


if __name__ == "__main__":
    main()
