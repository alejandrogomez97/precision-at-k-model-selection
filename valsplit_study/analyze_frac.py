"""analyze_frac.py — Figuras y resumen del estudio con barrido por FRACCIÓN.

Lee results/gridf__*.json (E1/E2 × retrain/no-retrain, barrido por fracción del pool
de desarrollo). Como todos los datasets están en todos los puntos de fracción, la
curva agregada no sufre sesgo de composición y las curvas de 'con reentrenamiento'
son idénticas en ambas figuras (salen del mismo run).

Genera:
  fig_A_story.png  — E1 vs E2 (AP vs fracción; diferencia pareada; vs desbalanceo)
  fig_retrain.png  — 4 variantes (mismas curvas retrain) + efecto del reentrenamiento
  summary_frac.json — números agregados y estadísticos para el artículo
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
C1, C2, CD = "#1f77b4", "#c0392b", "#2ca02c"


def load():
    df = pd.DataFrame([r for f in glob.glob(f"{STUDY}/results/gridf__*.json")
                       for r in json.load(open(f)) if "error" not in r
                       and r.get("test_ap_by_ap") is not None])
    return df


def balanced(df):
    """Datasets presentes en TODAS las fracciones (para curvas coherentes)."""
    fr = sorted(df["frac"].unique())
    present = df.groupby("dataset")["frac"].apply(lambda s: set(s.unique()))
    keep = [d for d, ss in present.items() if set(fr).issubset(ss)]
    return df[df.dataset.isin(keep)], fr, len(keep)


def main():
    df = load()
    if df.empty:
        print("gridf sin datos"); return
    df, fracs, ndb = balanced(df)
    if ndb == 0:
        print("aún no hay datasets con todas las fracciones"); return
    print(f"gridf: {df.dataset.nunique()} datasets (panel completo), "
          f"seeds={sorted(df.seed.unique())}, fracciones={fracs}")

    out = {"n_datasets": int(df.dataset.nunique()), "fracs": fracs,
           "seeds": sorted(int(s) for s in df.seed.unique())}

    # agregados por fracción
    A = {}
    for strat in ("E1", "E2"):
        s = df[df.strat == strat]
        A[f"{strat}-retrain"] = s.groupby("frac").test_ap_by_ap.mean()
        A[f"{strat}-noretrain"] = s.groupby("frac").test_ap_nr_by_ap.mean()
    A = pd.DataFrame(A)
    out["ap_by_frac"] = {k: {float(f): round(v, 4) for f, v in A[k].items()} for k in A}
    # tiempos de búsqueda por estrategia y fracción (para ver iso-tiempo)
    if "search_time" in df.columns:
        tt = df.groupby(["strat", "frac"]).search_time.mean()
        out["time_by_frac"] = {s: {float(f): round(float(tt[s][f]), 1) for f in tt[s].index}
                               for s in tt.index.levels[0] if s in tt.index.get_level_values(0)}
        out["time_overall"] = {s: round(float(df[df.strat == s].search_time.mean()), 1)
                               for s in ("E1", "E2")}

    # ---------- FIGURA 1: E1 vs E2 (con reentrenamiento) ----------
    p = df.pivot_table(index=["dataset", "frac", "seed"], columns="strat",
                       values="test_ap_by_ap").reset_index().dropna(subset=["E1", "E2"])
    p["diff"] = p.E1 - p.E2
    g = p.groupby("frac").agg(md=("diff", "mean"),
                              se=("diff", lambda x: x.std()/np.sqrt(len(x)))).reset_index()
    ir = df[["dataset", "imb_ratio"]].drop_duplicates("dataset")
    p = p.merge(ir, on="dataset", how="left")
    p["irb"] = pd.cut(p["imb_ratio"], [0, 5, 10, 20, 1e9],
                      labels=["<5", "5–10", "10–20", ">20"])
    gi = p.groupby("irb", observed=True).agg(md=("diff", "mean"),
            se=("diff", lambda x: x.std()/np.sqrt(len(x)))).reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))
    ax[0].plot(A.index, A["E1-retrain"], "o-", color=C1, label=L("E1 (val separado)", "E1 (held-out val)"))
    ax[0].plot(A.index, A["E2-retrain"], "s--", color=C2, label=L("E2 (solo CV)", "E2 (CV only)"))
    ax[0].set_xlabel(L("fracción del pool de desarrollo", "development pool fraction")); ax[0].set_ylabel("AP (test)")
    ax[0].set_title(L("E1 vs E2 (con reentrenamiento)", "E1 vs E2 (with retraining)")); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].errorbar(g["frac"], g.md, yerr=g.se, fmt="o-", color=CD, capsize=3)
    ax[1].set_xlabel(L("fracción del pool de desarrollo", "development pool fraction")); ax[1].set_ylabel("AP(E1) − AP(E2)")
    ax[1].set_title(L("Diferencia pareada E1 − E2", "Paired difference E1 − E2")); ax[1].grid(alpha=.3)
    ax[2].axhline(0, color="k", lw=.8)
    ax[2].bar(range(len(gi)), gi.md, yerr=gi.se, color=CD, alpha=.7, capsize=3)
    ax[2].set_xticks(range(len(gi))); ax[2].set_xticklabels(gi.irb)
    ax[2].set_xlabel(L("índice de desbalanceo", "imbalance ratio")); ax[2].set_ylabel("AP(E1) − AP(E2)")
    ax[2].set_title(L("E1 − E2 vs desbalanceo", "E1 − E2 vs imbalance")); ax[2].grid(alpha=.3)
    plt.suptitle(L(f"E1 vs E2 — barrido por fracción ({df.dataset.nunique()} datasets, todos en cada punto)",
                   f"E1 vs E2 — fraction sweep ({df.dataset.nunique()} datasets, all present at every point)"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_A_story{SUF}.png", dpi=135); plt.close()

    d = p["diff"].values
    out["E1_vs_E2"] = {"mean": float(d.mean()), "p": float(stats.ttest_1samp(d, 0).pvalue),
                       "winrate_E1": float((d > 0).mean()), "n": int(len(d))}
    # cruce: E1-E2 en fracciones BAJAS vs ALTAS (aísla la hipótesis)
    fr_all = sorted(p["frac"].unique())
    lo_th, hi_th = fr_all[0], fr_all[-1]
    lo = p[p["frac"] <= (0.3 if max(fr_all) > 0.3 else lo_th)]["diff"].values
    hi = p[p["frac"] >= 0.8]["diff"].values
    out["crossover"] = {
        "low": {"mean": float(lo.mean()), "winrate_E1": float((lo > 0).mean()),
                "p": float(stats.ttest_1samp(lo, 0).pvalue) if len(lo) > 1 else None, "n": int(len(lo))},
        "high": {"mean": float(hi.mean()), "winrate_E1": float((hi > 0).mean()),
                 "p": float(stats.ttest_1samp(hi, 0).pvalue) if len(hi) > 1 else None, "n": int(len(hi))},
    }

    # ---------- FIGURA 2: 4 variantes + efecto reentrenamiento ----------
    colors = {"E1-retrain": C1, "E1-noretrain": "#7fb2d6",
              "E2-retrain": C2, "E2-noretrain": "#e08e83"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for v in ["E1-retrain", "E1-noretrain", "E2-retrain", "E2-noretrain"]:
        ls = "--" if v.endswith("noretrain") else "-"
        ax[0].plot(A.index, A[v], marker="o", ls=ls, color=colors[v], label=v)
    ax[0].set_xlabel(L("fracción del pool de desarrollo", "development pool fraction")); ax[0].set_ylabel("AP (test)")
    ax[0].set_title(L("Las 4 variantes", "The 4 variants")); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    out["retrain"] = {}
    for strat, col in (("E1", C1), ("E2", C2)):
        s = df[df.strat == strat]
        gg = s.assign(dd=s.test_ap_by_ap - s.test_ap_nr_by_ap).groupby("frac").agg(
            md=("dd", "mean"), se=("dd", lambda x: x.std()/np.sqrt(len(x)))).reset_index()
        ax[1].errorbar(gg["frac"], gg.md, yerr=gg.se, fmt="o-", color=col, capsize=3, label=strat)
        dd = (s.test_ap_by_ap - s.test_ap_nr_by_ap).values
        out["retrain"][strat] = {"mean": float(dd.mean()),
                                 "p": float(stats.ttest_1samp(dd, 0).pvalue),
                                 "winrate_retrain": float((dd > 0).mean()), "n": int(len(dd))}
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].set_xlabel(L("fracción del pool de desarrollo", "development pool fraction")); ax[1].set_ylabel("AP(retrain) − AP(no-retrain)")
    ax[1].set_title(L("¿Ayuda reentrenar?", "Does retraining help?")); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.suptitle(L(f"Reentrenamiento — barrido por fracción ({df.dataset.nunique()} datasets)",
                   f"Retraining — fraction sweep ({df.dataset.nunique()} datasets)"),
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_retrain{SUF}.png", dpi=135); plt.close()

    json.dump(out, open(f"{STUDY}/summary_frac.json", "w"), indent=1)
    print("E1 vs E2:", out["E1_vs_E2"])
    print("retrain:", out["retrain"])
    print("figuras -> fig_A_story.png, fig_retrain.png ; summary_frac.json")


if __name__ == "__main__":
    main()
