"""analyze_balanced.py — Regenera las figuras de tamaño usando un PANEL BALANCEADO
(mismo conjunto de datasets en TODOS los tamaños) y desde UN SOLO run (grid2, que
tiene las 4 variantes). Así:
  - la curva sube de forma monótona (sin artefacto de composición), y
  - las curvas de 'con reentrenamiento' (E1-retrain, E2-retrain) son IDÉNTICAS en
    la figura E1-vs-E2 y en la figura de las 4 variantes.

Genera fig_A_story.png (E1 vs E2) y fig_retrain.png (4 variantes) de forma coherente.
Solo usa los tamaños en los que TODOS los datasets del panel tienen dato (para las
semillas disponibles), de modo que cada punto compara el mismo conjunto.
"""
import glob, json, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
BAL = open("/tmp/balanced.txt").read().split() if __import__("os").path.exists("/tmp/balanced.txt") else []


def load():
    df = pd.DataFrame([r for f in glob.glob(f"{STUDY}/results/grid2__*.json")
                       for r in json.load(open(f)) if "error" not in r])
    return df[df.dataset.isin(BAL)].copy()


def balanced_sizes(df):
    """Tamaños en los que están los N datasets del panel (mismas condiciones)."""
    n_target = df.dataset.nunique()
    ok = []
    for s in sorted(df["size"].unique()):
        if df[df["size"] == s].dataset.nunique() == n_target:
            ok.append(s)
    return ok, n_target


def agg_variants(df, sizes):
    """AP medio por tamaño para las 4 variantes (media sobre datasets y semillas)."""
    d = df[df["size"].isin(sizes)]
    out = {}
    for strat in ("E1", "E2"):
        s = d[d.strat == strat]
        out[f"{strat}-retrain"] = s.groupby("size").test_ap_by_ap.mean()
        out[f"{strat}-noretrain"] = s.groupby("size").test_ap_nr_by_ap.mean()
    return pd.DataFrame(out)


def main():
    df = load()
    if df.empty:
        print("sin datos del panel todavía"); return
    sizes, n = balanced_sizes(df)
    if len(sizes) < 2:
        print(f"panel aún incompleto: {df.dataset.nunique()} datasets, "
              f"tamaños completos={sizes}"); return
    print(f"PANEL BALANCEADO: {n} datasets, seeds={sorted(df.seed.unique())}, "
          f"tamaños completos={sizes}")
    A = agg_variants(df, sizes)
    print(A.round(4).to_string())

    C1, C2 = "#1f77b4", "#c0392b"

    # -------- FIGURA 1: E1 vs E2 (con reentrenamiento) --------
    dd = df[df["size"].isin(sizes)]
    p = dd.pivot_table(index=["dataset", "size", "seed"], columns="strat",
                       values="test_ap_by_ap").reset_index().dropna(subset=["E1", "E2"])
    p["diff"] = p.E1 - p.E2
    g = p.groupby("size").agg(md=("diff", "mean"),
                              se=("diff", lambda x: x.std()/np.sqrt(len(x)))).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(A.index, A["E1-retrain"], "o-", color=C1, label="E1 (val separado)")
    ax[0].plot(A.index, A["E2-retrain"], "s--", color=C2, label="E2 (solo CV)")
    ax[0].set_xscale("log"); ax[0].set_xlabel("tamaño del pool de desarrollo")
    ax[0].set_ylabel("Average Precision (test)")
    ax[0].set_title("E1 vs E2 (con reentrenamiento)"); ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].axhline(0, color="k", lw=.8)
    ax[1].errorbar(g["size"], g.md, yerr=g.se, fmt="o-", color="#2ca02c", capsize=3)
    ax[1].set_xscale("log"); ax[1].set_xlabel("tamaño del pool de desarrollo")
    ax[1].set_ylabel("AP(E1) − AP(E2)")
    ax[1].set_title("Diferencia pareada E1 − E2"); ax[1].grid(alpha=.3)
    plt.suptitle(f"E1 vs E2 — panel balanceado ({n} datasets en todos los tamaños)",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_A_story.png", dpi=135); plt.close()

    # -------- FIGURA 2: las 4 variantes (mismas curvas retrain que arriba) --------
    colors = {"E1-retrain": C1, "E1-noretrain": "#7fb2d6",
              "E2-retrain": C2, "E2-noretrain": "#e08e83"}
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for v in ["E1-retrain", "E1-noretrain", "E2-retrain", "E2-noretrain"]:
        ls = "-" if v.endswith("retrain") and "noretrain" not in v else "--"
        ax[0].plot(A.index, A[v], marker="o", ls=ls, color=colors[v], label=v)
    ax[0].set_xscale("log"); ax[0].set_xlabel("tamaño del pool de desarrollo")
    ax[0].set_ylabel("AP test"); ax[0].set_title("Las 4 variantes")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for strat, col in (("E1", C1), ("E2", C2)):
        s = dd[dd.strat == strat]
        gg = s.assign(d=s.test_ap_by_ap - s.test_ap_nr_by_ap).groupby("size").agg(
            md=("d", "mean"), se=("d", lambda x: x.std()/np.sqrt(len(x)))).reset_index()
        ax[1].errorbar(gg["size"], gg.md, yerr=gg.se, fmt="o-", color=col, capsize=3, label=strat)
    ax[1].axhline(0, color="k", lw=.8); ax[1].set_xscale("log")
    ax[1].set_xlabel("tamaño del pool de desarrollo"); ax[1].set_ylabel("AP(retrain) − AP(noretrain)")
    ax[1].set_title("¿Ayuda reentrenar?"); ax[1].legend(); ax[1].grid(alpha=.3)
    plt.suptitle(f"Reentrenamiento — panel balanceado ({n} datasets)", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_retrain.png", dpi=135); plt.close()

    # stats
    print("\n=== stats (panel balanceado) ===")
    dstat = p["diff"].values
    print(f"E1 vs E2: mean={dstat.mean():+.4f} p={stats.ttest_1samp(dstat,0).pvalue:.1e} n={len(dstat)}")
    for strat in ("E1", "E2"):
        s = dd[dd.strat == strat]
        d = (s.test_ap_by_ap - s.test_ap_nr_by_ap).values
        print(f"retrain {strat}: mean={d.mean():+.4f} p={stats.ttest_1samp(d,0).pvalue:.1e} n={len(d)}")
    print("\nfiguras -> fig_A_story.png, fig_retrain.png (coherentes, panel balanceado)")


if __name__ == "__main__":
    main()
