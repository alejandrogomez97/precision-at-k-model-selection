"""analyze_hpo.py — ¿ayuda más presupuesto de HPO, y qué método es mejor?

Lee results/hpo__*.json (method × budget × test_ap × search_time) y produce
fig_hpo.png: AP en test vs nº de configuraciones, una curva por método (grid,
random, TPE, CMA-ES), para fracción baja (0.3) y alta (1.0); más panel de tiempo.
summary_hpo.csv / summary_hpo.json para el artículo.
"""
import glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
COL = {"grid": "#1f77b4", "random": "#7f7f7f", "tpe": "#ff7f0e", "cmaes": "#2ca02c",
       "gp": "#9467bd", "hyperband": "#e377c2", "bohb": "#17becf"}
ORDER = ["grid", "random", "tpe", "cmaes", "gp", "hyperband", "bohb"]
import os
EN = os.environ.get("FIGLANG") == "en"
def L(es, en): return en if EN else es
SUF = "_en" if EN else ""


def main():
    tag = os.environ.get("HPO_TAG", "hpo")
    rows = [r for f in glob.glob(f"{STUDY}/results/{tag}__*.json") for r in json.load(open(f))]
    if not rows:
        print("hpo sin datos"); return
    d = pd.DataFrame(rows)
    # PANEL BALANCEADO: solo celdas (dataset,frac,seed) presentes en TODOS los métodos
    # (a máximo presupuesto), para no comparar medias sobre subconjuntos distintos.
    nm = d.method.nunique()
    cell = d[d.budget == d.budget.max()].groupby(["dataset", "frac", "seed"]).method.nunique()
    keep = set(cell[cell == nm].index)
    d = d[d.set_index(["dataset", "frac", "seed"]).index.isin(keep)].copy()
    print(f"hpo: {d.dataset.nunique()} datasets, métodos={sorted(d.method.unique())}, "
          f"budgets={sorted(d.budget.unique())}, celdas balanceadas={len(keep)}")

    fracs = sorted(d.frac.unique())
    fig, axes = plt.subplots(1, len(fracs) + 1, figsize=(6 * (len(fracs) + 1), 5))
    for ax, fr in zip(axes, fracs):
        sub = d[d.frac == fr]
        for meth in ORDER:
            s = sub[sub.method == meth]
            if not len(s): continue
            g = s.groupby("budget").test_ap.mean()
            ax.plot(g.index, g.values, "o-", color=COL[meth], label=meth)
        ax.set_xscale("log"); ax.set_xlabel(L("nº de configuraciones", "# configurations"))
        ax.set_ylabel("AP (test)"); ax.set_title(L(f"AP vs presupuesto — fracción {fr:.0%}", f"AP vs budget — fraction {fr:.0%}"))
        ax.legend(); ax.grid(alpha=.3)
    # panel de tiempo (frac alta)
    ax = axes[-1]
    sub = d[d.frac == max(fracs)]
    for meth in ORDER:
        s = sub[sub.method == meth]
        if not len(s): continue
        g = s.groupby("budget").search_time.mean()
        ax.plot(g.index, g.values, "o-", color=COL[meth], label=meth)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel(L("nº de configuraciones", "# configurations"))
    ax.set_ylabel(L("tiempo búsqueda (s)", "search time (s)")); ax.set_title(L(f"Tiempo — fracción {max(fracs):.0%}", f"Time — fraction {max(fracs):.0%}"))
    ax.legend(); ax.grid(alpha=.3)
    plt.suptitle(L("HPO — ¿mejora con más presupuesto? ¿qué método?",
                   "HPO — does more budget help? which method?"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_hpo{SUF}.png", dpi=130); plt.close()

    # resumen: AP y tiempo medios por método y presupuesto
    piv = d.pivot_table(index="budget", columns="method", values="test_ap", aggfunc="mean")
    piv.to_csv(f"{STUDY}/summary_hpo.csv")
    d.pivot_table(index="budget", columns="method", values="search_time",
                  aggfunc="mean").to_csv(f"{STUDY}/summary_hpo_time.csv")
    # ¿mejora del mejor método sobre grid, al mayor presupuesto?
    from scipy import stats
    out = {}
    maxb = d.budget.max(); minb = d.budget.min()
    # comparación pareada de cada método vs grid al máximo presupuesto
    pw = d[d.budget == maxb].pivot_table(index=["dataset", "frac", "seed"],
                                         columns="method", values="test_ap")
    for meth in d.method.unique():
        at_max = d[(d.method == meth) & (d.budget == maxb)].test_ap.mean()
        at_min = d[(d.method == meth) & (d.budget == minb)].test_ap.mean()
        rec = {"ap_min_budget": round(float(at_min), 4),
               "ap_max_budget": round(float(at_max), 4),
               "gain_budget": round(float(at_max - at_min), 4)}
        if meth != "grid" and "grid" in pw.columns and meth in pw.columns:
            delta = (pw[meth] - pw["grid"]).dropna().values
            if len(delta) >= 3:
                rec["vs_grid_mean"] = round(float(delta.mean()), 4)
                rec["vs_grid_p"] = float(stats.ttest_1samp(delta, 0).pvalue)
                rec["vs_grid_winrate"] = round(float((delta > 0).mean()), 3)
                rec["vs_grid_n"] = int(len(delta))
        out[meth] = rec
    json.dump(out, open(f"{STUDY}/summary_hpo.json", "w"), indent=1)
    print(piv.round(4).to_string())
    print(json.dumps(out, indent=1))
    print("fig -> fig_hpo.png")


if __name__ == "__main__":
    main()
