"""analyze_bc.py — Figuras de las fases B (Optuna vs grid) y C (familias+ensembles).

Fase B: compara Optuna vs grid en AP de test y tiempo de búsqueda, por estrategia
        y tamaño (sobre los (dataset,size,seed) comunes).
Fase C: curva anytime del pool+ensemble (AP de test vs tiempo acumulado) frente a
        la referencia E1-grid (AP y tiempo de búsqueda) para el mismo (ds,size,seed).
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
import os as _os
EN = _os.environ.get("FIGLANG") == "en"
def L(es, en): return en if EN else es
SUF = "_en" if EN else ""
RES = f"{STUDY}/results"


def load_tag(tag):
    rows = []
    for fp in glob.glob(f"{RES}/{tag}__*.json"):
        for r in json.load(open(fp)):
            if "error" not in r:
                rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
#  FASE B — Optuna vs grid
# --------------------------------------------------------------------------- #
def phase_b():
    keys = ["dataset", "frac", "seed", "strat"]
    g = load_tag("gridf")
    o40 = load_tag("optunaf"); o18 = load_tag("optuna18f")
    if not len(g) or (not len(o40) and not len(o18)):
        print("[B] faltan resultados de optuna todavía"); return
    g = g[keys + ["test_ap_by_ap", "search_time"]].rename(
        columns={"test_ap_by_ap": "ap_grid", "search_time": "t_grid"})

    def merge(o, sfx):
        if not len(o): return None
        oo = o[keys + ["test_ap_by_ap", "search_time"]].rename(
            columns={"test_ap_by_ap": f"ap_{sfx}", "search_time": f"t_{sfx}"})
        return g.merge(oo, on=keys)

    m18 = merge(o18, "o18"); m40 = merge(o40, "o40")
    base = m18 if m18 is not None else m40
    # unir ambos presupuestos sobre las celdas comunes al grid
    M = base
    if m18 is not None and m40 is not None:
        M = m18.merge(m40[keys + ["ap_o40", "t_o40"]], on=keys, how="outer")
        # ap_grid/t_grid pueden faltar en filas solo-o40; rellenar desde g
        M = M.drop(columns=[c for c in ["ap_grid", "t_grid"] if c in M]).merge(
            g, on=keys, how="left")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    # panel 1: AP medio por fracción (promediado sobre E1/E2)
    def curve(col):
        return M.dropna(subset=[col]).groupby("frac")[col].mean()
    axes[0].plot(curve("ap_grid").index, curve("ap_grid").values, "o-", color="#1f77b4",
                 label="grid (18 configs)")
    if "ap_o18" in M:
        axes[0].plot(curve("ap_o18").index, curve("ap_o18").values, "s--", color="#ff7f0e",
                     label=L("optuna (18 trials) — iso-presupuesto", "optuna (18 trials) — equal budget"))
    if "ap_o40" in M:
        axes[0].plot(curve("ap_o40").index, curve("ap_o40").values, "^:", color="#d62728",
                     label="optuna (40 trials)")
    axes[0].set_title(L("AP test — mismo nº de configuraciones (18)", "test AP — same # of configurations (18)"))
    axes[0].set_xlabel(L("fracción dev", "dev fraction")); axes[0].set_ylabel("AP (test)")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=.3)

    # panel 2: ganancia vs grid
    axes[1].axhline(0, color="k", lw=.8)
    if "ap_o18" in M:
        gg = M.assign(d=M.ap_o18 - M.ap_grid).dropna(subset=["d"]).groupby("frac").d.mean()
        axes[1].plot(gg.index, gg.values, "s-", color="#ff7f0e", label="optuna-18 − grid")
    if "ap_o40" in M:
        gg = M.assign(d=M.ap_o40 - M.ap_grid).dropna(subset=["d"]).groupby("frac").d.mean()
        axes[1].plot(gg.index, gg.values, "^-", color="#d62728", label="optuna-40 − grid")
    axes[1].set_title(L("Ganancia de AP sobre el grid", "AP gain over the grid")); axes[1].set_xlabel(L("fracción dev", "dev fraction"))
    axes[1].set_ylabel("Δ AP"); axes[1].legend(); axes[1].grid(alpha=.3)

    # panel 3: tiempos
    for col, lab, st in (("t_grid", "grid (18)", "o-"), ("t_o18", "optuna-18", "s-"),
                         ("t_o40", "optuna-40", "^-")):
        if col in M:
            tt = M.dropna(subset=[col]).groupby("frac")[col].mean()
            axes[2].plot(tt.index, tt.values, st, label=lab)
    axes[2].set_yscale("log"); axes[2].set_title(L("Tiempo de búsqueda (s)", "Search time (s)"))
    axes[2].set_xlabel(L("fracción dev", "dev fraction")); axes[2].legend(); axes[2].grid(alpha=.3)
    plt.suptitle(L("Fase B — Optuna vs Grid a igual nº de configuraciones",
                   "Phase B — Optuna vs Grid at the same # of configurations"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_B_optuna{SUF}.png", dpi=130); plt.close()

    # resumen para el artículo (iso-presupuesto 18 vs 18)
    out = {}
    if "ap_o18" in M:
        dd = (M.ap_o18 - M.ap_grid).dropna()
        out["iso18"] = {"gain": float(dd.mean()), "winrate": float((dd > 0).mean()), "n": int(len(dd))}
    if "ap_o40" in M:
        dd = (M.ap_o40 - M.ap_grid).dropna()
        out["b40"] = {"gain": float(dd.mean()), "winrate": float((dd > 0).mean()), "n": int(len(dd))}
    import json as _j; _j.dump(out, open(f"{STUDY}/summary_B.json", "w"), indent=1)
    # csv por fracción (para tabla)
    summ = M.groupby("frac").agg(
        ap_grid=("ap_grid", "mean"),
        ap_o18=("ap_o18", "mean") if "ap_o18" in M else ("ap_grid", "mean"),
        ap_o40=("ap_o40", "mean") if "ap_o40" in M else ("ap_grid", "mean"),
        t_grid=("t_grid", "mean"),
        t_o18=("t_o18", "mean") if "t_o18" in M else ("t_grid", "mean"),
        t_o40=("t_o40", "mean") if "t_o40" in M else ("t_grid", "mean")).reset_index()
    summ.to_csv(f"{STUDY}/summary_B.csv", index=False)
    print(out); print(summ.round(4).to_string(index=False)); print("fig -> fig_B_optuna.png")


# --------------------------------------------------------------------------- #
#  FASE C — familias + ensembles (anytime) vs E1-grid
# --------------------------------------------------------------------------- #
def phase_c():
    f = load_tag("familiesf"); g = load_tag("gridf")
    if not len(f):
        print("[C] sin resultados de familias todavía"); return
    gE1 = g[g.strat == "E1"].set_index(["dataset", "frac", "seed"])

    # referencia E1-grid por celda
    recs = []
    curves = {}  # size -> list of (time_grid, curves...) para promediar en malla
    for _, row in f.iterrows():
        key = (row["dataset"], row["frac"], row["seed"])
        if key not in gE1.index or "anytime" not in row or not isinstance(row["anytime"], list):
            continue
        ref_ap = float(gE1.loc[key, "test_ap_by_ap"])
        ref_t = float(gE1.loc[key, "search_time"])
        at = row["anytime"]
        # tiempo hasta que el ensemble iguala/supera E1-grid
        t_beat = next((a["cum_time"] for a in at if a["ens_test"] >= ref_ap), None)
        recs.append({
            "dataset": row["dataset"], "frac": row["frac"], "seed": row["seed"],
            "ref_ap": ref_ap, "ref_time": ref_t,
            "ens_final_test": at[-1]["ens_test"], "ens_final_val": at[-1]["ens_val"],
            "single_final_test": at[-1]["best_single_test"],
            "single_best_name": at[-1]["best_single_name"],
            "ens_total_time": at[-1]["cum_time"],
            "t_beat_ref": t_beat,
            "beat_ref": at[-1]["ens_test"] >= ref_ap,
            "beat_in_less_time": (t_beat is not None and t_beat <= ref_t),
        })
    d = pd.DataFrame(recs)
    if not len(d):
        print("[C] sin celdas comunes con E1-grid todavía"); return
    d.to_csv(f"{STUDY}/summary_C.csv", index=False)

    # fig: por tamaño, AP de ensemble vs E1-grid + fracción que gana en menos tiempo
    by = d.groupby("frac").agg(
        ens=("ens_final_test", "mean"), single=("single_final_test", "mean"),
        e1=("ref_ap", "mean"), winrate=("beat_ref", "mean"),
        win_less_time=("beat_in_less_time", "mean"),
        ens_time=("ens_total_time", "mean"), e1_time=("ref_time", "mean"),
        n=("ens_final_test", "size")).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].plot(by["frac"], by["e1"], "o-", label="E1-grid LGBM")
    axes[0].plot(by["frac"], by["single"], "^-", label=L("mejor familia individual", "best single family"))
    axes[0].plot(by["frac"], by["ens"], "s-", label="ensemble (Caruana)")
    axes[0].set_title(L("AP test", "test AP")); axes[0].set_xlabel(L("fracción dev", "dev fraction"))
    axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(by["frac"], by["winrate"], "o-", color="#9467bd", label="ens ≥ E1 (test)")
    axes[1].plot(by["frac"], by["win_less_time"], "s-", color="#2ca02c",
                 label=L("ens ≥ E1 en ≤ tiempo E1", "ens ≥ E1 in ≤ E1's time"))
    axes[1].axhline(.5, color="k", lw=.8, ls="--")
    axes[1].set_ylim(0, 1); axes[1].set_title(L("Fracción de celdas", "Fraction of cells")); axes[1].set_xlabel(L("fracción dev", "dev fraction"))
    axes[1].legend(fontsize=8); axes[1].grid(alpha=.3)

    axes[2].scatter(d["ens_final_val"], d["ens_final_test"], s=14, alpha=.5)
    lim = [min(d.ens_final_val.min(), d.ens_final_test.min()),
           max(d.ens_final_val.max(), d.ens_final_test.max())]
    axes[2].plot(lim, lim, "k--", lw=.8)
    axes[2].set_xlabel(L("AP ensemble en val", "ensemble AP on val")); axes[2].set_ylabel(L("AP ensemble en test", "ensemble AP on test"))
    axes[2].set_title(L("¿Consolida en test?", "Does it hold on test?")); axes[2].grid(alpha=.3)
    plt.suptitle(L("Fase C — familias+ensembles vs E1-grid",
                   "Phase C — families+ensembles vs E1-grid"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_C_families{SUF}.png", dpi=130); plt.close()
    print(by.to_string(index=False))
    print("fig -> fig_C_families.png")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    if which in ("b", "both"):
        print("=== FASE B ==="); phase_b()
    if which in ("c", "both"):
        print("\n=== FASE C ==="); phase_c()
