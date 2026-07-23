"""Genera las tablas del artículo como IMÁGENES PNG con celdas coloreadas (mejor
verde / peor rojo por fila), para subir a Medium. Salida en medium/assets/."""
import json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
A = f"{STUDY}/medium/assets"
GREEN = "#c8ecd6"; RED = "#f6d0cb"; HEAD = "#2f3640"; GRID = "#e6e8ec"


def render(fname, title, collabels, rows, color_groups, note=""):
    """rows: lista de listas (strings). color_groups: lista de (col_indices, higher_better)
    -> por fila colorea el mejor (verde) y el peor (rojo) de esas columnas (según valor
    numérico parseado)."""
    ncol = len(collabels); nrow = len(rows)
    fig_h = 0.9 + 0.34 * (nrow + 1) + (0.4 if note else 0)
    fig_w = max(6, 1.5 * ncol)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h)); ax.axis("off")
    top = 0.86 if note else 0.90
    tbl = ax.table(cellText=rows, colLabels=collabels, cellLoc="center",
                   bbox=[0, (0.10 if note else 0.0), 1, top])
    tbl.auto_set_font_size(False); tbl.set_fontsize(11)
    # cabecera
    for c in range(ncol):
        cell = tbl[0, c]; cell.set_facecolor(HEAD); cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor(GRID)
    for r in range(1, nrow + 1):
        for c in range(ncol):
            tbl[r, c].set_edgecolor(GRID)
    # colorear por grupo
    for cols, hib in color_groups:
        for ri, row in enumerate(rows):
            vals = []
            for c in cols:
                try: vals.append(float(str(row[c]).replace("s", "").replace("+", "")))
                except Exception: vals.append(np.nan)
            arr = [v for v in vals if v == v]
            if len(set(arr)) < 2: continue
            best = max(arr) if hib else min(arr); worst = min(arr) if hib else max(arr)
            for c, v in zip(cols, vals):
                if v != v: continue
                if v == best: tbl[ri + 1, c].set_facecolor(GREEN)
                elif v == worst: tbl[ri + 1, c].set_facecolor(RED)
    ax.set_title(title, fontweight="bold", fontsize=13, pad=14)
    if note:
        fig.text(0.5, 0.02, note, ha="center", fontsize=9, style="italic", color="#555")
    plt.tight_layout(); plt.savefig(f"{A}/{fname}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); print(fname, "escrito")


def pct(x): return f"{x*100:.0f}%"

# ---- Tabla 1: E1 vs E2 por fracción ----
sf = json.load(open(f"{STUDY}/summary_frac.json")); ap = sf["ap_by_frac"]; tb = sf.get("time_by_frac", {})
def g(d, k, f): return d.get(k, {}).get(f, d.get(k, {}).get(str(f)))
rows = []
for f in sf["fracs"]:
    f = float(f); e1 = g(ap, "E1-retrain", f); e2 = g(ap, "E2-retrain", f)
    t1 = g(tb, "E1", f); t2 = g(tb, "E2", f)
    rows.append([pct(f), f"{e1:.3f}", f"{e2:.3f}", f"{e1-e2:+.4f}", f"{t1:.0f}s", f"{t2:.0f}s"])
render("table1_e1_vs_e2.png", "E1 vs E2 by data fraction",
       ["fraction", "AP E1", "AP E2", "ΔAP (E1−E2)", "t E1", "t E2"], rows,
       [([1, 2], True), ([4, 5], False)],
       f"Global ΔAP(E1−E2) = {sf['E1_vs_E2']['mean']:+.4f}, p = {sf['E1_vs_E2']['p']:.0e}, n = {sf['E1_vs_E2']['n']}. Green = better / faster.")

HPO_ORDER = ["grid", "random", "tpe", "cmaes", "gp", "hyperband", "bohb"]
# ---- Tabla 2: HPO AP por presupuesto y método ----
s = pd.read_csv(f"{STUDY}/summary_hpo.csv", index_col=0)
ms = [m for m in HPO_ORDER if m in s.columns]
rows = [[str(int(b))] + [f"{s.loc[b, m]:.3f}" for m in ms] for b in s.index]
render("table2_hpo.png", "HPO methods: test AP by budget",
       ["# configs"] + ms, rows, [(list(range(1, len(ms) + 1)), True)],
       "Green = best method at that budget. No method beats the grid significantly (all p > 0.2) "
       "— including the multi-fidelity ones (hyperband, bohb) and GP-Bayesian.")

# ---- Tabla 2b: HPO tiempo de búsqueda (s) por presupuesto y método ----
st = pd.read_csv(f"{STUDY}/summary_hpo_time.csv", index_col=0)
mst = [m for m in HPO_ORDER if m in st.columns]
rows = [[str(int(b))] + [f"{st.loc[b, m]:.0f}s" for m in mst] for b in st.index]
render("table2b_hpo_time.png", "HPO methods: total search time (s) by budget",
       ["# configs"] + mst, rows, [(list(range(1, len(mst) + 1)), False)],
       "Green = fastest, red = slowest. Same number of configs, wildly different wall-clock: the "
       "multi-fidelity methods (hyperband, bohb) match the others' AP in ~15x less time by killing "
       "bad configs after a few trees; GP is ~2.5x faster than the grid.")

# ---- Tabla 3: iso-tiempo ----
d = pd.read_csv(f"{STUDY}/summary_isofinal.csv").dropna(subset=["e1_at", "e2_at", "ens_e1_at"])
by = d.groupby("frac").mean(numeric_only=True).reset_index()
rows = [[pct(r.frac), f"{r.e1_ap:.3f}", f"{r.e1_at:.3f}", f"{r.ens_e1:.3f}", f"{r.ens_e1_at:.3f}",
         f"{r.e2_ap:.3f}", f"{r.e2_at:.3f}", f"{r.ens_e2:.3f}"] for _, r in by.iterrows()]
render("table3_isotime.png", "Equal-time: AP by fraction (t* = ens-E2's time)",
       ["frac", "E1", "E1@t*", "ens-E1", "ens-E1@t*", "E2", "E2@t*", "ens-E2"], rows,
       [(list(range(1, 8)), True)],
       "Green = best at that fraction. ens-E2 wins everywhere even at equal time.")

# ---- Tabla 3b: tiempos de las estrategias del Capítulo 4 (base y @t*) ----
dt = pd.read_csv(f"{STUDY}/summary_isofinal.csv").dropna(subset=["e2ens_t"])
bt = dt.groupby("frac").mean(numeric_only=True).reset_index()
rows = [[pct(r.frac), f"{r.e1_t:.0f}s", f"{r.ens_e1_t:.0f}s", f"{r.e2_t:.0f}s", f"{r.e2ens_t:.0f}s",
         f"{r.e1_at_t:.0f}s", f"{r.e2_at_t:.0f}s", f"{r.ens_e1_at_t:.0f}s"] for _, r in bt.iterrows()]
render("table3b_ens_time.png", "Chapter 4 — wall-clock time by fraction (s)",
       ["frac", "E1", "ens-E1", "E2", "ens-E2\n(=t*)", "E1@t*", "E2@t*", "ens-E1@t*"], rows,
       [(list(range(1, 5)), False)],
       "Base costs (green = cheapest, red = costliest): ens-E1 is the cheapest, ens-E2 the priciest "
       "(~5x). All three @t* strategies are now grown until they cross t*, so E1@t*, E2@t* and "
       "ens-E1@t* all spend ens-E2's full time — a genuinely fair equal-time comparison.")

# ---- Tabla 4: feature engineering ----
fe = json.load(open(f"{STUDY}/summary_feateng.json"))
rows = [[f"{r['level']*100:.0f}%", f"{r['ap']:.3f}", f"{r['dAP']:+.4f}", f"{r['time']:.0f}s"] for r in fe["by_level"]]
render("table4_feateng.png", "Speculative feature engineering: AP vs % random features",
       ["% invented", "AP", "ΔAP vs 0%", "time"], rows, [([1], True)],
       f"At 100% added features: ΔAP = {fe['dAP_100_mean']:+.4f}, p = {fe['dAP_100_p']:.2f}. AP barely moves.")
# ---- Tabla 2c: HPO E1 vs E2 (AP y tiempo por método) ----
ee = json.load(open(f"{STUDY}/summary_hpo_e1e2.json"))
mm = ["grid", "random", "tpe", "cmaes", "gp", "hyperband", "bohb"]
rows = [[m, f"{ee[m]['e1_ap']:.3f}", f"{ee[m]['e1_t']:.0f}s",
         f"{ee[m]['e2_ap']:.3f}", f"{ee[m]['e2_t']:.0f}s"] for m in mm]
render("table2c_e1_vs_e2.png", "HPO: E1 (held-out val) vs E2 (CV/OOF) — AP & search time @160 configs",
       ["method", "AP (E1)", "time (E1)", "AP (E2)", "time (E2)"], rows,
       [([1], True), ([2], False), ([3], True), ([4], False)],
       "Green = best AP / fastest, per column. The time flips: under E1 the multi-fidelity methods "
       "(hyperband, bohb) are ~15x FASTER than the grid; under honest CV selection (E2) they're ~2x "
       "SLOWER — their speedup only exists when pruning rides a single held-out val set.")

# ---- Tabla 6: reducción por correlación de Spearman ----
fr = json.load(open(f"{STUDY}/summary_featreduce.json"))["by_thr"]
base = [x for x in fr if x["thr"] > 1][0]
thrs = sorted([x for x in fr if x["thr"] <= 1], key=lambda x: -x["thr"])
def _rr(x, lab): return [lab, f"{x['n_removed']:.1f}", f"{x['pct_removed']:.0f}%",
                         f"{x['ap']:.3f}", f"{x['dAP']:+.4f}", f"{x['time']:.0f}s"]
rows = [_rr(base, "baseline (keep all)")] + [_rr(x, f"|ρ|≥{x['thr']:.2f}") for x in thrs]
render("table6_featreduce.png", "Removing correlated features (|Spearman|): AP & time by threshold",
       ["threshold", "removed (#)", "removed (%)", "AP", "ΔAP vs base", "time"], rows,
       [([3], True), ([5], False)],
       "Green = higher AP / faster. Removing near-duplicates (|ρ|≥0.95–0.99, ~9–14% of features) is "
       "harmless and a touch faster; aggressive pruning (|ρ|≥0.75–0.85, ~40%) significantly HURTS AP "
       "(−0.04 to −0.06). E2 + LightGBM, test AP, 8 datasets × 2 seeds.")
print("tablas-imagen generadas")
