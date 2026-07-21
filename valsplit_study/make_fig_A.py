"""Figura de 3 paneles que resume la Fase A (E1 vs E2)."""
import glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
import os
EN = os.environ.get("FIGLANG") == "en"
def L(es, en): return en if EN else es
SUF = "_en" if EN else ""
rows = [r for f in glob.glob(f"{STUDY}/results/grid__*.json")
        for r in json.load(open(f)) if "error" not in r]
df = pd.DataFrame(rows)
p = df.pivot_table(index=["dataset", "size", "seed", "imb_ratio"],
                   columns="strat", values="test_ap_by_ap").reset_index().dropna(subset=["E1", "E2"])
p["diff"] = p["E1"] - p["E2"]

C1, C2, CD = "#1f77b4", "#d62728", "#2ca02c"
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5))

# panel 1: AP medio vs tamaño (curvas casi solapadas -> equivalencia)
g = p.groupby("size").agg(E1=("E1", "mean"), E2=("E2", "mean")).reset_index()
ax[0].plot(g["size"], g["E1"], "o-", color=C1, label=L("E1 (val separado)", "E1 (held-out val)"))
ax[0].plot(g["size"], g["E2"], "s--", color=C2, label=L("E2 (solo CV)", "E2 (CV only)"))
ax[0].set_xscale("log"); ax[0].set_xlabel(L("tamaño del pool de desarrollo", "development pool size"))
ax[0].set_ylabel("Average Precision (test)")
ax[0].set_title(L("Rendimiento casi idéntico", "Almost identical performance")); ax[0].legend(); ax[0].grid(alpha=.3)

# panel 2: diferencia pareada vs tamaño
g2 = p.groupby("size").agg(md=("diff", "mean"),
                           se=("diff", lambda x: x.std()/np.sqrt(len(x))),
                           n=("diff", "size")).reset_index()
ax[1].axhline(0, color="k", lw=.8)
ax[1].errorbar(g2["size"], g2["md"], yerr=g2["se"], fmt="o-", color=CD, capsize=3)
ax[1].fill_between(g2["size"], g2["md"]-g2["se"], g2["md"]+g2["se"], color=CD, alpha=.15)
ax[1].set_xscale("log"); ax[1].set_xlabel(L("tamaño del pool de desarrollo", "development pool size"))
ax[1].set_ylabel("AP(E1) − AP(E2)")
ax[1].set_title(L("Ventaja (diminuta) para E2 en datos escasos", "Tiny edge for E2 when data is scarce")); ax[1].grid(alpha=.3)

# panel 3: diferencia pareada vs desbalanceo
p["irb"] = pd.cut(p["imb_ratio"], [0, 5, 10, 20, 1e9], labels=["<5", "5–10", "10–20", ">20"])
g3 = p.groupby("irb", observed=True).agg(md=("diff", "mean"),
        se=("diff", lambda x: x.std()/np.sqrt(len(x)))).reset_index()
ax[2].axhline(0, color="k", lw=.8)
ax[2].bar(range(len(g3)), g3["md"], yerr=g3["se"], color=CD, alpha=.7, capsize=3)
ax[2].set_xticks(range(len(g3))); ax[2].set_xticklabels(g3["irb"])
ax[2].set_xlabel(L("índice de desbalanceo (mayoritaria/minoritaria)", "imbalance ratio (majority/minority)"))
ax[2].set_ylabel("AP(E1) − AP(E2)")
ax[2].set_title(L("E2 gana más cuanto más desbalanceado", "E2 wins more the more imbalanced")); ax[2].grid(alpha=.3)

plt.suptitle(L("Fase A — ¿un solo conjunto de validación (E2) o dos (E1)?  ",
               "Phase A — one validation set (E2) or two (E1)?  ")
             + L(f"n={len(p)} celdas pareadas, 89 datasets", f"n={len(p)} paired cells, 89 datasets"), fontweight="bold")
plt.tight_layout(); plt.savefig(f"{STUDY}/fig_A_story{SUF}.png", dpi=135); plt.close()
print(f"fig_A_story{SUF}.png escrito")
