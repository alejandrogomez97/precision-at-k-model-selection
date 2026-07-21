"""analyze.py — Agrega resultados y genera figuras del estudio val-único vs val-doble.

Lee results/<tag>__*.json (por defecto tag=grid), construye un DataFrame largo y
produce:
  - fig_ap_vs_size.png      : AP medio en test vs tamaño de dev, E1 vs E2
  - fig_diff_vs_size.png    : diferencia pareada (E1-E2) y win-rate de E1 vs tamaño
  - summary.csv / summary.json
Métrica primaria: AP. Se normaliza el AP a "skill" = (AP-prev)/(1-prev) para
comparar datasets con prevalencias distintas.
"""
import os, sys, json, glob
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
RES = f"{STUDY}/results"


def load(tag="grid"):
    rows = []
    for fp in glob.glob(f"{RES}/{tag}__*.json"):
        for r in json.load(open(fp)):
            if "error" in r:
                continue
            rows.append(r)
    df = pd.DataFrame(rows)
    if len(df):
        df["skill_ap"] = (df["test_ap_by_ap"] - df["prevalence"]) / (1 - df["prevalence"])
    return df


def paired(df, metric="test_ap_by_ap"):
    """Tabla pareada E1 vs E2 por (dataset,size,seed)."""
    idx = ["dataset", "size", "seed"]
    p = df.pivot_table(index=idx, columns="strat", values=metric).reset_index()
    p = p.dropna(subset=["E1", "E2"])
    return p


def main(tag="grid"):
    df = load(tag)
    if not len(df):
        print("sin datos todavía"); return
    n_cells = df.groupby("strat").size().to_dict()
    print(f"[{tag}] filas={len(df)} por estrat={n_cells} datasets={df.dataset.nunique()}")

    # ---- agregación por tamaño (media sobre datasets y semillas) ----
    agg = df.groupby(["strat", "size"]).agg(
        test_ap=("test_ap_by_ap", "mean"),
        skill=("skill_ap", "mean"),
        test_ap_ll=("test_ap_by_ll", "mean"),
        search_time=("search_time", "mean"),
        n=("test_ap_by_ap", "size"),
    ).reset_index()
    agg.to_csv(f"{STUDY}/summary.csv", index=False)

    # ---- figura 1: AP y skill vs tamaño ----
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for strat, col in (("E1", "#1f77b4"), ("E2", "#d62728")):
        a = agg[agg.strat == strat].sort_values("size")
        axes[0].plot(a["size"], a["test_ap"], "o-", color=col, label=strat)
        axes[1].plot(a["size"], a["skill"], "o-", color=col, label=strat)
    for ax, ttl in ((axes[0], "AP medio en test"), (axes[1], "Skill = (AP-prev)/(1-prev)")):
        ax.set_xscale("log"); ax.set_xlabel("tamaño del pool de desarrollo (dev)")
        ax.set_title(ttl); ax.legend(); ax.grid(alpha=.3)
    plt.suptitle(f"E1 (val separado) vs E2 (solo CV) — {tag}", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_ap_vs_size.png", dpi=130); plt.close()

    # ---- figura 2: diferencia pareada y win-rate ----
    p = paired(df)
    p["diff"] = p["E1"] - p["E2"]
    g = p.groupby("size").agg(mean_diff=("diff", "mean"),
                              se=("diff", lambda x: x.std() / max(1, np.sqrt(len(x)))),
                              winrate=("diff", lambda x: (x > 0).mean()),
                              n=("diff", "size")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].axhline(0, color="k", lw=.8)
    axes[0].errorbar(g["size"], g["mean_diff"], yerr=g["se"], fmt="o-", color="#2ca02c")
    axes[0].set_title("Diferencia pareada de AP (E1 − E2)")
    axes[0].set_ylabel("AP(E1) − AP(E2)")
    axes[1].axhline(.5, color="k", lw=.8, ls="--")
    axes[1].plot(g["size"], g["winrate"], "o-", color="#9467bd")
    axes[1].set_title("Win-rate de E1 sobre E2")
    axes[1].set_ylabel("P(E1 > E2)"); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.set_xscale("log"); ax.set_xlabel("tamaño del pool de dev"); ax.grid(alpha=.3)
    plt.suptitle(f"¿Cuándo gana el val separado? — {tag}", fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_diff_vs_size.png", dpi=130); plt.close()

    g.to_json(f"{STUDY}/paired_by_size.json", orient="records", indent=1)
    print(agg.to_string(index=False))
    print("\n--- diferencia pareada por tamaño ---")
    print(g.to_string(index=False))
    print(f"\nfiguras -> {STUDY}/fig_ap_vs_size.png, fig_diff_vs_size.png")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "grid")
