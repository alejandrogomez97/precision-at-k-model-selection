"""analyze_isotime.py — Fase C reformulada como comparación ISO-TIEMPO.

Pregunta: dado el MISMO tiempo de búsqueda que gasta E1 (o E2) en su grid de 18
configs, ¿un ensemble de familias alcanza más AP en test?

Para cada celda (dataset, fracción, semilla):
  - E1/E2: tiempo de búsqueda (search_time) y AP en test (de gridf).
  - Ensemble: de su curva anytime (AP vs tiempo acumulado entrenando el pool), se
    toma el mejor AP alcanzable DENTRO del presupuesto de tiempo de E1 y de E2.
Se comparan a igualdad de tiempo. Genera fig_C_isotime.png y summary_isotime.csv.
"""
import glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
import os as _os
EN = _os.environ.get("FIGLANG") == "en"
def L(es, en): return en if EN else es
SUF = "_en" if EN else ""


def load(tag):
    return pd.DataFrame([r for f in glob.glob(f"{STUDY}/results/{tag}__*.json")
                         for r in json.load(open(f)) if "error" not in r])


def ap_within(anytime, T):
    """Mejor AP del ensemble con tiempo acumulado <= T (None si ni un modelo entra)."""
    best = None
    for a in anytime:
        if a["cum_time"] <= T:
            best = a["ens_test"]
    return best


def main():
    fam = load("familiesf"); g = load("gridf"); e2f = load("ens_e2f")
    if fam.empty or g.empty:
        print("faltan datos"); return
    gi = g.set_index(["dataset", "frac", "seed", "strat"])
    # ensemble estilo E2 por celda
    e2map = {}
    if not e2f.empty:
        for _, r in e2f.iterrows():
            if r.get("e2_ens_test") is not None and r["e2_ens_test"] == r["e2_ens_test"]:
                e2map[(r["dataset"], r["frac"], r["seed"])] = (float(r["e2_ens_test"]),
                                                               float(r["e2_ens_time"]))
    recs = []
    for _, row in fam.iterrows():
        if not isinstance(row.get("anytime"), list) or not row["anytime"]:
            continue
        k = (row["dataset"], row["frac"], row["seed"])
        try:
            e1_t = float(gi.loc[(*k, "E1"), "search_time"]); e1_ap = float(gi.loc[(*k, "E1"), "test_ap_by_ap"])
            e2_t = float(gi.loc[(*k, "E2"), "search_time"]); e2_ap = float(gi.loc[(*k, "E2"), "test_ap_by_ap"])
        except KeyError:
            continue
        at = row["anytime"]
        ens_e1 = ap_within(at, e1_t); ens_e2 = ap_within(at, e2_t)
        e2ens_ap, e2ens_t = e2map.get(k, (np.nan, np.nan))
        recs.append({
            "dataset": row["dataset"], "frac": row["frac"], "seed": row["seed"],
            "e1_ap": e1_ap, "e2_ap": e2_ap, "e1_t": e1_t, "e2_t": e2_t,
            "ens_full": at[-1]["ens_test"], "ens_full_t": at[-1]["cum_time"],
            "ens_at_e1": ens_e1, "ens_at_e2": ens_e2,
            "e2ens_ap": e2ens_ap, "e2ens_t": e2ens_t,
            # ensemble E1 (blend en val) vs E1 a igual tiempo
            "beat_e1_isot": (ens_e1 is not None and ens_e1 > e1_ap),
            # ensemble E2 (blend OOF + refit dev) vs E2 (en test; y si es más rápido)
            "beat_e2ens": (e2ens_ap == e2ens_ap and e2ens_ap > e2_ap),
            "e2ens_faster": (e2ens_t == e2ens_t and e2ens_t <= e2_t),
            # referencia antigua: ensemble E1 vs E2 a tiempo de E2
            "beat_e2_isot": (ens_e2 is not None and ens_e2 > e2_ap),
        })
    d = pd.DataFrame(recs)
    if d.empty:
        print("sin celdas comunes"); return
    d.to_csv(f"{STUDY}/summary_isotime.csv", index=False)

    by = d.groupby("frac").agg(
        e1_ap=("e1_ap", "mean"), e2_ap=("e2_ap", "mean"),
        ens_e1=("ens_at_e1", "mean"), ens_e2=("e2ens_ap", "mean"),
        win_e1=("beat_e1_isot", "mean"), win_e2ens=("beat_e2ens", "mean"),
        e1_t=("e1_t", "mean"), e2_t=("e2_t", "mean"),
        ens_e1_t=("ens_full_t", "mean"), ens_e2_t=("e2ens_t", "mean"),
        n=("e1_ap", "size")).reset_index()

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    # panel 1: AP — E1 y su ensemble-E1 (a tiempo de E1); E2 y su ensemble-E2
    ax[0].plot(by["frac"], by["e1_ap"], "o-", color="#1f77b4", label="E1 (grid LGBM)")
    ax[0].plot(by["frac"], by["ens_e1"], "^--", color="#5ba3d0", label=L("ensemble-E1 (blend en val)", "ensemble-E1 (blend on val)"))
    ax[0].plot(by["frac"], by["e2_ap"], "s-", color="#c0392b", label="E2 (grid LGBM)")
    ax[0].plot(by["frac"], by["ens_e2"], "v--", color="#e08e83", label=L("ensemble-E2 (blend OOF+refit)", "ensemble-E2 (blend OOF+refit)"))
    ax[0].set_xlabel(L("fracción dev", "dev fraction")); ax[0].set_ylabel("AP (test)")
    ax[0].set_title(L("AP: cada ensemble vs su estrategia", "AP: each ensemble vs its strategy")); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].axhline(.5, color="k", lw=.8, ls="--")
    ax[1].plot(by["frac"], by["win_e1"], "o-", color="#1f77b4", label=L("ens-E1 > E1 (iso-tiempo)", "ens-E1 > E1 (equal time)"))
    ax[1].plot(by["frac"], by["win_e2ens"], "s-", color="#c0392b", label="ens-E2 > E2 (test)")
    ax[1].set_ylim(0, 1); ax[1].set_xlabel(L("fracción dev", "dev fraction"))
    ax[1].set_title(L("¿El ensemble gana a su estrategia?", "Does the ensemble beat its strategy?")); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    ax[2].plot(by["frac"], by["e1_t"], "o-", color="#1f77b4", label=L("tiempo E1", "time E1"))
    ax[2].plot(by["frac"], by["ens_e1_t"], "^-", color="#5ba3d0", label=L("tiempo ens-E1", "time ens-E1"))
    ax[2].plot(by["frac"], by["e2_t"], "s-", color="#c0392b", label=L("tiempo E2", "time E2"))
    ax[2].plot(by["frac"], by["ens_e2_t"], "v-", color="#e08e83", label=L("tiempo ens-E2", "time ens-E2"))
    ax[2].set_yscale("log"); ax[2].set_xlabel(L("fracción dev", "dev fraction")); ax[2].set_ylabel("s")
    ax[2].set_title(L("Tiempos", "Times")); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    plt.suptitle(L("Fase C — ensemble estilo E1 vs E1, y estilo E2 vs E2",
                   "Phase C — E1-style ensemble vs E1, and E2-style vs E2"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_C_isotime{SUF}.png", dpi=130); plt.close()

    import json as _j
    from scipy import stats

    def sig(delta):
        delta = delta.dropna().values
        if len(delta) < 3:
            return {"mean": float("nan"), "p": None, "winrate": float("nan"), "n": int(len(delta))}
        return {"mean": float(delta.mean()), "p": float(stats.ttest_1samp(delta, 0).pvalue),
                "winrate": float((delta > 0).mean()), "n": int(len(delta))}

    e1d = pd.Series(d["ens_at_e1"]) - d["e1_ap"]
    e2d = pd.Series(d["e2ens_ap"]) - d["e2_ap"]
    out = {"ens_e1_beats_e1_isotime": float(d.beat_e1_isot.mean()),
           "ens_e2_beats_e2": float(d.beat_e2ens.mean()) if "beat_e2ens" in d else None,
           "ens_e2_faster": float(d.e2ens_faster.mean()) if "e2ens_faster" in d else None,
           "n": int(len(d)), "n_e2": int(d["e2ens_ap"].notna().sum()),
           "sig_ens_e1": sig(e1d), "sig_ens_e2": sig(e2d)}
    _j.dump(out, open(f"{STUDY}/summary_isotime.json", "w"), indent=1)
    print(by.round(3).to_string(index=False)); print(out); print("fig -> fig_C_isotime.png")


if __name__ == "__main__":
    main()
