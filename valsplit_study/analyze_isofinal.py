"""analyze_isofinal.py — Apartado 8: a IGUALDAD DE TIEMPO. Tabla estilo apartado 7
con E1, ens-E1, E2, ens-E2 (naturales) + sus versiones @t* (haciendo más configs /
miembros hasta consumir el tiempo de ens-E2). Une:
  - summary_isotime.csv : E1, ens-E1, E2, ens-E2 naturales + tiempos (apartado 7)
  - results/isoE1       : ens-E1@t* (más miembros)
  - results/isogrid     : E1@t*, E2@t* (más configuraciones)
Salida: summary_isofinal.csv/.json + fig_isofinal.png
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
DS8 = ["jm1", "online_shoppers_intention", "Bank_marketing_data_set_UCI",
       "Pulsar-Dataset-HTRU2", "letter", "satimage", "mammography", "rl"]


def loadjson(tag):
    return pd.DataFrame([r for f in glob.glob(f"{STUDY}/results/{tag}__*.json") for r in json.load(open(f))])


def main():
    nat = pd.read_csv(f"{STUDY}/summary_isotime.csv") if __import__("os").path.exists(f"{STUDY}/summary_isotime.csv") else pd.DataFrame()
    isoE1 = loadjson("isoE1"); isog = loadjson("isogrid")
    if nat.empty or isoE1.empty:
        print("faltan datos (summary_isotime / isoE1)"); return
    nat = nat[nat.dataset.isin(DS8)]
    key = ["dataset", "frac", "seed"]
    # naturales (apartado 7): e1_ap, ens_at_e1 (=ens-E1 natural), e2_ap, e2ens_ap + tiempos
    n = nat[key + ["e1_ap", "ens_at_e1", "e2_ap", "e2ens_ap",
                   "e1_t", "ens_full_t", "e2_t", "e2ens_t"]].copy()
    n = n.rename(columns={"ens_at_e1": "ens_e1", "e2ens_ap": "ens_e2",
                          "ens_full_t": "ens_e1_t"})
    m = n.merge(isoE1[key + ["ap_at_tstar", "time_used"]].rename(
        columns={"ap_at_tstar": "ens_e1_at", "time_used": "ens_e1_at_t"}), on=key, how="left")
    if not isog.empty:
        m = m.merge(isog[key + ["e1_tstar_ap", "e1_tstar_time", "e2_tstar_ap", "e2_tstar_time"]].rename(
            columns={"e1_tstar_ap": "e1_at", "e1_tstar_time": "e1_at_t",
                     "e2_tstar_ap": "e2_at", "e2_tstar_time": "e2_at_t"}), on=key, how="left")
    for c in ["e1_at", "e1_at_t", "e2_at", "e2_at_t"]:
        if c not in m: m[c] = np.nan
    m.to_csv(f"{STUDY}/summary_isofinal.csv", index=False)

    def sig(a, b):
        d = (m[a] - m[b]).dropna().values
        if len(d) < 3: return {"mean": float("nan"), "p": None, "winrate": float("nan"), "n": int(len(d))}
        return {"mean": float(d.mean()), "p": float(stats.ttest_1samp(d, 0).pvalue),
                "winrate": float((d > 0).mean()), "n": int(len(d))}

    # PANEL BALANCEADO: solo celdas con TODAS las columnas @t* presentes (si no, las
    # medias promediarían sobre datasets distintos y engañarían mientras isogrid corre)
    atcols = ["e1_at", "e2_at", "ens_e1_at"]
    mb = m.dropna(subset=[c for c in atcols if c in m])
    out = {"n": int(len(mb)), "n_total": int(len(m)),
           "ens_e2_vs_ens_e1_at": sig("ens_e2", "ens_e1_at"),
           "ens_e2_vs_e1_at": sig("ens_e2", "e1_at"),
           "ens_e2_vs_e2_at": sig("ens_e2", "e2_at"),
           "mean_ap": {k: float(mb[k].mean()) for k in
                       ["e1_ap", "e1_at", "ens_e1", "ens_e1_at", "e2_ap", "e2_at", "ens_e2"]},
           "mean_t": {k: float(mb[k].mean()) for k in
                      ["e1_t", "e1_at_t", "ens_e1_t", "ens_e1_at_t", "e2_t", "e2_at_t", "e2ens_t"]}}
    json.dump(out, open(f"{STUDY}/summary_isofinal.json", "w"), indent=1)

    by = mb.groupby("frac").agg(**{k: (k, "mean") for k in
        ["e1_ap", "e1_at", "ens_e1", "ens_e1_at", "e2_ap", "e2_at", "ens_e2", "e2ens_t"]}).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(by.frac, by.ens_e2, "s-", color="#c0392b", lw=2, label=L("ens-E2 (referencia t*)", "ens-E2 (t* reference)"))
    ax[0].plot(by.frac, by.ens_e1_at, "v--", color="#e08e83", label="ens-E1 @t*")
    ax[0].plot(by.frac, by.e1_at, "o--", color="#1f77b4", label="E1 @t*")
    ax[0].plot(by.frac, by.e2_at, "^--", color="#5ba3d0", label="E2 @t*")
    ax[0].set_xlabel(L("fracción dev", "dev fraction")); ax[0].set_ylabel(L("AP (test) a igualdad de tiempo", "AP (test) at equal time"))
    ax[0].set_title(L("A igualdad de tiempo (t* = tiempo de ens-E2)", "At equal time (t* = ens-E2's time)"))
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    m["best_at"] = m[["e1_at", "e2_at", "ens_e1_at"]].max(axis=1)
    ax[1].scatter(m.best_at, m.ens_e2, s=16, alpha=.6)
    lim = [np.nanmin([m.best_at.min(), m.ens_e2.min()]), np.nanmax([m.best_at.max(), m.ens_e2.max()])]
    ax[1].plot(lim, lim, "k--", lw=.8)
    ax[1].set_xlabel(L("mejor alternativa @t* (E1/E2/ens-E1)", "best alternative @t* (E1/E2/ens-E1)")); ax[1].set_ylabel("ens-E2")
    ax[1].set_title(L("ens-E2 vs mejor alternativa @t* (por celda)", "ens-E2 vs best alternative @t* (per cell)")); ax[1].grid(alpha=.3)
    plt.suptitle(L("Apartado 8 — a igualdad de tiempo, ¿qué gana?",
                   "Section 8 — at equal time, what wins?"), fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{STUDY}/fig_isofinal{SUF}.png", dpi=130); plt.close()
    print(by.round(3).to_string(index=False)); print(json.dumps(out, indent=1)[:700])
    print("fig -> fig_isofinal.png")


if __name__ == "__main__":
    main()
