"""isotime_ensE1.py — Apartado final (iso-tiempo). Deja crecer el ensemble estilo E1
(más miembros) hasta consumir el MISMO tiempo que gasta ens-E2, y mide su AP en ese
punto. Así se compara, a igualdad de tiempo, contra ens-E2 (y contra E1-grid, que ya
tenemos del barrido HPO con tiempos).

Para cada celda (dataset, fracción, semilla):
  - t_star = tiempo de ens-E2 (de results/ens_e2f).
  - Se entrenan miembros de un pool ampliado (variantes de las 10 familias) uno a uno
    sobre train, con blend greedy sobre val, registrando (tiempo acumulado, AP test).
  - Se reporta el AP del ensemble en el último punto con tiempo <= t_star.
Salida: results/isoE1__*.json
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
import families_study as F
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

DATASETS = ["jm1", "online_shoppers_intention", "Bank_marketing_data_set_UCI",
            "Pulsar-Dataset-HTRU2", "letter", "satimage", "mammography", "rl"]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [0, 1]


MAX_MEMBERS = 400   # tope de seguridad


def member_stream(seed, spw):
    """Genera miembros INDEFINIDAMENTE para poder gastar cualquier t* (equal-time justo):
    primero un lote diverso fijo, luego repite variantes sembradas con semillas crecientes
    hasta que quien consume decide parar (al cruzar t*)."""
    # --- lote fijo (una sola vez): lgbm grid, hgb, knn, lr, gnb ---
    for nl in (15, 31, 63, 127):
        for lr in (0.03, 0.05, 0.1):
            yield (f"lgbm{nl}_{lr}", lgb.LGBMClassifier(n_estimators=400, learning_rate=lr,
                   num_leaves=nl, class_weight="balanced", n_jobs=1, verbose=-1, random_state=seed))
    for mi in (300, 600):
        yield (f"hgb{mi}", HistGradientBoostingClassifier(max_iter=mi, learning_rate=0.05,
               early_stopping=True, random_state=seed))
    for k in (15, 25, 50):
        yield (f"knn{k}", KNeighborsClassifier(n_neighbors=k))
    for c in (0.1, 1.0, 10.0):
        yield (f"lr{c}", LogisticRegression(max_iter=1000, class_weight="balanced", C=c))
    yield ("gnb", GaussianNB())
    # --- luego, cíclicamente, variantes sembradas con semilla creciente cada ronda ---
    r = 0
    while True:
        # miembros algo más pesados (más árboles) para que cada uno cueste más → hacen falta
        # menos para llenar t*, con menos overhead de inicialización (mismo estilo E1: 1 entreno)
        s = seed + r * 1000 + 7
        yield (f"rf{s}", RandomForestClassifier(n_estimators=500, class_weight="balanced_subsample", n_jobs=1, random_state=s))
        yield (f"et{s}", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", n_jobs=1, random_state=s))
        yield (f"xgb{s}", XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=6, subsample=0.9,
               colsample_bytree=0.8, n_jobs=1, eval_metric="aucpr", scale_pos_weight=spw,
               tree_method="hist", verbosity=0, random_state=s))
        yield (f"cat{s}", CatBoostClassifier(iterations=600, learning_rate=0.05, depth=6,
               auto_class_weights="Balanced", verbose=0, random_seed=s, thread_count=1))
        yield (f"mlp{s}", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, early_stopping=True, random_state=s))
        yield (f"lgbmx{s}", lgb.LGBMClassifier(n_estimators=800, learning_rate=0.05, num_leaves=63,
               subsample=0.9, colsample_bytree=0.8, class_weight="balanced", n_jobs=1, verbose=-1, random_state=s))
        r += 1


def load_tstar():
    """t_star y AP de ens-E2 por celda (de results/ens_e2f)."""
    out = {}
    for f in __import__("glob").glob(f"{K.RES}/ens_e2f__*.json"):
        for r in json.load(open(f)):
            if r.get("e2_ens_test") is not None and r["e2_ens_test"] == r["e2_ens_test"]:
                out[(r["dataset"], r["frac"], r["seed"])] = (float(r["e2_ens_time"]), float(r["e2_ens_test"]))
    return out


def run_cell(Xdev, ydev, Xte, yte, meta, seed, t_star):
    ydev = np.asarray(ydev); yte = np.asarray(yte)
    Xtr, Xval, ytr, yval = train_test_split(Xdev, ydev, test_size=K.VAL_FRAC,
                                            stratify=ydev, random_state=seed)
    prep = K.C.make_preprocessor(meta); prep.fit(Xtr, ytr)
    Xtr_t = prep.transform(Xtr); Xval_t = prep.transform(Xval); Xte_t = prep.transform(Xte)
    spw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    val_p, test_p, cum = [], [], 0.0
    ap_at, k_at = np.nan, 0
    for nm, est in member_stream(seed, spw):
        t0 = time.perf_counter()
        try:
            est.fit(Xtr_t, ytr); vp = est.predict_proba(Xval_t)[:, 1]; tp = est.predict_proba(Xte_t)[:, 1]
        except Exception:
            continue
        cum += time.perf_counter() - t0
        val_p.append(vp); test_p.append(tp)
        if len(val_p) >= 2:
            _, ens_test, _ = F.greedy_ensemble(val_p, yval, test_p, yte)
        else:
            ens_test = K._score(yte, tp)[0]
        # añadir miembros hasta CRUZAR t* (>= t*); tope de seguridad por si t* es enorme
        ap_at, k_at = ens_test, len(val_p)
        if cum >= t_star or len(val_p) >= MAX_MEMBERS:
            break
    return {"ap_at_tstar": float(ap_at), "n_members": int(k_at), "time_used": float(cum)}


def main():
    tstar = load_tstar()
    datasets = sys.argv[1:] if len(sys.argv) > 1 else DATASETS
    for i, name in enumerate(datasets):
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
        outp = f"{K.RES}/isoE1__{safe}.json"
        done = {(r["frac"], r["seed"]) for r in json.load(open(outp))} if os.path.exists(outp) else set()
        results = json.load(open(outp)) if os.path.exists(outp) else []
        X, y, meta = K.C.load_dataset(name)
        print(f"[{i+1}/{len(DATASETS)}] {name}", flush=True)
        for seed in SEEDS:
            Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
            dev_base = min(len(ydev_full), K.DEV_CAP)
            for frac in FRACS:
                if (frac, seed) in done:
                    continue
                key = (name, frac, seed)
                if key not in tstar:
                    continue
                t_star, ap_e2 = tstar[key]
                Xdev, ydev = K.subsample(Xdev_full, ydev_full, int(round(frac * dev_base)), seed)
                if ydev.sum() < K.CV_K:
                    continue
                r = run_cell(Xdev, ydev, Xte, yte, meta, seed, t_star)
                r.update({"dataset": name, "frac": frac, "seed": seed,
                          "t_star": t_star, "ap_e2ens": ap_e2})
                results.append(r); json.dump(results, open(outp, "w"), indent=1)
                print(f"  f{frac} s{seed}: ens-E1@{t_star:.0f}s = {r['ap_at_tstar']:.4f} "
                      f"({r['n_members']} miembros) vs ens-E2 {ap_e2:.4f}", flush=True)


if __name__ == "__main__":
    main()
