"""
core.py — Núcleo del estudio "¿un solo conjunto de validación o dos?".

Compara dos estrategias de selección de modelos sobre datasets binarios
desbalanceados, barriendo la cantidad de datos de desarrollo (dev) con un
conjunto de TEST FIJO COMÚN a ambas estrategias en cada punto:

  E1 (val separado):  dev -> train + val.  El nº de árboles se fija por CROSS
      VALIDATION dentro de train (early stopping). El MEJOR modelo se elige con
      el conjunto val independiente (no usado para el early stopping). El modelo
      final se REENTRENA en train+val (= dev) y se evalúa en test.

  E2 (solo CV):       dev -> CV. Los folds (OOF) deciden A LA VEZ el nº de
      árboles (early stopping) y el mejor modelo. El modelo final se reentrena
      en dev y se evalúa en test.

Ambas estrategias comparten: mismo pool de desarrollo dev, mismo test, mismo
grid de candidatos. La ÚNICA diferencia es el mecanismo de selección, por lo que
el experimento aísla el efecto de la maldición del ganador vs el ruido de
selección en función de la cantidad de datos.

Preprocesado ajustado por-fold (sin fugas). Métrica primaria: Average Precision
(PR-AUC); se registra también logloss. Se registran TIEMPOS de todo.

Reutiliza la infraestructura de datos de ../benchmark/common.py.
"""
import os, sys, json, time, warnings, pickle
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BENCH = "/home/agomez/proyectos/precision-at-k-study/benchmark"
sys.path.insert(0, BENCH)
import common as C  # noqa: E402

from sklearn.model_selection import StratifiedKFold, train_test_split       # noqa: E402
from sklearn.metrics import average_precision_score, log_loss              # noqa: E402
import lightgbm as lgb                                                       # noqa: E402

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
RES = f"{STUDY}/results"
os.makedirs(RES, exist_ok=True)

# --------------------------------------------------------------------------- #
#  Configuración del experimento
# --------------------------------------------------------------------------- #
CV_K = 4
ES_ROUNDS = 50
MAX_TREES = 3000
LR = 0.05
TEST_FRAC = 0.30      # test = 30% del dataset completo...
TEST_CAP = 6000       # ...pero como mucho 6000 filas (por coste)
VAL_FRAC = 0.25       # E1: val = 25% de dev

# Grid de LightGBM (6 combinaciones) x técnicas de desbalanceo (3) = 18 candidatos
LGBM_GRID = [
    {"num_leaves": nl, "reg_lambda": rl}
    for nl in (15, 31, 63)
    for rl in (0.0, 1.0)
]
IMB_TECHNIQUES = ("none", "class_weight", "smote")

# Barrido por FRACCIÓN del pool de desarrollo de cada dataset (no por nº absoluto):
# así todos los datasets están presentes en todos los puntos y la curva agregada no
# sufre el sesgo de composición (solo los grandes llegan a tamaños absolutos altos).
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
DEV_CAP = 20000     # se capa el pool de desarrollo a 20000 filas por coste/RAM
SIZES = [300, 600, 1200, 2500, 5000, 10000, 20000]   # (legado; ya no se usa por defecto)
SEEDS = [0, 1, 2]


def make_candidates():
    cands = []
    for g in LGBM_GRID:
        for imb in IMB_TECHNIQUES:
            cands.append({**g, "imb": imb})
    return cands


# --------------------------------------------------------------------------- #
#  Preprocesado (ajustado solo con el train dado) + resampling
# --------------------------------------------------------------------------- #
def _fit_transform(prep, X):
    return prep.transform(X)


def _resample(Xtr, ytr, imb, seed):
    """Devuelve (Xr, yr) tras aplicar la técnica de desbalanceo al fold-train."""
    if imb != "smote":
        return Xtr, ytr
    from imblearn.over_sampling import SMOTE
    n_min = int(ytr.sum())
    n_min = min(n_min, int((ytr == 0).sum()))
    if n_min < 2:
        return Xtr, ytr
    k = min(5, n_min - 1)
    try:
        sm = SMOTE(random_state=seed, k_neighbors=k)
        return sm.fit_resample(Xtr, ytr)
    except Exception:
        return Xtr, ytr


def _lgbm_params(cand, ytr_resampled=None, imb="none"):
    p = dict(
        n_estimators=MAX_TREES, learning_rate=cand.get("learning_rate", LR),
        num_leaves=cand["num_leaves"], reg_lambda=cand["reg_lambda"],
        reg_alpha=cand.get("reg_alpha", 0.0),
        min_child_samples=cand.get("min_child_samples", 20),
        subsample=cand.get("subsample", 0.9), subsample_freq=1,
        colsample_bytree=cand.get("colsample_bytree", 0.8),
        n_jobs=1, verbose=-1, random_state=42,
        metric="average_precision",
    )
    if imb == "class_weight":
        p["class_weight"] = "balanced"
    return p


def fit_fold(Xtr_raw, ytr, Xev_raw, yev, meta, cand, seed, early_stop=True,
             n_trees=None, X_extra_raw=None):
    """
    Ajusta preprocesador SOLO en Xtr_raw, resamplea si procede, entrena LGBM.
    - early_stop=True: usa (Xev,yev) como eval set y devuelve best_iteration.
    - early_stop=False: entrena n_trees árboles fijos (Xev solo para predecir).
    - X_extra_raw: si se da, también predice sobre ese conjunto (para el modelo
      entrenado solo con Xtr) -> 'proba_extra'. Sirve para la variante SIN
      reentrenamiento: usar directamente el modelo del fold/train sobre test.
    Devuelve dict(proba_ev, best_iter, fit_time[, proba_extra]).
    """
    prep = C.make_preprocessor(meta)
    prep.fit(Xtr_raw, ytr)
    Xtr = prep.transform(Xtr_raw)
    Xev = prep.transform(Xev_raw)
    Xtr, ytr2 = _resample(Xtr, np.asarray(ytr), cand["imb"], seed)
    params = _lgbm_params(cand, imb=cand["imb"])
    t0 = time.perf_counter()
    clf = lgb.LGBMClassifier(**params)
    if early_stop:
        clf.set_params(n_estimators=MAX_TREES)
        clf.fit(Xtr, ytr2, eval_set=[(Xev, yev)], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(ES_ROUNDS, verbose=False),
                           lgb.log_evaluation(0)])
        best_iter = clf.best_iteration_ or MAX_TREES
    else:
        clf.set_params(n_estimators=int(max(1, n_trees)))
        clf.fit(Xtr, ytr2)
        best_iter = int(max(1, n_trees))
    dt = time.perf_counter() - t0
    out = {"proba": clf.predict_proba(Xev)[:, 1], "best_iter": int(best_iter),
           "fit_time": dt}
    if X_extra_raw is not None:
        out["proba_extra"] = clf.predict_proba(prep.transform(X_extra_raw))[:, 1]
    return out


def _score(y, p):
    y = np.asarray(y).astype(int)
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    ap = average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan
    try:
        ll = log_loss(y, p, labels=[0, 1])
    except Exception:
        ll = np.nan
    return float(ap), float(ll)


# --------------------------------------------------------------------------- #
#  Estrategias
# --------------------------------------------------------------------------- #
def cv_best_iter_and_oof(X_raw, y, meta, cand, seed, want_oof=False,
                         X_test_raw=None):
    """CV estratificada. Devuelve (mean_best_iter, cv_time, oof_proba|None,
    test_bag|None). test_bag = media de las predicciones de los k modelos de fold
    sobre X_test (ensemble bagged de la CV = variante E2 SIN reentrenar)."""
    skf = StratifiedKFold(n_splits=CV_K, shuffle=True, random_state=seed)
    y = np.asarray(y)
    iters, tsum = [], 0.0
    oof = np.full(len(y), np.nan) if want_oof else None
    test_bag = None
    X_raw = X_raw.reset_index(drop=True)
    for tr, ev in skf.split(X_raw, y):
        r = fit_fold(X_raw.iloc[tr], y[tr], X_raw.iloc[ev], y[ev],
                     meta, cand, seed, early_stop=True, X_extra_raw=X_test_raw)
        iters.append(r["best_iter"]); tsum += r["fit_time"]
        if want_oof:
            oof[ev] = r["proba"]
        if X_test_raw is not None:
            test_bag = r["proba_extra"] if test_bag is None else test_bag + r["proba_extra"]
    if test_bag is not None:
        test_bag = test_bag / CV_K
    return int(np.mean(iters)), tsum, oof, test_bag


def eval_E1(X_dev, y_dev, X_test, y_test, meta, cands, seed):
    """Estrategia 1: val separado. Devuelve dict con resultados por métrica."""
    y_dev = np.asarray(y_dev)
    Xtr, Xval, ytr, yval = train_test_split(
        X_dev, y_dev, test_size=VAL_FRAC, stratify=y_dev, random_state=seed)
    Xtr = Xtr.reset_index(drop=True); Xval = Xval.reset_index(drop=True)
    search_time = 0.0
    rows = []
    for cand in cands:
        # 1) nº de árboles por CV dentro de train (early stopping)
        n_trees, cv_t, _, _ = cv_best_iter_and_oof(Xtr, ytr, meta, cand, seed)
        # 2) selección: entrenar en train con n_trees fijos, medir en val.
        #    De paso predice test con ESE modelo (train-only) = variante sin-retrain
        r = fit_fold(Xtr, ytr, Xval, yval, meta, cand, seed,
                     early_stop=False, n_trees=n_trees, X_extra_raw=X_test)
        search_time += cv_t + r["fit_time"]
        ap, ll = _score(yval, r["proba"])
        ap_nr, ll_nr = _score(y_test, r["proba_extra"])
        rows.append({"cand": cand, "n_trees": n_trees, "val_ap": ap, "val_ll": ll,
                     "test_ap_nr": ap_nr, "test_ll_nr": ll_nr})
    return _finalize(rows, X_dev, y_dev, X_test, y_test, meta, seed,
                     sel_key_ap="val_ap", sel_key_ll="val_ll",
                     search_time=search_time, strat="E1")


def eval_E2(X_dev, y_dev, X_test, y_test, meta, cands, seed):
    """Estrategia 2: solo CV (OOF decide árboles y modelo)."""
    y_dev = np.asarray(y_dev)
    X_dev = X_dev.reset_index(drop=True)
    search_time = 0.0
    rows = []
    for cand in cands:
        # OOF para selección + ensemble bagged de los k folds sobre test (sin-retrain)
        n_trees, cv_t, oof, test_bag = cv_best_iter_and_oof(
            X_dev, y_dev, meta, cand, seed, want_oof=True, X_test_raw=X_test)
        search_time += cv_t
        m = ~np.isnan(oof)
        ap, ll = _score(y_dev[m], oof[m])
        ap_nr, ll_nr = _score(y_test, test_bag)
        rows.append({"cand": cand, "n_trees": n_trees, "val_ap": ap, "val_ll": ll,
                     "test_ap_nr": ap_nr, "test_ll_nr": ll_nr})
    return _finalize(rows, X_dev, y_dev, X_test, y_test, meta, seed,
                     sel_key_ap="val_ap", sel_key_ll="val_ll",
                     search_time=search_time, strat="E2")


def _finalize(rows, X_dev, y_dev, X_test, y_test, meta, seed,
              sel_key_ap, sel_key_ll, search_time, strat):
    """Selecciona ganador por AP y por logloss, reentrena en dev, evalúa en test."""
    out = {"strat": strat, "search_time": search_time, "n_cands": len(rows)}
    for metric, key, better in (("ap", sel_key_ap, "max"), ("ll", sel_key_ll, "min")):
        valid = [r for r in rows if not np.isnan(r[key])]
        if not valid:
            for f in (f"test_ap_by_{metric}", f"test_ll_by_{metric}",
                      f"test_ap_nr_by_{metric}", f"test_ll_nr_by_{metric}"):
                out[f] = np.nan
            continue
        best = max(valid, key=lambda r: r[key]) if better == "max" \
            else min(valid, key=lambda r: r[key])
        # variante CON reentrenamiento: refit del ganador en todo dev -> test
        t0 = time.perf_counter()
        r = fit_fold(X_dev, y_dev, X_test, y_test, meta, best["cand"], seed,
                     early_stop=False, n_trees=best["n_trees"])
        ft = time.perf_counter() - t0
        ap, ll = _score(y_test, r["proba"])
        out[f"test_ap_by_{metric}"] = ap
        out[f"test_ll_by_{metric}"] = ll
        out[f"final_fit_time_by_{metric}"] = ft
        # variante SIN reentrenamiento: test del modelo ya entrenado (train-only / bagged)
        out[f"test_ap_nr_by_{metric}"] = best["test_ap_nr"]
        out[f"test_ll_nr_by_{metric}"] = best["test_ll_nr"]
        out[f"winner_by_{metric}"] = {**best["cand"], "n_trees": best["n_trees"]}
    return out


# --------------------------------------------------------------------------- #
#  Un dataset completo (barrido de tamaños x semillas)
# --------------------------------------------------------------------------- #
def get_dev_test(X, y, seed):
    """Separa un test fijo (30% capado a 6000) y devuelve (X_dev_full, y_dev_full,
    X_test, y_test)."""
    y = np.asarray(y)
    n = len(y)
    test_size = min(TEST_FRAC, TEST_CAP / n) if n * TEST_FRAC > TEST_CAP else TEST_FRAC
    Xdev, Xte, ydev, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=1000 + seed)
    return Xdev.reset_index(drop=True), np.asarray(ydev), \
        Xte.reset_index(drop=True), np.asarray(yte)


def subsample(X_dev, y_dev, size, seed):
    """Submuestra estratificado a `size` filas del pool de desarrollo."""
    if size >= len(y_dev):
        return X_dev, y_dev
    Xs, _, ys, _ = train_test_split(
        X_dev, y_dev, train_size=size, stratify=y_dev, random_state=2000 + seed)
    return Xs.reset_index(drop=True), np.asarray(ys)


def run_dataset(name, fracs=None, seeds=None, out_dir=RES, tag="gridf"):
    """Ejecuta E1 y E2 sobre un dataset barriendo FRACCIONES del pool de desarrollo
    (cada dataset presente en todos los puntos). Resumible por (frac, seed, strat)."""
    fracs = fracs or FRACS
    seeds = seeds or SEEDS
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{out_dir}/{tag}__{safe}.json"
    done = {}
    if os.path.exists(outp):
        done = {(r["frac"], r["seed"], r["strat"]): r for r in json.load(open(outp))}

    X, y, meta = C.load_dataset(name)
    cands = make_candidates()
    results = list(done.values())
    prevalence = float(np.mean(y))
    n_pos = int(np.sum(y))

    for seed in seeds:
        Xdev_full, ydev_full, Xte, yte = get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), DEV_CAP)   # pool de referencia (capado)
        for frac in fracs:
            size = int(round(frac * dev_base))
            Xdev, ydev = subsample(Xdev_full, ydev_full, size, seed)
            # nº mínimo de positivos para que CV+SMOTE tenga sentido
            if ydev.sum() < CV_K or (ydev == 0).sum() < CV_K:
                continue
            for strat, fn in (("E1", eval_E1), ("E2", eval_E2)):
                if (frac, seed, strat) in done:
                    continue
                t0 = time.perf_counter()
                try:
                    r = fn(Xdev, ydev, Xte, yte, meta, cands, seed)
                except Exception as e:
                    r = {"strat": strat, "error": f"{type(e).__name__}: {e}"}
                r.update({
                    "dataset": name, "frac": frac, "size": int(len(ydev)),
                    "dev_base": int(dev_base), "seed": seed,
                    "n_test": int(len(yte)), "prevalence": prevalence,
                    "n_pos_full": n_pos, "imb_ratio": float(meta["imbalance_ratio"]),
                    "n_features": int(meta["n_features"]),
                    "wall_time": time.perf_counter() - t0, "tag": tag,
                })
                results.append(r)
                done[(frac, seed, strat)] = r
                json.dump(results, open(outp, "w"), indent=1)
                ap1 = r.get("test_ap_by_ap", float("nan"))
                print(f"  [{name}] frac={frac} (n={len(ydev)}) seed={seed} {strat} "
                      f"test_AP={ap1:.4f} search={r.get('search_time',0):.1f}s "
                      f"wall={r['wall_time']:.1f}s", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*", help="nombres de datasets (vacío=todos)")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--fracs", type=str, default="")
    ap.add_argument("--tag", type=str, default="gridf")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    fracs = [float(s) for s in args.fracs.split(",")] if args.fracs else None
    names = args.datasets or C.list_materialized()
    print(f"[core] {len(names)} datasets | seeds={seeds} | tag={args.tag}", flush=True)
    for i, nm in enumerate(names):
        t0 = time.time()
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, fracs=fracs, seeds=seeds, tag=args.tag)
        except Exception as e:
            print(f"  !! {nm} fallo: {type(e).__name__}: {e}", flush=True)
        print(f"  ({nm} en {time.time()-t0:.0f}s)", flush=True)
