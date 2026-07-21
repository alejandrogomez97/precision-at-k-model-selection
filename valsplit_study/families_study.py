"""families_study.py — Fase C. En vez de muchos LightGBM, entrena un POOL de
familias distintas y construye ENSEMBLES (selección greedy de Caruana sobre val).

Marco E1 (val separado, selección honesta): dev -> train/val, preprocesado en
train. Cada familia se entrena en train y se puntúa en val y test. Se registra el
tiempo de entrenamiento acumulado y, tras cada modelo, la CURVA ANYTIME:
  - mejor modelo individual hasta ahora (por AP en val) -> su AP en test
  - mejor ensemble greedy hasta ahora (pesos ajustados en val)  -> su AP en test

Objetivo: ver si el pool+ensemble alcanza (y consolida en test) un AP mejor que
la estrategia E1 con grid de LightGBM, y en cuánto tiempo.
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
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

FRACS_C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]  # 10 en 10, como el resto
SEEDS_C = [0, 1]


def base_models(seed, spw):
    """Pool ordenado aprox. de barato->caro (define la secuencia anytime)."""
    return [
        ("logreg", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)),
        ("gnb", GaussianNB()),
        ("knn", KNeighborsClassifier(n_neighbors=25)),
        ("lgbm", lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05,
                                    num_leaves=31, class_weight="balanced",
                                    n_jobs=1, verbose=-1)),
        ("hgb", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                               early_stopping=True, random_state=seed)),
        ("xgb", XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=6,
                              subsample=0.9, colsample_bytree=0.8, n_jobs=1,
                              eval_metric="aucpr", scale_pos_weight=spw,
                              tree_method="hist", verbosity=0)),
        ("extratrees", ExtraTreesClassifier(n_estimators=300, class_weight="balanced",
                                            n_jobs=1, random_state=seed)),
        ("rf", RandomForestClassifier(n_estimators=300,
                                      class_weight="balanced_subsample",
                                      n_jobs=1, random_state=seed)),
        ("catboost", CatBoostClassifier(iterations=400, learning_rate=0.05, depth=6,
                                        auto_class_weights="Balanced", verbose=0,
                                        random_seed=seed, thread_count=1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300,
                              early_stopping=True, random_state=seed)),
    ]


def greedy_ensemble(val_probas, yval, test_probas, yte, max_iter=50):
    """Selección greedy con reemplazo (Caruana) maximizando AP en val.
    Devuelve (val_ap, test_ap, weights)."""
    n = len(val_probas)
    counts = np.zeros(n)
    sum_val = np.zeros_like(val_probas[0])
    sum_test = np.zeros_like(test_probas[0])
    # init con el mejor individual
    aps = [K._score(yval, p)[0] for p in val_probas]
    b = int(np.nanargmax(aps))
    counts[b] += 1; sum_val += val_probas[b]; sum_test += test_probas[b]
    best_val = aps[b]
    for _ in range(max_iter):
        tot = counts.sum()
        cand_ap = [K._score(yval, (sum_val + val_probas[i]) / (tot + 1))[0]
                   for i in range(n)]
        j = int(np.nanargmax(cand_ap))
        if cand_ap[j] <= best_val + 1e-6:
            break
        best_val = cand_ap[j]
        counts[j] += 1; sum_val += val_probas[j]; sum_test += test_probas[j]
    w = counts / counts.sum()
    val_ap = K._score(yval, sum_val / counts.sum())[0]
    test_ap = K._score(yte, sum_test / counts.sum())[0]
    return val_ap, test_ap, w.tolist()


def run_cell(X_dev, y_dev, X_test, y_test, meta, seed):
    y_dev = np.asarray(y_dev)
    Xtr, Xval, ytr, yval = train_test_split(
        X_dev, y_dev, test_size=K.VAL_FRAC, stratify=y_dev, random_state=seed)
    prep = K.C.make_preprocessor(meta); prep.fit(Xtr, ytr)
    Xtr_t = prep.transform(Xtr); Xval_t = prep.transform(Xval); Xte_t = prep.transform(X_test)
    spw = float((ytr == 0).sum() / max(1, (ytr == 1).sum()))
    y_test = np.asarray(y_test)

    names, val_p, test_p, times = [], [], [], []
    anytime = []
    cum = 0.0
    for nm, est in base_models(seed, spw):
        t0 = time.perf_counter()
        try:
            est.fit(Xtr_t, ytr)
            vp = est.predict_proba(Xval_t)[:, 1]
            tp = est.predict_proba(Xte_t)[:, 1]
        except Exception as e:
            print(f"      [{nm}] fallo {type(e).__name__}", flush=True)
            continue
        dt = time.perf_counter() - t0; cum += dt
        names.append(nm); val_p.append(vp); test_p.append(tp); times.append(dt)
        # mejor individual hasta ahora
        aps = [K._score(yval, p)[0] for p in val_p]
        bi = int(np.nanargmax(aps))
        best_single_val = aps[bi]
        best_single_test = K._score(y_test, test_p[bi])[0]
        # ensemble greedy hasta ahora
        if len(val_p) >= 2:
            ens_val, ens_test, w = greedy_ensemble(val_p, yval, test_p, y_test)
        else:
            ens_val, ens_test = best_single_val, best_single_test
        anytime.append({
            "k": len(names), "added": nm, "cum_time": cum,
            "best_single_val": best_single_val, "best_single_test": best_single_test,
            "best_single_name": names[bi],
            "ens_val": ens_val, "ens_test": ens_test,
        })
    return {"names": names, "per_model_time": dict(zip(names, times)),
            "anytime": anytime}


def run_cell_e2(X_dev, y_dev, X_test, y_test, meta, seed):
    """Ensemble estilo E2: pesos del blend sobre predicciones OOF de la CV (usa TODO
    dev), y familias reentrenadas en todo dev para el modelo final. Comparable a E2."""
    from sklearn.model_selection import StratifiedKFold
    y_dev = np.asarray(y_dev); y_test = np.asarray(y_test)
    Xd = X_dev.reset_index(drop=True)
    spw = float((y_dev == 0).sum() / max(1, (y_dev == 1).sum()))
    names = [nm for nm, _ in base_models(seed, spw)]
    oof = {nm: np.full(len(y_dev), np.nan) for nm in names}
    bag = {nm: np.zeros(len(y_test)) for nm in names}   # predicciones bagged de test (sin-retrain)
    t0 = time.perf_counter()
    # 1) OOF de cada familia + predicciones de test de cada modelo de fold (para bagged)
    skf = StratifiedKFold(n_splits=K.CV_K, shuffle=True, random_state=seed)
    for tr, ev in skf.split(Xd, y_dev):
        prep = K.C.make_preprocessor(meta); prep.fit(Xd.iloc[tr], y_dev[tr])
        Xtr_t = prep.transform(Xd.iloc[tr]); Xev_t = prep.transform(Xd.iloc[ev])
        Xte_f = prep.transform(X_test)
        for nm, est in base_models(seed, spw):
            try:
                est.fit(Xtr_t, y_dev[tr])
                oof[nm][ev] = est.predict_proba(Xev_t)[:, 1]
                bag[nm] += est.predict_proba(Xte_f)[:, 1] / K.CV_K
            except Exception:
                oof[nm][ev] = np.nan
    t_bag = time.perf_counter() - t0            # tiempo de la versión SIN retrain (solo CV)
    # 2) refit de cada familia en TODO dev -> predicción en test (versión CON retrain)
    prep = K.C.make_preprocessor(meta); prep.fit(Xd, y_dev)
    Xd_t = prep.transform(Xd); Xte_t = prep.transform(X_test)
    test_p, bag_p, keep = [], [], []
    for nm, est in base_models(seed, spw):
        if np.isnan(oof[nm]).any():
            continue
        try:
            est.fit(Xd_t, y_dev)
            test_p.append(est.predict_proba(Xte_t)[:, 1]); bag_p.append(bag[nm]); keep.append(nm)
        except Exception:
            pass
    t = time.perf_counter() - t0               # tiempo de la versión CON retrain
    oof_list = [oof[nm] for nm in keep]
    if len(oof_list) < 2:
        return {"e2_ens_val": np.nan, "e2_ens_test": np.nan, "e2_ens_time": t,
                "e2_single_test": np.nan, "e2nr_ens_test": np.nan, "e2nr_ens_time": t_bag}
    ens_val, ens_test, w = greedy_ensemble(oof_list, y_dev, test_p, y_test)      # retrain
    _, ens_test_nr, _ = greedy_ensemble(oof_list, y_dev, bag_p, y_test)          # bagged (sin retrain)
    best_single_test = max(K._score(y_test, tp)[0] for tp in test_p)
    return {"e2_ens_val": float(ens_val), "e2_ens_test": float(ens_test),
            "e2_ens_time": float(t), "e2_single_test": float(best_single_test),
            "e2nr_ens_test": float(ens_test_nr), "e2nr_ens_time": float(t_bag),
            "e2_weights": {nm: float(wi) for nm, wi in zip(keep, w) if wi > 0}}


def run_dataset(name, fracs=None, seeds=None, tag="familiesf", mode="e1"):
    fracs = fracs or FRACS_C
    seeds = seeds or SEEDS_C
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{K.RES}/{tag}__{safe}.json"
    done = {}
    if os.path.exists(outp):
        done = {(r["frac"], r["seed"]): r for r in json.load(open(outp))}
    X, y, meta = K.C.load_dataset(name)
    results = list(done.values())
    prevalence = float(np.mean(y))
    for seed in seeds:
        Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), K.DEV_CAP)
        for frac in fracs:
            size = int(round(frac * dev_base))
            if (frac, seed) in done:
                continue
            Xdev, ydev = K.subsample(Xdev_full, ydev_full, size, seed)
            if ydev.sum() < K.CV_K or (ydev == 0).sum() < K.CV_K:
                continue
            t0 = time.perf_counter()
            try:
                if mode == "e2":
                    r = run_cell_e2(Xdev, ydev, Xte, yte, meta, seed)
                elif mode == "both":
                    r = run_cell(Xdev, ydev, Xte, yte, meta, seed)
                    r.update(run_cell_e2(Xdev, ydev, Xte, yte, meta, seed))
                else:
                    r = run_cell(Xdev, ydev, Xte, yte, meta, seed)
            except Exception as e:
                r = {"error": f"{type(e).__name__}: {e}"}
            r.update({"dataset": name, "frac": frac, "size": int(len(ydev)),
                      "dev_base": int(dev_base), "seed": seed,
                      "n_test": int(len(yte)), "prevalence": prevalence,
                      "imb_ratio": float(meta["imbalance_ratio"]),
                      "wall_time": time.perf_counter() - t0, "tag": tag})
            results.append(r); done[(frac, seed)] = r
            json.dump(results, open(outp, "w"), indent=1)
            if "e2_ens_test" in r and not np.isnan(r.get("e2_ens_test", np.nan)):
                print(f"  [{name}] frac={frac} (n={len(ydev)}) seed={seed} "
                      f"E2-ens_test={r['e2_ens_test']:.4f} t={r['wall_time']:.0f}s", flush=True)
            elif "anytime" in r and r["anytime"]:
                a = r["anytime"][-1]
                print(f"  [{name}] frac={frac} (n={len(ydev)}) seed={seed} "
                      f"ens_test={a['ens_test']:.4f} best_single_test={a['best_single_test']:.4f} "
                      f"t={r['wall_time']:.0f}s", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--fracs", type=str, default="")
    ap.add_argument("--tag", type=str, default="familiesf")
    ap.add_argument("--mode", type=str, default="e1", choices=["e1", "e2", "both"])
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    fracs = [float(s) for s in args.fracs.split(",")] if args.fracs else None
    names = args.datasets or K.C.list_materialized()
    print(f"[families] {len(names)} datasets seeds={seeds} mode={args.mode}", flush=True)
    for i, nm in enumerate(names):
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, fracs=fracs, seeds=seeds, tag=args.tag, mode=args.mode)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
