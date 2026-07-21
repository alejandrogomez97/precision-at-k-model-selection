"""optuna_study.py — Fase B. Igual que core.py pero usando Optuna (TPE) para la
búsqueda de hiperparámetros de LightGBM en lugar del grid fijo, en ambas
estrategias E1 y E2. Se registra tiempo de búsqueda y AP en test para comparar
Optuna vs grid a igualdad de coste temporal.

El objetivo que optimiza Optuna es el MISMO criterio de selección de cada
estrategia:
  E1 -> AP en el conjunto val independiente (nº de árboles por CV en train)
  E2 -> AP OOF de la CV sobre dev
Tras N trials, el mejor candidato se reentrena en dev y se evalúa en el test fijo.
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")

STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
from sklearn.model_selection import train_test_split
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 40


def _suggest(trial):
    return {
        "num_leaves": trial.suggest_int("num_leaves", 7, 255, log=True),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "imb": trial.suggest_categorical("imb", list(K.IMB_TECHNIQUES)),
    }


def optuna_eval(strat, X_dev, y_dev, X_test, y_test, meta, seed, n_trials=N_TRIALS):
    y_dev = np.asarray(y_dev)
    state = {"time": 0.0}
    if strat == "E1":
        Xtr, Xval, ytr, yval = train_test_split(
            X_dev, y_dev, test_size=K.VAL_FRAC, stratify=y_dev, random_state=seed)
        Xtr = Xtr.reset_index(drop=True); Xval = Xval.reset_index(drop=True)

        def objective(trial):
            cand = _suggest(trial)
            t0 = time.perf_counter()
            n_trees, _, _, _ = K.cv_best_iter_and_oof(Xtr, ytr, meta, cand, seed)
            r = K.fit_fold(Xtr, ytr, Xval, yval, meta, cand, seed,
                           early_stop=False, n_trees=n_trees)
            state["time"] += time.perf_counter() - t0
            ap, _ = K._score(yval, r["proba"])
            trial.set_user_attr("n_trees", n_trees)
            trial.set_user_attr("cand", cand)
            return ap if ap == ap else -1.0
    else:
        Xd = X_dev.reset_index(drop=True)

        def objective(trial):
            cand = _suggest(trial)
            t0 = time.perf_counter()
            n_trees, _, oof, _ = K.cv_best_iter_and_oof(Xd, y_dev, meta, cand, seed,
                                                        want_oof=True)
            state["time"] += time.perf_counter() - t0
            m = ~np.isnan(oof)
            ap, _ = K._score(y_dev[m], oof[m])
            trial.set_user_attr("n_trees", n_trees)
            trial.set_user_attr("cand", cand)
            return ap if ap == ap else -1.0

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best = study.best_trial
    cand = best.user_attrs["cand"]; n_trees = best.user_attrs["n_trees"]
    t0 = time.perf_counter()
    r = K.fit_fold(X_dev, y_dev, X_test, y_test, meta, cand, seed,
                   early_stop=False, n_trees=n_trees)
    ft = time.perf_counter() - t0
    ap, ll = K._score(y_test, r["proba"])
    return {"strat": strat, "search_time": state["time"], "n_trials": n_trials,
            "final_fit_time_by_ap": ft, "test_ap_by_ap": ap, "test_ll_by_ap": ll,
            "best_val_ap": float(best.value),
            "winner_by_ap": {**cand, "n_trees": n_trees}}


def run_dataset(name, fracs=None, seeds=None, tag="optunaf", n_trials=N_TRIALS):
    fracs = fracs or K.FRACS
    seeds = seeds or K.SEEDS
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{K.RES}/{tag}__{safe}.json"
    done = {}
    if os.path.exists(outp):
        done = {(r["frac"], r["seed"], r["strat"]): r for r in json.load(open(outp))}
    X, y, meta = K.C.load_dataset(name)
    results = list(done.values())
    prevalence = float(np.mean(y))
    for seed in seeds:
        Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), K.DEV_CAP)
        for frac in fracs:
            size = int(round(frac * dev_base))
            Xdev, ydev = K.subsample(Xdev_full, ydev_full, size, seed)
            if ydev.sum() < K.CV_K or (ydev == 0).sum() < K.CV_K:
                continue
            for strat in ("E1", "E2"):
                if (frac, seed, strat) in done:
                    continue
                t0 = time.perf_counter()
                try:
                    r = optuna_eval(strat, Xdev, ydev, Xte, yte, meta, seed, n_trials)
                except Exception as e:
                    r = {"strat": strat, "error": f"{type(e).__name__}: {e}"}
                r.update({"dataset": name, "frac": frac, "size": int(len(ydev)),
                          "dev_base": int(dev_base), "seed": seed,
                          "n_test": int(len(yte)), "prevalence": prevalence,
                          "imb_ratio": float(meta["imbalance_ratio"]),
                          "n_features": int(meta["n_features"]),
                          "wall_time": time.perf_counter() - t0, "tag": tag})
                results.append(r); done[(frac, seed, strat)] = r
                json.dump(results, open(outp, "w"), indent=1)
                print(f"  [{name}] frac={frac} (n={len(ydev)}) seed={seed} {strat} "
                      f"test_AP={r.get('test_ap_by_ap', float('nan')):.4f} "
                      f"search={r.get('search_time', 0):.1f}s", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--fracs", type=str, default="")
    ap.add_argument("--tag", type=str, default="optunaf")
    ap.add_argument("--trials", type=int, default=N_TRIALS)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    fracs = [float(s) for s in args.fracs.split(",")] if args.fracs else None
    names = args.datasets or K.C.list_materialized()
    print(f"[optuna] {len(names)} datasets seeds={seeds} trials={args.trials}", flush=True)
    for i, nm in enumerate(names):
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, fracs=fracs, seeds=seeds, tag=args.tag, n_trials=args.trials)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
