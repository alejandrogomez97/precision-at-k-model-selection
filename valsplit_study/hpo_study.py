"""hpo_study.py — ¿mejora la optimización de hiperparámetros al aumentar el
presupuesto, y hay métodos mejores que Optuna-TPE?

Marco E1 (dev -> train/val; nº de árboles por CV en train; selección en val; el
ganador se reentrena en dev y se evalúa en test). Para cada método se evalúan
hasta 160 configuraciones y se hacen CHECKPOINTS del mejor-hasta-ahora a
{20,40,80,160} configs (una sola corrida da todos los presupuestos).

Métodos: grid (grid grande barajado), random, TPE, CMA-ES (samplers de Optuna).
Salida: results/hpo__*.json con (method, budget, test_ap, search_time).
"""
import os, sys, json, time, random, warnings
import numpy as np
warnings.filterwarnings("ignore")
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
from sklearn.model_selection import train_test_split
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

CHECKPOINTS = [20, 40, 80, 160]
MAXB = 160
FRACS_H = [0.3, 1.0]
SEEDS_H = [0, 1]
METHODS = ["grid", "random", "tpe", "cmaes"]


def _space(trial):
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


def grid_configs(n=MAXB, seed=0):
    space = []
    for nl in (7, 15, 31, 63, 127):
        for mcs in (5, 20, 50):
            for lr in (0.02, 0.05, 0.1, 0.2):
                for rl in (0.0, 1.0):
                    for imb in K.IMB_TECHNIQUES:
                        space.append({"num_leaves": nl, "min_child_samples": mcs,
                                      "learning_rate": lr, "reg_lambda": rl, "imb": imb})
    random.Random(seed).shuffle(space)
    return space[:n]


def eval_config(cand, Xtr, ytr, Xval, yval, meta, seed):
    """AP en val de un candidato (nº de árboles por CV en train). Devuelve (ap, n_trees, t)."""
    t0 = time.perf_counter()
    n_trees, _, _, _ = K.cv_best_iter_and_oof(Xtr, ytr, meta, cand, seed)
    r = K.fit_fold(Xtr, ytr, Xval, yval, meta, cand, seed, early_stop=False, n_trees=n_trees)
    ap, _ = K._score(yval, r["proba"])
    return (ap if ap == ap else -1.0), n_trees, time.perf_counter() - t0


def collect(method, Xtr, ytr, Xval, yval, meta, seed):
    """Evalúa hasta MAXB configs con el método dado. Lista de {cand,val_ap,n_trees,t}."""
    evals = []
    if method == "grid":
        for cand in grid_configs(MAXB, seed):
            ap, nt, t = eval_config(cand, Xtr, ytr, Xval, yval, meta, seed)
            evals.append({"cand": cand, "val_ap": ap, "n_trees": nt, "t": t})
    else:
        sampler = {"random": optuna.samplers.RandomSampler(seed=seed),
                   "tpe": optuna.samplers.TPESampler(seed=seed),
                   "cmaes": optuna.samplers.CmaEsSampler(seed=seed),
                   "gp": optuna.samplers.GPSampler(seed=seed)}[method]
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def obj(trial):
            cand = _space(trial)
            ap, nt, t = eval_config(cand, Xtr, ytr, Xval, yval, meta, seed)
            trial.set_user_attr("cand", cand); trial.set_user_attr("nt", nt)
            trial.set_user_attr("t", t)
            return ap
        study.optimize(obj, n_trials=MAXB)
        for tr in sorted(study.trials, key=lambda x: x.number):
            evals.append({"cand": tr.user_attrs["cand"], "val_ap": tr.value,
                          "n_trees": tr.user_attrs["nt"], "t": tr.user_attrs["t"]})
    return evals


def checkpoints(evals, X_dev, y_dev, X_test, y_test, meta, seed, method):
    cum = 0.0
    for e in evals:
        cum += e["t"]; e["cum"] = cum
    out = []
    for B in CHECKPOINTS:
        sub = evals[:B]
        valid = [e for e in sub if e["val_ap"] == e["val_ap"] and e["val_ap"] > -1]
        if not valid:
            continue
        best = max(valid, key=lambda e: e["val_ap"])
        r = K.fit_fold(X_dev, y_dev, X_test, y_test, meta, best["cand"], seed,
                       early_stop=False, n_trees=best["n_trees"])
        ap, _ = K._score(y_test, r["proba"])
        out.append({"method": method, "budget": B, "test_ap": float(ap),
                    "val_ap": float(best["val_ap"]), "search_time": float(sub[-1]["cum"])})
    return out


def run_dataset(name, fracs=None, seeds=None, methods=None, tag="hpo"):
    fracs = fracs or FRACS_H; seeds = seeds or SEEDS_H; methods = methods or METHODS
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{K.RES}/{tag}__{safe}.json"
    done = set()
    results = []
    if os.path.exists(outp):
        results = json.load(open(outp))
        done = {(r["frac"], r["seed"], r["method"], r["budget"]) for r in results}
    X, y, meta = K.C.load_dataset(name)
    for seed in seeds:
        Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), K.DEV_CAP)
        for frac in fracs:
            Xdev, ydev = K.subsample(Xdev_full, ydev_full, int(round(frac * dev_base)), seed)
            if ydev.sum() < K.CV_K or (ydev == 0).sum() < K.CV_K:
                continue
            Xtr, Xval, ytr, yval = train_test_split(
                Xdev, ydev, test_size=K.VAL_FRAC, stratify=ydev, random_state=seed)
            Xtr = Xtr.reset_index(drop=True); Xval = Xval.reset_index(drop=True)
            for method in methods:
                if all((frac, seed, method, B) in done for B in CHECKPOINTS):
                    continue
                t0 = time.perf_counter()
                try:
                    evals = collect(method, Xtr, ytr, Xval, yval, meta, seed)
                    ck = checkpoints(evals, Xdev, ydev, Xte, yte, meta, seed, method)
                except Exception as e:
                    print(f"  !! {name} {method} f{frac} s{seed}: {type(e).__name__}: {e}", flush=True)
                    continue
                for c in ck:
                    c.update({"dataset": name, "frac": frac, "seed": seed,
                              "n_test": int(len(yte)), "imb_ratio": float(meta["imbalance_ratio"])})
                    results.append(c); done.add((frac, seed, method, c["budget"]))
                json.dump(results, open(outp, "w"), indent=1)
                best160 = max((c["test_ap"] for c in ck), default=float("nan"))
                print(f"  [{name}] f{frac} s{seed} {method:7s} "
                      f"test_AP@160={best160:.4f} ({time.perf_counter()-t0:.0f}s)", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--fracs", type=str, default="")
    ap.add_argument("--methods", type=str, default="")
    ap.add_argument("--tag", type=str, default="hpo")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    fracs = [float(s) for s in a.fracs.split(",")] if a.fracs else None
    methods = a.methods.split(",") if a.methods else None
    names = a.datasets or K.C.list_materialized()
    print(f"[hpo] {len(names)} datasets seeds={seeds} methods={methods or METHODS}", flush=True)
    for i, nm in enumerate(names):
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, fracs=fracs, seeds=seeds, methods=methods, tag=a.tag)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
