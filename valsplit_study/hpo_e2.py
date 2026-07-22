"""hpo_e2.py — MISMO estudio de HPO que hpo_study.py/hpo_mf.py pero en marco E2
(coherente con la Parte 1: la selección de configuración se hace por AP OOF de la CV,
sin conjunto de validación separado).

  - full-budget (grid/random/tpe/cmaes/gp): cada config se puntúa por su AP OOF
    (cv_best_iter_and_oof), que también fija el nº de árboles; el ganador se reentrena
    en TODO dev y se evalúa en test.
  - multi-fidelity (hyperband/bohb): K folds entrenados INCREMENTALMENTE (init_model, sin
    reentrenar por rung -> no infla el tiempo de las supervivientes); en cada rung de nº de
    árboles se calcula el AP OOF y se poda con HyperbandPruner.

Salida: results/hpoe2__*.json (mismo formato: method,budget,test_ap,search_time).
"""
import os, sys, json, time, warnings, gc
import numpy as np
warnings.filterwarnings("ignore")
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
import hpo_study as H
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

METHODS_FULL = ["grid", "random", "tpe", "cmaes", "gp"]
METHODS_MF = ["hyperband", "bohb"]
MIN_RES, MAX_RES, RF = 50, 2000, 3


# ---------- full-budget: selección por AP OOF ----------
def eval_config_e2(cand, Xdev, ydev, meta, seed):
    t0 = time.perf_counter()
    n_trees, _, oof, _ = K.cv_best_iter_and_oof(Xdev, ydev, meta, cand, seed, want_oof=True)
    m = ~np.isnan(oof)
    ap, _ = K._score(np.asarray(ydev)[m], oof[m])
    return (ap if ap == ap else -1.0), int(n_trees), time.perf_counter() - t0


# ---------- multi-fidelity: OOF por rungs, entrenamiento incremental ----------
def _native_params(cand, ytr):
    p = dict(objective="binary", metric="average_precision",
             learning_rate=cand.get("learning_rate", K.LR), num_leaves=cand["num_leaves"],
             reg_lambda=cand["reg_lambda"], reg_alpha=cand.get("reg_alpha", 0.0),
             min_child_samples=cand.get("min_child_samples", 20),
             subsample=cand.get("subsample", 0.9), subsample_freq=1,
             colsample_bytree=cand.get("colsample_bytree", 0.8),
             num_threads=1, verbosity=-1, seed=42)
    if cand["imb"] == "class_weight":
        pos = max(1, int((ytr == 1).sum())); neg = int((ytr == 0).sum())
        p["scale_pos_weight"] = neg / pos
    return p


def _rungs():
    r, b = [], MIN_RES
    while b < MAX_RES:
        r.append(int(b)); b *= RF
    r.append(MAX_RES)
    return r


def eval_config_mf_e2(trial, cand, Xdev, ydev, meta, seed):
    yd = np.asarray(ydev); Xd = Xdev.reset_index(drop=True)
    skf = StratifiedKFold(n_splits=K.CV_K, shuffle=True, random_state=seed)
    # preparar cada fold una vez (preprocesado en train del fold + resample)
    folds = []
    for tr, ev in skf.split(Xd, yd):
        prep = K.C.make_preprocessor(meta); prep.fit(Xd.iloc[tr], yd[tr])
        Xtr = prep.transform(Xd.iloc[tr]); Xev = prep.transform(Xd.iloc[ev])
        Xtr2, ytr2 = K._resample(Xtr, yd[tr], cand["imb"], seed)
        params = _native_params(cand, ytr2)
        folds.append({"dtr": lgb.Dataset(Xtr2, label=ytr2, free_raw_data=False), "Xev": Xev, "ev": ev,
                      "params": params, "booster": None})
    oof = np.full(len(yd), np.nan)
    best_ap, best_b, prev = -1.0, MIN_RES, 0
    try:
        for b in _rungs():
            for f in folds:
                f["booster"] = lgb.train(f["params"], f["dtr"], num_boost_round=b - prev,
                                         init_model=f["booster"], keep_training_booster=True)
                oof[f["ev"]] = f["booster"].predict(f["Xev"])
            ap, _ = K._score(yd, oof)
            if ap == ap and ap > best_ap:
                best_ap, best_b = ap, b
            trial.report(ap if ap == ap else -1.0, b)
            prev = b
            if trial.should_prune():
                raise optuna.TrialPruned()
        return best_ap, int(best_b)
    finally:
        # liberar Datasets/boosters de LightGBM (free_raw_data=False acumula en C++)
        for f in folds:
            f["booster"] = None; f["dtr"] = None; f["Xev"] = None
        folds.clear()
        gc.collect()


# ---------- colectores ----------
def collect_e2(method, Xdev, ydev, meta, seed):
    evals = []
    if method == "grid":
        for cand in H.grid_configs(H.MAXB, seed):
            ap, nt, t = eval_config_e2(cand, Xdev, ydev, meta, seed)
            evals.append({"cand": cand, "val_ap": ap, "n_trees": nt, "t": t})
        return evals
    if method in METHODS_FULL:
        sampler = {"random": optuna.samplers.RandomSampler(seed=seed),
                   "tpe": optuna.samplers.TPESampler(seed=seed),
                   "cmaes": optuna.samplers.CmaEsSampler(seed=seed),
                   "gp": optuna.samplers.GPSampler(seed=seed)}[method]
        study = optuna.create_study(direction="maximize", sampler=sampler)

        def obj(trial):
            cand = H._space(trial); ap, nt, t = eval_config_e2(cand, Xdev, ydev, meta, seed)
            trial.set_user_attr("cand", cand); trial.set_user_attr("nt", nt); trial.set_user_attr("t", t)
            return ap
        study.optimize(obj, n_trials=H.MAXB)
        for tr in sorted(study.trials, key=lambda x: x.number):
            evals.append({"cand": tr.user_attrs["cand"], "val_ap": tr.value if tr.value is not None else -1.0,
                          "n_trees": tr.user_attrs["nt"], "t": tr.user_attrs["t"]})
        return evals
    # multi-fidelity
    pruner = optuna.pruners.HyperbandPruner(min_resource=MIN_RES, max_resource=MAX_RES, reduction_factor=RF)
    sampler = (optuna.samplers.RandomSampler(seed=seed) if method == "hyperband"
               else optuna.samplers.TPESampler(seed=seed))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def obj(trial):
        cand = H._space(trial); trial.set_user_attr("cand", cand); t0 = time.perf_counter()
        try:
            ap, nt = eval_config_mf_e2(trial, cand, Xdev, ydev, meta, seed)
            trial.set_user_attr("t", time.perf_counter() - t0); trial.set_user_attr("nt", nt)
            trial.set_user_attr("val_ap", ap); return ap
        except optuna.TrialPruned:
            trial.set_user_attr("t", time.perf_counter() - t0)
            trial.set_user_attr("nt", int(getattr(trial, "_best_b", MIN_RES)))
            raise
    study.optimize(obj, n_trials=H.MAXB)
    for tr in sorted(study.trials, key=lambda x: x.number):
        ua = tr.user_attrs
        if "cand" not in ua: continue
        evals.append({"cand": ua["cand"], "val_ap": float(ua.get("val_ap", -1.0)),
                      "n_trees": int(ua.get("nt", MIN_RES)), "t": float(ua.get("t", 0.0))})
    return evals


def run_dataset(name, fracs, seeds, tag="hpoe2"):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{K.RES}/{tag}__{safe}.json"
    results = json.load(open(outp)) if os.path.exists(outp) else []
    done = {(r["frac"], r["seed"], r["method"], r["budget"]) for r in results}
    X, y, meta = K.C.load_dataset(name)
    for seed in seeds:
        Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), K.DEV_CAP)
        for frac in fracs:
            Xdev, ydev = K.subsample(Xdev_full, ydev_full, int(round(frac * dev_base)), seed)
            if ydev.sum() < K.CV_K or (ydev == 0).sum() < K.CV_K:
                continue
            Xdev = Xdev.reset_index(drop=True)
            for method in METHODS_FULL + METHODS_MF:
                if all((frac, seed, method, B) in done for B in H.CHECKPOINTS):
                    continue
                t0 = time.perf_counter()
                try:
                    evals = collect_e2(method, Xdev, ydev, meta, seed)
                    ck = H.checkpoints(evals, Xdev, ydev, Xte, yte, meta, seed, method)
                except Exception as e:
                    print(f"  !! {name} {method} f{frac} s{seed}: {type(e).__name__}: {e}", flush=True)
                    continue
                for c in ck:
                    c.update({"dataset": name, "frac": frac, "seed": seed,
                              "n_test": int(len(yte)), "imb_ratio": float(meta["imbalance_ratio"])})
                    results.append(c); done.add((frac, seed, method, c["budget"]))
                json.dump(results, open(outp, "w"), indent=1)
                best160 = max((c["test_ap"] for c in ck), default=float("nan"))
                print(f"  [{name}] f{frac} s{seed} {method:9s} test_AP@160={best160:.4f} "
                      f"({time.perf_counter()-t0:.0f}s)", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--fracs", type=str, default="0.3,1.0")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    fracs = [float(s) for s in a.fracs.split(",") if s]
    print(f"[hpo_e2] {len(a.datasets)} datasets seeds={seeds} fracs={fracs}", flush=True)
    for i, nm in enumerate(a.datasets):
        print(f"[{i+1}/{len(a.datasets)}] {nm}", flush=True)
        try:
            run_dataset(nm, fracs, seeds)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
