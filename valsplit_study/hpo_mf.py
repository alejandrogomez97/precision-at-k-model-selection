"""hpo_mf.py — métodos de HPO MULTI-FIDELITY (Hyperband y BOHB) para el mismo estudio
que hpo_study.py, escribiendo en el mismo formato results/hpo__*.json.

Idea: en vez de entrenar cada configuración a presupuesto completo, se reporta el AP en
val DURANTE el boosting (cada `PERIOD` árboles) a un HyperbandPruner, que MATA pronto las
configuraciones malas (pocos árboles) y solo deja crecer las prometedoras. Fidelidad = nº
de árboles. Un solo entrenamiento por config (con early stopping), así que el tiempo de las
supervivientes NO se infla; las podadas cuestan poco -> ahí está el ahorro de tiempo.

  - hyperband: RandomSampler + HyperbandPruner  (Hyperband clásico)
  - bohb:      TPESampler   + HyperbandPruner    (Bayesian Optimization + Hyperband)

Se evalúan MAXB=160 trials (iniciados) y se hacen checkpoints del mejor-hasta-ahora a
{20,40,80,160}, igual que los demás métodos (comparables por nº de configs y por tiempo).
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
import hpo_study as H
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

PERIOD = 25          # cada cuántos árboles se reporta AP(val) al pruner
MIN_RES = 50         # recurso mínimo (árboles) antes de poder podar
METHODS_MF = ["hyperband", "bohb"]


def _prune_callback(trial):
    """Callback de LightGBM: lee el average_precision de val cada PERIOD árboles,
    lo reporta al pruner y guarda el mejor-hasta-ahora; poda si procede."""
    best = {"ap": -1.0, "it": 1}

    def cb(env):
        it = env.iteration + 1
        if it % PERIOD != 0 and it != env.end_iteration:
            return
        ap = None
        for item in env.evaluation_result_list:
            # item: (data_name, eval_name, value, is_higher_better[, ...])
            if "average_precision" in item[1] or item[1] == "ap":
                ap = float(item[2]); break
        if ap is None:
            return
        if ap > best["ap"]:
            best["ap"], best["it"] = ap, it
        trial.report(ap, it)
        trial.set_user_attr("val_ap", best["ap"]); trial.set_user_attr("nt", best["it"])
        if trial.should_prune():
            raise optuna.TrialPruned()
    cb.order = 10
    cb._best = best
    return cb


def train_mf(trial, cand, Xtr_raw, ytr, Xval_raw, yval, meta, seed):
    """Un entrenamiento con early stopping + pruning por árboles. Devuelve (best_ap, best_it)."""
    prep = K.C.make_preprocessor(meta); prep.fit(Xtr_raw, ytr)
    Xtr = prep.transform(Xtr_raw); Xval = prep.transform(Xval_raw)
    Xtr, ytr2 = K._resample(Xtr, np.asarray(ytr), cand["imb"], seed)
    params = K._lgbm_params(cand, imb=cand["imb"]); params["n_estimators"] = K.MAX_TREES
    clf = lgb.LGBMClassifier(**params)
    cb = _prune_callback(trial)
    clf.fit(Xtr, ytr2, eval_set=[(Xval, yval)], eval_metric="average_precision",
            callbacks=[lgb.early_stopping(K.ES_ROUNDS, verbose=False),
                       lgb.log_evaluation(0), cb])
    return cb._best["ap"], cb._best["it"]


def collect_mf(method, Xtr, ytr, Xval, yval, meta, seed):
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=MIN_RES, max_resource=K.MAX_TREES, reduction_factor=3)
    sampler = (optuna.samplers.RandomSampler(seed=seed) if method == "hyperband"
               else optuna.samplers.TPESampler(seed=seed))
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    def obj(trial):
        cand = H._space(trial)
        trial.set_user_attr("cand", cand)
        t0 = time.perf_counter()
        try:
            ap, it = train_mf(trial, cand, Xtr, ytr, Xval, yval, meta, seed)
            trial.set_user_attr("t", time.perf_counter() - t0)
            trial.set_user_attr("val_ap", ap); trial.set_user_attr("nt", it)
            return ap
        except optuna.TrialPruned:
            trial.set_user_attr("t", time.perf_counter() - t0)
            raise
    study.optimize(obj, n_trials=H.MAXB)
    evals = []
    for tr in sorted(study.trials, key=lambda x: x.number):
        ua = tr.user_attrs
        if "cand" not in ua:
            continue
        evals.append({"cand": ua["cand"], "val_ap": float(ua.get("val_ap", -1.0)),
                      "n_trees": int(ua.get("nt", 1)), "t": float(ua.get("t", 0.0))})
    return evals


def run_dataset(name, fracs, seeds, tag="hpo"):
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
            Xtr, Xval, ytr, yval = train_test_split(
                Xdev, ydev, test_size=K.VAL_FRAC, stratify=ydev, random_state=seed)
            Xtr = Xtr.reset_index(drop=True); Xval = Xval.reset_index(drop=True)
            for method in METHODS_MF:
                if all((frac, seed, method, B) in done for B in H.CHECKPOINTS):
                    continue
                t0 = time.perf_counter()
                try:
                    evals = collect_mf(method, Xtr, ytr, Xval, yval, meta, seed)
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
                npr = sum(1 for e in evals if e["val_ap"] <= -1 or e["n_trees"] < K.MAX_TREES)
                print(f"  [{name}] f{frac} s{seed} {method:9s} test_AP@160={best160:.4f} "
                      f"({time.perf_counter()-t0:.0f}s, {len(evals)} trials)", flush=True)
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
    names = a.datasets
    print(f"[hpo_mf] {len(names)} datasets seeds={seeds} fracs={fracs} methods={METHODS_MF}", flush=True)
    for i, nm in enumerate(names):
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, fracs, seeds)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
