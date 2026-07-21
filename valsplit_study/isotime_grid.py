"""isotime_grid.py — Apartado 8. Versiones @t* de E1 y E2: se deja crecer el grid de
LightGBM (más configuraciones) hasta consumir el mismo tiempo que gasta ens-E2 (t*),
para comparar a igualdad de tiempo. Marco E1 (train/val) y E2 (solo CV).

Para cada celda (dataset, fracción, semilla), con t* de results/ens_e2f:
  - E1@t*: evalúa configs (grid grande barajado); cada una: CV-árboles en train + fit
    en train, puntúa en val; el mejor se reentrena en dev -> test. Para al superar t*.
  - E2@t*: igual pero selección por OOF de la CV sobre dev; refit en dev -> test.
Salida: results/isogrid__*.json
"""
import os, sys, json, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
import hpo_study as H
from sklearn.model_selection import train_test_split

DATASETS = ["jm1", "online_shoppers_intention", "Bank_marketing_data_set_UCI",
            "Pulsar-Dataset-HTRU2", "letter", "satimage", "mammography", "rl"]
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [0, 1]
MAXC = 800   # subido para que el grid llegue de verdad a t* (antes se quedaba corto)


def load_tstar():
    out = {}
    import glob
    for f in glob.glob(f"{K.RES}/ens_e2f__*.json"):
        for r in json.load(open(f)):
            if r.get("e2_ens_time") is not None and r["e2_ens_time"] == r["e2_ens_time"]:
                out[(r["dataset"], r["frac"], r["seed"])] = float(r["e2_ens_time"])
    return out


def e1_at_tstar(Xtr, ytr, Xval, yval, Xdev, ydev, Xte, yte, meta, seed, t_star):
    """Grid E1 creciendo configs hasta t*: mejor-en-val -> refit en dev -> test."""
    cum, best = 0.0, None
    for cand in H.grid_configs(MAXC, seed):
        ap, nt, t = H.eval_config(cand, Xtr, ytr, Xval, yval, meta, seed)
        cum += t
        if best is None or ap > best["ap"]:
            best = {"cand": cand, "nt": nt, "ap": ap}
        if cum >= t_star:
            break
    r = K.fit_fold(Xdev, ydev, Xte, yte, meta, best["cand"], seed, early_stop=False, n_trees=best["nt"])
    return K._score(yte, r["proba"])[0], cum


def e2_at_tstar(Xdev, ydev, Xte, yte, meta, seed, t_star):
    """Grid E2 creciendo configs hasta t*: mejor-en-OOF -> refit en dev -> test."""
    cum, best = 0.0, None
    for cand in H.grid_configs(MAXC, seed):
        t0 = time.perf_counter()
        n_trees, _, oof, _ = K.cv_best_iter_and_oof(Xdev, ydev, meta, cand, seed, want_oof=True)
        m = ~np.isnan(oof); ap, _ = K._score(np.asarray(ydev)[m], oof[m])
        cum += time.perf_counter() - t0
        if best is None or ap > best["ap"]:
            best = {"cand": cand, "nt": n_trees, "ap": ap}
        if cum >= t_star:
            break
    r = K.fit_fold(Xdev, ydev, Xte, yte, meta, best["cand"], seed, early_stop=False, n_trees=best["nt"])
    return K._score(yte, r["proba"])[0], cum


def run_dataset(name, tstar, tag="isogrid"):
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{K.RES}/{tag}__{safe}.json"
    done = {(r["frac"], r["seed"]) for r in json.load(open(outp))} if os.path.exists(outp) else set()
    results = json.load(open(outp)) if os.path.exists(outp) else []
    X, y, meta = K.C.load_dataset(name)
    for seed in SEEDS:
        Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), K.DEV_CAP)
        for frac in FRACS:
            if (frac, seed) in done or (name, frac, seed) not in tstar:
                continue
            t_star = tstar[(name, frac, seed)]
            Xdev, ydev = K.subsample(Xdev_full, ydev_full, int(round(frac * dev_base)), seed)
            if ydev.sum() < K.CV_K:
                continue
            Xtr, Xval, ytr, yval = train_test_split(Xdev, ydev, test_size=K.VAL_FRAC,
                                                    stratify=ydev, random_state=seed)
            Xtr = Xtr.reset_index(drop=True); Xval = Xval.reset_index(drop=True)
            try:
                e1_ap, e1_t = e1_at_tstar(Xtr, ytr, Xval, yval, Xdev, ydev, Xte, yte, meta, seed, t_star)
                e2_ap, e2_t = e2_at_tstar(Xdev, ydev, Xte, yte, meta, seed, t_star)
            except Exception as e:
                print(f"  !! {name} f{frac} s{seed}: {type(e).__name__}: {e}", flush=True); continue
            rec = {"dataset": name, "frac": frac, "seed": seed, "t_star": t_star,
                   "e1_tstar_ap": float(e1_ap), "e1_tstar_time": float(e1_t),
                   "e2_tstar_ap": float(e2_ap), "e2_tstar_time": float(e2_t)}
            results.append(rec); done.add((frac, seed))
            json.dump(results, open(outp, "w"), indent=1)
            print(f"  [{name}] f{frac} s{seed} t*={t_star:.0f}s: E1@t*={e1_ap:.4f} "
                  f"E2@t*={e2_ap:.4f}", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    a = ap.parse_args()
    tstar = load_tstar()
    names = a.datasets or DATASETS
    print(f"[isogrid] {len(names)} datasets", flush=True)
    for i, nm in enumerate(names):
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, tstar)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
