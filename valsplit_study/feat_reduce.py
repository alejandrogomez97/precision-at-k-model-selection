"""feat_reduce.py — Cierre de la Parte 4. Si AÑADIR variables aleatorias no aporta,
¿aporta QUITAR variables redundantes (muy correlacionadas entre sí)?

Para cada dataset: se calcula la correlación de Spearman (en valor absoluto) entre pares
de features numéricas, medida SOLO en dev (sin fuga). Se barre el umbral
{0.75,0.80,0.85,0.90,0.95,0.99}: en cada nivel se eliminan las features que tienen
|Spearman| >= umbral con otra feature ya conservada (se queda la primera de cada grupo).
Se reporta cuántas se eliminan (nº y %) y la performance en test (AP y tiempo) con la
MISMA estrategia E2 + LightGBM (18 candidatas, selección por OOF) que el resto del estudio.

Salida: results/featreduce__*.json
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K

DEV_CAP_FE = 8000
SEEDS = [0, 1]
THRS = [0.75, 0.80, 0.85, 0.90, 0.95, 0.99]   # umbrales de |Spearman|
DATASETS = ["jm1", "online_shoppers_intention", "Pulsar-Dataset-HTRU2", "letter",
            "satimage", "mammography", "coil_2000", "pendigits"]


def corr_drop(Xdev, num_cols, thr):
    """Devuelve la lista de columnas a ELIMINAR: features con |Spearman| >= thr respecto
    a otra feature conservada anteriormente (se queda la primera de cada grupo correlado)."""
    if len(num_cols) < 2:
        return []
    Z = Xdev[num_cols].apply(pd.to_numeric, errors="coerce")
    Z = Z.fillna(Z.median(numeric_only=True))
    C = Z.corr(method="spearman").abs().values
    n = len(num_cols); drop = set()
    for i in range(n):
        if num_cols[i] in drop:
            continue
        for j in range(i + 1, n):
            if num_cols[j] in drop:
                continue
            c = C[i, j]
            if c == c and c >= thr:      # NaN-safe
                drop.add(num_cols[j])
    return [c for c in num_cols if c in drop]


def eval_reduced(Xdev, ydev, Xte, yte, meta, drop_cols, seed):
    """E2 + LightGBM sobre el dataset SIN las columnas eliminadas. Devuelve (ap, t)."""
    keep_num = [c for c in meta["num_cols"] if c not in set(drop_cols)]
    keep_all = keep_num + list(meta.get("cat_cols", []))
    m2 = dict(meta); m2["num_cols"] = keep_num
    m2["n_num"] = len(keep_num); m2["n_features"] = len(keep_all)
    Xd = Xdev[keep_all].reset_index(drop=True)
    Xt = Xte[keep_all].reset_index(drop=True)
    t0 = time.perf_counter()
    r = K.eval_E2(Xd, ydev, Xt, yte, m2, K.make_candidates(), seed)
    return float(r["test_ap_by_ap"]), time.perf_counter() - t0


def run_dataset(name, seeds=None, tag="featreduce"):
    seeds = seeds or SEEDS
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{K.RES}/{tag}__{safe}.json"
    done = {(r["seed"], r["thr"]) for r in json.load(open(outp))} if os.path.exists(outp) else set()
    results = json.load(open(outp)) if os.path.exists(outp) else []
    X, y, meta = K.C.load_dataset(name)
    num_cols = list(meta["num_cols"]); n_orig = len(num_cols)
    if n_orig < 2:
        print(f"  {name}: <2 num cols, skip"); return
    for seed in seeds:
        Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), DEV_CAP_FE)
        Xdev, ydev = K.subsample(Xdev_full, ydev_full, dev_base, seed)
        if ydev.sum() < K.CV_K:
            continue
        # thr=1.01 = baseline sin eliminar nada (mismo montaje, para comparar limpio)
        for thr in [1.01] + THRS:
            if (seed, thr) in done:
                continue
            drop = corr_drop(Xdev, num_cols, thr)
            try:
                ap, t = eval_reduced(Xdev, ydev, Xte, yte, meta, drop, seed)
            except Exception as e:
                print(f"  !! {name} s{seed} thr{thr}: {type(e).__name__}: {e}", flush=True); continue
            rec = {"dataset": name, "seed": seed, "thr": thr, "n_orig": n_orig,
                   "n_removed": len(drop), "pct_removed": round(100 * len(drop) / n_orig, 1),
                   "n_kept": n_orig - len(drop), "ap": ap, "wall_time": t,
                   "n_dev": int(len(ydev))}
            results.append(rec); done.add((seed, thr))
            json.dump(results, open(outp, "w"), indent=1)
            lab = "baseline" if thr > 1 else f"|ρ|>={thr}"
            print(f"  [{name}] s{seed} {lab:11s} quita {len(drop):3d}/{n_orig} "
                  f"({rec['pct_removed']:.0f}%) AP={ap:.4f} ({t:.0f}s)", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--seeds", type=str, default="0,1")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",") if s]
    names = a.datasets or DATASETS
    print(f"[featreduce] {len(names)} datasets, thrs={THRS}", flush=True)
    for i, nm in enumerate(names):
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, seeds=seeds)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
