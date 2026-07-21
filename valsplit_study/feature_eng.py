"""feature_eng.py — Apartado 10. ¿Estorban las variables inventadas "a lo loco"?

Creencia a poner a prueba: "el modelo (árboles) descarta las variables irrelevantes,
así que aunque no aporten no molestan; y a veces das por casualidad con un patrón
que mejora". Lo comprobamos con la estrategia E2 usando LightGBM (el mejor modelo
ÚNICO del estudio y justo el árbol cuya robustez se discute; se evita ens-E2 porque
sus miembros no-árbol —logística, kNN, MLP— sí sufren con la basura y confundirían la
pregunta, además de ser ~5× más caro). Al 100% del dev añadimos variables inventadas
al azar (combinaciones sin sentido de pares de features: v_i/v_j, v_i*v_j, v_i-v_j,
ruido…) en proporciones crecientes (10%, 20%, …, 100% del nº de features originales)
y medimos el AP en test.

Salida: results/feateng__*.json  (por dataset, seed, nivel de basura -> AP).
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
STUDY = "/home/agomez/proyectos/precision-at-k-study/valsplit_study"
sys.path.insert(0, STUDY)
import core as K
import families_study as F

DEV_CAP_FE = 8000     # 100% del dev, capado por coste (ens-E2 con muchas features es caro)
LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SEEDS = [0, 1]
DATASETS = ["jm1", "online_shoppers_intention", "Pulsar-Dataset-HTRU2", "letter",
            "satimage", "mammography", "coil_2000", "pendigits"]
OPS = ["ratio", "prod", "diff", "sum", "absdiff", "noise"]


def junk_recipes(num_cols, n_junk, rng):
    """Genera n_junk 'recetas' (op, i, j) sobre columnas numéricas originales."""
    p = len(num_cols)
    recs = []
    for t in range(n_junk):
        op = OPS[rng.randint(len(OPS))]
        i, j = rng.randint(p), rng.randint(p)
        recs.append((op, num_cols[i], num_cols[j], t))
    return recs


def apply_junk(Xdf, recs, rng):
    """Aplica las recetas a un DataFrame -> columnas nuevas junk_t. El ruido se
    genera aparte por conjunto (es ruido, no tiene por qué ser reproducible)."""
    new = {}
    for op, ci, cj, t in recs:
        a = pd.to_numeric(Xdf[ci], errors="coerce").fillna(0).values.astype(float)
        b = pd.to_numeric(Xdf[cj], errors="coerce").fillna(0).values.astype(float)
        if op == "ratio":   c = a / (np.abs(b) + 1e-6)
        elif op == "prod":  c = a * b
        elif op == "diff":  c = a - b
        elif op == "sum":   c = a + b
        elif op == "absdiff": c = np.abs(a - b)
        else:               c = rng.randn(len(a))
        new[f"junk_{t}"] = np.clip(c, -1e9, 1e9)
    return pd.DataFrame(new, index=Xdf.index)


def run_dataset(name, seeds=None, levels=None, tag="feateng"):
    seeds = seeds or SEEDS; levels = levels or LEVELS
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    outp = f"{K.RES}/{tag}__{safe}.json"
    done = {(r["seed"], r["level"]) for r in json.load(open(outp))} if os.path.exists(outp) else set()
    results = json.load(open(outp)) if os.path.exists(outp) else []
    X, y, meta = K.C.load_dataset(name)
    num_cols = meta["num_cols"]
    n_orig = len(num_cols)
    if n_orig < 2:
        print(f"  {name}: <2 num cols, skip"); return
    for seed in seeds:
        Xdev_full, ydev_full, Xte, yte = K.get_dev_test(X, y, seed)
        dev_base = min(len(ydev_full), DEV_CAP_FE)
        Xdev, ydev = K.subsample(Xdev_full, ydev_full, dev_base, seed)
        if ydev.sum() < K.CV_K:
            continue
        for lv in levels:
            if (seed, lv) in done:
                continue
            n_junk = int(round(lv * n_orig))
            rng = np.random.RandomState(1234 + seed)
            recs = junk_recipes(num_cols, n_junk, rng)
            meta2 = dict(meta)
            if n_junk > 0:
                Jd = apply_junk(Xdev, recs, np.random.RandomState(seed))
                Jt = apply_junk(Xte, recs, np.random.RandomState(seed + 999))
                Xd2 = pd.concat([Xdev.reset_index(drop=True), Jd.reset_index(drop=True)], axis=1)
                Xt2 = pd.concat([Xte.reset_index(drop=True), Jt.reset_index(drop=True)], axis=1)
                meta2["num_cols"] = list(num_cols) + list(Jd.columns)
                meta2["n_features"] = meta["n_features"] + n_junk
            else:
                Xd2, Xt2 = Xdev.reset_index(drop=True), Xte.reset_index(drop=True)
            t0 = time.perf_counter()
            try:
                r = K.eval_E2(Xd2, ydev, Xt2, yte, meta2, K.make_candidates(), seed)
                ap = r["test_ap_by_ap"]
            except Exception as e:
                print(f"  !! {name} s{seed} lv{lv}: {type(e).__name__}: {e}", flush=True); continue
            rec = {"dataset": name, "seed": seed, "level": lv, "n_orig": n_orig,
                   "n_junk": n_junk, "n_total": n_orig + n_junk, "ap": float(ap),
                   "n_dev": int(len(ydev)), "wall_time": time.perf_counter() - t0}
            results.append(rec); done.add((seed, lv))
            json.dump(results, open(outp, "w"), indent=1)
            print(f"  [{name}] s{seed} basura={lv:.0%} ({n_junk}/{n_orig}) AP={ap:.4f} "
                  f"({rec['wall_time']:.0f}s)", flush=True)
    return outp


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="*")
    ap.add_argument("--seeds", type=str, default="0,1")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",") if s]
    names = args.datasets or DATASETS
    print(f"[feateng] {len(names)} datasets, niveles={LEVELS}", flush=True)
    for i, nm in enumerate(names):
        print(f"[{i+1}/{len(names)}] {nm}", flush=True)
        try:
            run_dataset(nm, seeds=seeds)
        except Exception as e:
            print(f"  !! {nm}: {type(e).__name__}: {e}", flush=True)
