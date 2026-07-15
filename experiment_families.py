"""
Robustness check across model families. Same protocol as experiment_real3.py
(K-sweep model selection over 93 datasets), but the model bank is Random Forest
or Logistic Regression instead of LightGBM. Question: does the conclusion hold
(log-loss / AP select better than precision@k)?

Reuses metrics, cleaning and dataset loading from experiment_real3.
Outputs results_families.json (rows tagged with 'family').
"""
import json, warnings, time
import numpy as np
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import experiment_real3 as E   # precision_at_k, auc_vec, ap_vec, logloss_vec, load_all

OUT = E.OUT
C_CONFIGS = 25
R_BOOT = 100
MAX_N = 15000
RATIOS = [0.25, 0.5, 1, 2, 4, 8, 16]
SEL = ["P@K", "AP", "AUC", "logloss"]

def make_logreg(rng):
    C = float(10 ** rng.uniform(-3, 2))
    cw = rng.choice([None, "balanced"])
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight=cw,
                                                max_iter=400, solver="lbfgs"))])

def make_rf(rng):
    depth = int(rng.choice([0, 6, 10, 16]))
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(
                         n_estimators=int(rng.integers(60, 160)),
                         max_depth=(None if depth == 0 else depth),
                         max_features=rng.choice(["sqrt", "log2", None]),
                         min_samples_leaf=int(rng.integers(1, 20)),
                         class_weight=rng.choice([None, "balanced", "balanced_subsample"]),
                         n_jobs=2, random_state=int(rng.integers(1, 1e6))))])

FACTORY = {"logreg": make_logreg, "rf": make_rf}

def sanitize(X):
    return np.where(np.isfinite(X), X, np.nan).astype(np.float32)

def run_one(family, name, X, y, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    if n > MAX_N:
        idx, _ = train_test_split(np.arange(n), train_size=MAX_N, stratify=y, random_state=seed)
        X, y = X[idx], y[idx]
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, train_size=0.5, stratify=y, random_state=seed)
    Xv, Xt, yv, yt = train_test_split(Xtmp, ytmp, train_size=0.5, stratify=ytmp, random_state=seed)
    if yv.sum() < 12 or yt.sum() < 12:
        return []
    Xtr, Xv, Xt = sanitize(Xtr), sanitize(Xv), sanitize(Xt)
    vp = np.empty((C_CONFIGS, len(yv)), np.float32)
    tp = np.empty((C_CONFIGS, len(yt)), np.float32)
    crng = np.random.default_rng(seed + 123)
    for c in range(C_CONFIGS):
        m = FACTORY[family](crng)
        m.fit(Xtr, ytr)
        vp[c] = m.predict_proba(Xv)[:, 1]
        tp[c] = m.predict_proba(Xt)[:, 1]
    nt = len(yt); nv = len(yv); npos_t = int(yt.sum()); prevalence = y.sum() / len(y)

    Kset = {}
    for r in RATIOS:
        Kt = int(np.clip(round(npos_t / r), 5, nt - 1)); Kset[Kt] = min(Kset.get(Kt, r), r)
    info = {}
    for Kt in Kset:
        tq = E.precision_at_k(tp, yt, Kt)
        info[Kt] = dict(tq=tq, oracle=float(tq.max()), meancfg=float(tq.mean()),
                        Kv=int(np.clip(round(Kt * nv / nt), 3, nv - 1)))
    sel = {Kt: {m: [] for m in SEL} for Kt in Kset}
    npool = len(yv)
    for _ in range(R_BOOT):
        bi = rng.integers(0, npool, size=npool); S = vp[:, bi]; ys = yv[bi]
        ap = E.ap_vec(S, ys); auc = E.auc_vec(S, ys); ll = -E.logloss_vec(S, ys)
        for Kt in Kset:
            Kv = info[Kt]["Kv"]; tq = info[Kt]["tq"]; pk = E.precision_at_k(S, ys, Kv)
            mv = {"P@K": pk, "AP": ap, "AUC": auc, "logloss": ll}
            for mm in SEL:
                sel[Kt][mm].append(float(tq[int(np.argmax(mv[mm]))]))
    out = []
    for Kt in Kset:
        it = info[Kt]; oracle = it["oracle"]; denom = max(oracle - it["meancfg"], 1e-6)
        rec = dict(family=family, name=name, K=Kt, n=nt, n_pos=npos_t, prevalence=prevalence,
                   npos_over_K=npos_t / Kt, oracle=oracle, meancfg=it["meancfg"],
                   pos_in_budget=oracle * Kt)
        for mm in SEL:
            a = np.array(sel[Kt][mm]); rec[f"nregret_{mm}"] = float((oracle - a.mean()) / denom)
        out.append(rec)
    return out

if __name__ == "__main__":
    t0 = time.time(); results = []
    # cache datasets once
    datasets = [(name, src, X, y) for name, src, X, y in E.load_all()]
    print(f"loaded {len(datasets)} datasets")
    for family in ["logreg", "rf"]:
        ndat = 0
        for name, src, X, y in datasets:
            try:
                rows = run_one(family, name, X, y)
            except Exception as e:
                print(f"  [{family}] ERROR {name}: {type(e).__name__} {str(e)[:50]}"); continue
            if not rows:
                continue
            for r in rows: r["source"] = src
            results += rows; ndat += 1
        print(f"[{family}] done: {ndat} datasets, {time.time()-t0:.0f}s elapsed")
    json.dump(results, open(f"{OUT}/results_families.json", "w"), indent=2)
    print(f"\n{len(results)} rows -> results_families.json ({time.time()-t0:.0f}s)")
