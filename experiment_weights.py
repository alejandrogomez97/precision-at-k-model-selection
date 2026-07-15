"""
Capstone: dump per-configuration validation metrics + test quality, so we can
learn a WEIGHTING of validation metrics that minimises test regret, and try
cascade strategies (e.g. best precision@k among the top-n by AP).

Retrains the model banks (LightGBM / RF / LogReg) and stores, per (family,
dataset, budget K): for each config its validation [P@K, AP, AUC, -logloss] and
its true test precision@K, plus oracle and mean-config quality.

NOTE: weights fitted on test are optimistic (they use the test labels); the point
is an approximate *importance ranking* of the metrics, not a trustworthy regret.
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
import lightgbm as lgb
import experiment_real3 as E

OUT = E.OUT
C_CONFIGS = 30
MAX_N = 15000
RATIOS = [0.25, 0.5, 1, 2, 4, 8, 16]

def cfg_lgbm(rng):
    return lgb.LGBMClassifier(**E.random_config(rng))
def cfg_logreg(rng):
    C = float(10 ** rng.uniform(-3, 2)); cw = rng.choice([None, "balanced"])
    return Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("clf", LogisticRegression(C=C, class_weight=cw, max_iter=400))])
def cfg_rf(rng):
    d = int(rng.choice([0, 6, 10, 16]))
    return Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(n_estimators=int(rng.integers(40, 120)),
                        max_depth=(None if d == 0 else d),
                        max_features=rng.choice(["sqrt", "log2", None]),
                        min_samples_leaf=int(rng.integers(1, 20)),
                        class_weight=rng.choice([None, "balanced"]),
                        n_jobs=2, random_state=int(rng.integers(1, 1e6))))])
FACT = {"lightgbm": cfg_lgbm, "rf": cfg_rf, "logreg": cfg_logreg}

def san(X): return np.where(np.isfinite(X), X, np.nan).astype(np.float32)

def run(family, name, X, y, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    if n > MAX_N:
        idx, _ = train_test_split(np.arange(n), train_size=MAX_N, stratify=y, random_state=seed)
        X, y = X[idx], y[idx]
    Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, train_size=0.5, stratify=y, random_state=seed)
    Xv, Xt, yv, yt = train_test_split(Xtmp, ytmp, train_size=0.5, stratify=ytmp, random_state=seed)
    if yv.sum() < 12 or yt.sum() < 12: return []
    Xtr, Xv, Xt = san(Xtr), san(Xv), san(Xt)
    vp = np.empty((C_CONFIGS, len(yv)), np.float32); tp = np.empty((C_CONFIGS, len(yt)), np.float32)
    crng = np.random.default_rng(seed + 123)
    for c in range(C_CONFIGS):
        m = FACT[family](crng); m.fit(Xtr, ytr)
        vp[c] = m.predict_proba(Xv)[:, 1]; tp[c] = m.predict_proba(Xt)[:, 1]
    nt = len(yt); nv = len(yv); npos_t = int(yt.sum())
    ap = E.ap_vec(vp, yv); auc = E.auc_vec(vp, yv); ll = -E.logloss_vec(vp, yv)   # K-independent
    Kset = {}
    for r in RATIOS:
        Kt = int(np.clip(round(npos_t / r), 5, nt - 1)); Kset[Kt] = True
    out = []
    for Kt in Kset:
        Kv = int(np.clip(round(Kt * nv / nt), 3, nv - 1))
        pk = E.precision_at_k(vp, yv, Kv)            # validation P@K per config
        tq = E.precision_at_k(tp, yt, Kt)            # test quality per config
        oracle = float(tq.max()); mean = float(tq.mean())
        if oracle - mean < 0.02: continue
        out.append(dict(family=family, name=name, K=Kt,
                        val=dict(PK=pk.tolist(), AP=ap.tolist(), AUC=auc.tolist(), NLL=ll.tolist()),
                        tq=tq.tolist(), oracle=oracle, meancfg=mean))
    return out

if __name__ == "__main__":
    t0 = time.time()
    datasets = [(nm, src, X, y) for nm, src, X, y in E.load_all()]
    print(f"{len(datasets)} datasets loaded")
    results = []
    for family in ["lightgbm", "logreg", "rf"]:
        nd = 0
        for nm, src, X, y in datasets:
            try:
                rows = run(family, nm, X, y)
            except Exception as e:
                print(f"  [{family}] ERROR {nm}: {type(e).__name__}"); continue
            results += rows; nd += 1 if rows else 0
        print(f"[{family}] {nd} datasets, {time.time()-t0:.0f}s")
    json.dump(results, open(f"{OUT}/results_weights.json", "w"))
    print(f"{len(results)} cases -> results_weights.json ({time.time()-t0:.0f}s)")
