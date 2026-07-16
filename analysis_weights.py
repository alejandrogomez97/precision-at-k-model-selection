"""Learn a weighting of validation metrics that minimises TEST regret, and try
cascade strategies. Weights are fit on test -> optimistic; read them as an
approximate IMPORTANCE ranking, not a trustworthy regret."""
import json, itertools, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/home/agomez/proyectos/precision-at-k-study"
METRICS = ["PK", "AP", "AUC", "NLL"]          # all oriented so higher = better
NICE = {"PK": "precision@k", "AP": "avg precision", "AUC": "ROC-AUC", "NLL": "log-loss"}
COL = {"PK": "#c0392b", "AP": "#27ae60", "AUC": "#2471a3", "NLL": "#7d3c98"}
FAMS = ["lightgbm", "rf", "logreg"]
LAB = {"lightgbm": "LightGBM", "rf": "Random Forest", "logreg": "Logistic Regression"}

R = json.load(open(f"{OUT}/results_weights.json"))

def prep(case):
    Z = np.array([case["val"][m] for m in METRICS], float).T   # (C,4)
    mu = Z.mean(0); sd = Z.std(0); sd[sd == 0] = 1.0
    Zz = (Z - mu) / sd
    tq = np.array(case["tq"], float)
    denom = max(case["oracle"] - case["meancfg"], 1e-6)
    return Zz, tq, case["oracle"], denom

def regret_for_selector(cases, select_fn):
    r = []
    for Zz, tq, oracle, denom in cases:
        c = select_fn(Zz, tq)
        r.append((oracle - tq[c]) / denom)
    return float(np.mean(r))

def by_weight(w):
    return lambda Zz, tq: int(np.argmax(Zz @ w))

# weight grid on the simplex, step 0.1
grid = [np.array(w) / 10 for w in itertools.product(range(11), repeat=4) if sum(w) == 10]

summary = {}
for fam in FAMS + ["pooled"]:
    rows = R if fam == "pooled" else [c for c in R if c["family"] == fam]
    cases = [prep(c) for c in rows]
    # pure-metric baselines
    pure = {m: regret_for_selector(cases, by_weight(np.eye(4)[i])) for i, m in enumerate(METRICS)}
    equal = regret_for_selector(cases, by_weight(np.ones(4) / 4))
    # best weights on the grid
    best_w, best_r = None, 1e9
    for w in grid:
        r = regret_for_selector(cases, by_weight(w))
        if r < best_r:
            best_r, best_w = r, w
    summary[fam] = dict(n=len(cases), pure=pure, equal=equal,
                        best_w={m: float(best_w[i]) for i, m in enumerate(METRICS)},
                        best_r=best_r)
    print(f"\n=== {fam} (n={len(cases)}) ===")
    print("  pure:", {NICE[m]: round(pure[m], 3) for m in METRICS}, " equal:", round(equal, 3))
    print("  best weights:", {NICE[m]: round(float(best_w[i]), 2) for i, m in enumerate(METRICS)},
          " -> regret", round(best_r, 3))

# ---- cascade: among top-n by AP (validation), pick best validation P@K ----
def cascade_APthenPK(n):
    def sel(Zz, tq):
        ap = Zz[:, METRICS.index("AP")]; pk = Zz[:, METRICS.index("PK")]
        top = np.argsort(-ap)[:n]
        return int(top[np.argmax(pk[top])])
    return sel

cascade = {}
for fam in FAMS + ["pooled"]:
    rows = R if fam == "pooled" else [c for c in R if c["family"] == fam]
    cases = [prep(c) for c in rows]
    cascade[fam] = {n: regret_for_selector(cases, cascade_APthenPK(n)) for n in [1, 2, 3, 5, 10, 30]}
print("\ncascade (top-n by AP, then best P@K) mean regret:")
for fam in FAMS + ["pooled"]:
    print(f"  {fam:9s}", {n: round(v, 3) for n, v in cascade[fam].items()})

# ---- figure 1: best weights per family ----
fig, ax = plt.subplots(figsize=(9.5, 5)); w = 0.2; xb = np.arange(len(FAMS) + 1)
labels = FAMS + ["pooled"]
for i, m in enumerate(METRICS):
    vals = [100 * summary[f]["best_w"][m] for f in labels]
    ax.bar(xb + (i - 1.5) * w, vals, w, color=COL[m], label=NICE[m], alpha=0.85)
ax.set_xticks(xb); ax.set_xticklabels([LAB.get(f, "Pooled") for f in labels])
ax.set_ylabel("weight that minimises test regret (%)")
ax.set_title("A blend beats any single metric — but the exact weights are NOT identifiable\n"
             "(the four metrics are correlated and the optimum is flat; fit on test). Read only: don't rely on precision@k alone.", fontsize=10)
ax.grid(alpha=0.25, axis="y"); ax.legend(ncol=4, loc="upper center")
fig.tight_layout(); fig.savefig(f"{OUT}/figW_weights.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("\nsaved figW_weights.png")

# ---- figure 2: cascade curve ----
fig, ax = plt.subplots(figsize=(8.5, 5))
ns = [1, 2, 3, 5, 10, 30]
for fam in FAMS:
    ax.plot(range(len(ns)), [cascade[fam][n] for n in ns], "o-", lw=2, label=LAB[fam])
ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"top-{n}\nby AP" for n in ns])
ax.set_xlabel("keep the best validation precision@k among the top-n configs by validation AP")
ax.set_ylabel("mean normalized regret (test)")
ax.set_title("Cascade: AP to shortlist, precision@k to break ties\n"
             "n=1 is pure AP; large n lets precision@k decide (ties broken by AP)", fontsize=11)
ax.grid(alpha=0.25); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figX_cascade.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("saved figX_cascade.png")

json.dump({"weights": summary, "cascade": cascade}, open(f"{OUT}/weights_summary.json", "w"), indent=2)
print("saved weights_summary.json")
