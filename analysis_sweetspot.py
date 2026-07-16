"""Where does the TEST-best model sit in each VALIDATION metric's ranking?
Diagnostic that explains which metric surfaces the winner, and a search for a
better combined selection rule (rank-sum / rank-min / cascade) than any single metric."""
import json, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT = "/home/agomez/proyectos/precision-at-k-study"
METRICS = ["PK", "AP", "AUC", "NLL"]
NICE = {"PK": "precision@k", "AP": "avg precision", "AUC": "ROC-AUC", "NLL": "log-loss"}
COL = {"PK": "#c0392b", "AP": "#27ae60", "AUC": "#2471a3", "NLL": "#7d3c98"}
FAMS = ["lightgbm", "rf", "logreg"]
LAB = {"lightgbm": "LightGBM", "rf": "Random Forest", "logreg": "Logistic Regression"}
R = json.load(open(f"{OUT}/results_weights.json"))

def midrank_pct(v, idx):
    """percentile (0-100) of v[idx] within v, ties -> mid-rank (100 = top)."""
    v = np.asarray(v); less = np.sum(v < v[idx]); eq = np.sum(v == v[idx])
    return 100.0 * (less + 0.5 * eq) / len(v)

# ---- 1) percentile of the TEST oracle (and top-3) in each validation metric ----
pct = {m: [] for m in METRICS}                 # oracle
pct3 = {m: [] for m in METRICS}                # mean over top-3 test models
per_fam = {f: {m: [] for m in METRICS} for f in FAMS}
for c in R:
    tq = np.array(c["tq"]); order = np.argsort(-tq)
    o = int(order[0]); top3 = order[:3]
    for m in METRICS:
        v = c["val"][m]
        pct[m].append(midrank_pct(v, o))
        pct3[m].append(np.mean([midrank_pct(v, i) for i in top3]))
        per_fam[c["family"]][m].append(midrank_pct(v, o))

print("Mean validation-percentile of the TEST-best model (higher = metric surfaces the winner):")
for m in METRICS:
    print(f"  {NICE[m]:14s} oracle {np.mean(pct[m]):5.1f}  (median {np.median(pct[m]):5.1f})   top-3 {np.mean(pct3[m]):5.1f}")

# ---- figure: boxplot of oracle percentile per metric ----
fig, ax = plt.subplots(figsize=(9, 5))
data = [pct[m] for m in METRICS]
bp = ax.boxplot(data, positions=range(len(METRICS)), widths=.6, patch_artist=True, showfliers=False)
for p, m in zip(bp["boxes"], METRICS): p.set_facecolor(COL[m]); p.set_alpha(.55)
for md in bp["medians"]: md.set_color("black")
rng = np.random.default_rng(0)
for i, m in enumerate(METRICS):
    ax.scatter(i + rng.uniform(-.16, .16, len(pct[m])), pct[m], s=6, color=COL[m], edgecolor="white", linewidth=.2, zorder=3)
ax.plot(range(len(METRICS)), [np.mean(pct[m]) for m in METRICS], "D", color="black", ms=7, zorder=5, label="mean")
ax.axhline(100, ls="--", color="gray", lw=1, label="top of the validation ranking")
ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([NICE[m] for m in METRICS])
ax.set_ylabel("percentile of the test-best model\nin the validation-metric ranking (100 = top)")
ax.set_title("Where does the model that WINS on test sit in each validation ranking?\n"
             "Only mid-to-upper, and widely spread, in EVERY metric — no metric reliably surfaces the winner "
             "(AP/AUC medians ~68 vs log-loss ~58)", fontsize=10)
ax.grid(alpha=.25, axis="y"); ax.legend(loc="lower left")
fig.text(0.5, -0.02, "Each dot = one (dataset, budget) case; the config with the best TEST precision@k, and its percentile in each VALIDATION metric. "
         "Higher & tighter = that validation metric reliably ranks the true winner near the top.", ha="center", fontsize=8, color="#666")
fig.tight_layout(); fig.savefig(f"{OUT}/figY_oracle_percentile.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print("saved figY_oracle_percentile.png")

# ---- 2) search for a better combined selection rule ----
def prep(c):
    Z = {m: np.array(c["val"][m], float) for m in METRICS}
    # percentile-rank of every config per metric (0-1, ties mid-rank)
    Pr = {}
    for m in METRICS:
        v = Z[m]; order = np.argsort(v)
        r = np.empty(len(v));
        # average rank
        tmp = v.argsort(); ranks = np.empty_like(tmp, float); ranks[tmp] = np.arange(len(v))
        # simple: percentile via midrank
        Pr[m] = np.array([ (np.sum(v < v[i]) + 0.5*np.sum(v==v[i]))/len(v) for i in range(len(v)) ])
    tq = np.array(c["tq"], float); d = max(c["oracle"] - c["meancfg"], 1e-6)
    return Z, Pr, tq, c["oracle"], d

cases = {f: [prep(c) for c in R if c["family"] == f] for f in FAMS}
cases["pooled"] = [prep(c) for c in R]

def regret(caselist, selfn):
    return float(np.mean([(o - tq[selfn(Z, Pr, tq)]) / d for Z, Pr, tq, o, d in caselist]))

rules = {
    "pure AP":            lambda Z, Pr, tq: int(np.argmax(Z["AP"])),
    "pure P@k":           lambda Z, Pr, tq: int(np.argmax(Z["PK"])),
    "pure log-loss":      lambda Z, Pr, tq: int(np.argmax(Z["NLL"])),
    "cascade AP->P@k(3)": None,  # handled below
    "rank-sum AP+P@k":    lambda Z, Pr, tq: int(np.argmax(Pr["AP"] + Pr["PK"])),
    "rank-min(AP,P@k)":   lambda Z, Pr, tq: int(np.argmax(np.minimum(Pr["AP"], Pr["PK"]))),
    "rank-sum AP+P@k+LL": lambda Z, Pr, tq: int(np.argmax(Pr["AP"] + Pr["PK"] + Pr["NLL"])),
    "rank-sum all-4":     lambda Z, Pr, tq: int(np.argmax(Pr["AP"] + Pr["PK"] + Pr["AUC"] + Pr["NLL"])),
}
def cascade3(Z, Pr, tq):
    top = np.argsort(-Z["AP"])[:3]; return int(top[np.argmax(Z["PK"][top])])

print("\nCombined selection rules (mean normalized regret; lower better):")
hdr = "  {:22s}".format("rule") + "".join(f"{f:>10s}" for f in FAMS + ["pooled"])
print(hdr)
summary = {}
for name, fn in rules.items():
    f = cascade3 if name.startswith("cascade") else fn
    row = {fam: regret(cases[fam], f) for fam in FAMS + ["pooled"]}
    summary[name] = row
    print("  {:22s}".format(name) + "".join(f"{row[fam]:10.3f}" for fam in FAMS + ["pooled"]))

json.dump({"oracle_pct": {m: float(np.mean(pct[m])) for m in METRICS},
           "rules": summary}, open(f"{OUT}/sweetspot_summary.json", "w"), indent=2)
print("\nsaved sweetspot_summary.json")
