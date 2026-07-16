"""figT6: the endogenous count (precision x K) vs the model-free, a-priori
proxy min(K, n_pos) — nearly as predictive and known before training."""
import json, numpy as np
from scipy.stats import spearmanr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/home/agomez/proyectos/precision-at-k-study"
COL={"P@K":"#c0392b","AP":"#27ae60","AUC":"#2471a3","logloss":"#7d3c98"}; M3=["P@K","AP","AUC","logloss"]
R=json.load(open(f"{OUT}/results_real3.json"))
info=[r for r in R if (r["oracle"]-r["meancfg"])>=0.02]
for r in info: r["minKN"]=min(r["K"],r["n_pos"])
def c(rows,k): return np.array([r[k] for r in rows],float)
mean_reg=(c(info,"nregret_P@K")+c(info,"nregret_AP")+c(info,"nregret_logloss"))/3
def binned(key,nb=6):
    x=c(info,key); e=np.unique(np.quantile(x,np.linspace(0,1,nb+1)))
    cen=[];ser={m:[] for m in M3};ns=[]
    for b in range(len(e)-1):
        lo,hi=e[b],e[b+1]; mask=(x>=lo)&(x<hi if b<len(e)-2 else x<=hi)
        sub=[info[i] for i in range(len(info)) if mask[i]]
        if len(sub)<5: continue
        cen.append(np.median(x[mask])); ns.append(len(sub))
        for m in M3: ser[m].append(np.mean([r[f"nregret_{m}"] for r in sub]))
    return cen,ser,ns
fig,(a1,a2)=plt.subplots(1,2,figsize=(12.5,5.2),sharey=True)
for ax,key,ttl in [
    (a1,"pos_in_budget","Endogenous: positives the model achieves\n(≈ precision × K — needs a trained model)"),
    (a2,"minKN","A-priori & model-free: min(K, n_pos)\n(the most positives that COULD land in the budget)")]:
    cen,ser,ns=binned(key); xs=range(len(cen))
    for m in M3: ax.plot(xs,ser[m],"o-",color=COL[m],lw=2.2,ms=7,label=m)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"≈{v:.0f}\nn={n}" for v,n in zip(cen,ns)])
    rr=abs(spearmanr(c(info,key),mean_reg).correlation)
    ax.set_title(f"{ttl}\n|ρ| with regret = {rr:.2f}",fontsize=11)
    ax.set_xlabel(key.replace("pos_in_budget","precision × K (count)").replace("minKN","min(K, n_pos)"))
    ax.grid(alpha=.25)
a1.set_ylabel("mean normalized regret (lower better)"); a2.legend(title="metric",loc="upper right")
fig.text(0.5,-0.05,
         "How to read it (LightGBM):  → moving right = more positives in the budget (left: what the model achieves, precision×K;  right: the a-priori ceiling min(K, n_pos)).  "
         "Each line = a metric to pick the model; lower = better.\n"
         "n under each tick = cases averaged (a \"case\" = one dataset at one budget K). The n's sum to 322, NOT 93: each of the 93 datasets is evaluated at up to 7 budgets K.",
         ha="center",fontsize=8,color="#555")
rho=spearmanr(c(info,"pos_in_budget"),c(info,"minKN")).correlation
fig.suptitle(f"You don't need the model to know if precision@K will be reliable:\n"
             f"the a-priori min(K, n_pos) tracks the true (endogenous) count almost perfectly (ρ = {rho:.2f})",
             fontsize=12.5,y=1.03)
fig.tight_layout(); fig.savefig(f"{OUT}/figT6_apriori.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("saved figT6_apriori.png")
