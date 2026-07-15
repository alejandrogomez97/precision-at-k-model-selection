"""K-sweep meta-analysis: is n_pos/K (positives per slot) the deciding axis,
or does the absolute count of positives in the budget still govern?"""
import json, numpy as np
from scipy.stats import spearmanr, wilcoxon
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="/home/agomez/proyectos/precision-at-k-study"
COL={"P@K":"#c0392b","AP":"#27ae60","AUC":"#2471a3","logloss":"#7d3c98"}
SEL=["P@K","AP","AUC","logloss"]; M3=["P@K","AP","logloss"]
R=json.load(open(f"{OUT}/results_real3.json"))
info=[r for r in R if (r["oracle"]-r["meancfg"])>=0.02]
ndat=len(set(r["name"] for r in R))
print(f"{ndat} datasets, {len(R)} rows, {len(info)} informative")
def c(rows,k): return np.array([r[k] for r in rows],float)

# ---- correlations ----
mean_reg=(c(info,"nregret_P@K")+c(info,"nregret_AP")+c(info,"nregret_logloss"))/3
pen=c(info,"nregret_P@K")-np.minimum(c(info,"nregret_AP"),c(info,"nregret_logloss"))
print("\nSpearman vs MEAN regret (selection difficulty):")
for nm,k in [("n_pos/K (reader's ratio)","npos_over_K"),("positives IN budget (count)","pos_in_budget"),
             ("K","K"),("n_pos","n_pos"),("prevalence","prevalence"),("K/n","K_over_n")]:
    print(f"  {nm:32s} rho={spearmanr(c(info,k),mean_reg).correlation:+.3f}")
print("Spearman vs P@K penalty:")
for nm,k in [("n_pos/K","npos_over_K"),("positives IN budget","pos_in_budget")]:
    print(f"  {nm:32s} rho={spearmanr(c(info,k),pen).correlation:+.3f}")

# ---- paired Wilcoxon signed-rank tests on normalized regret (lower = better) ----
print("\nPaired Wilcoxon signed-rank on normalized regret (n=%d):"%len(info))
sig={}
def wtest(a,b):
    _,p=wilcoxon(c(info,f"nregret_{a}"),c(info,f"nregret_{b}")); return float(p)
for a,b in [("P@K","logloss"),("P@K","AP"),("AP","logloss"),("P@K","AUC")]:
    p=wtest(a,b); sig[f"{a}_vs_{b}"]=p
    print(f"  {a:7s} vs {b:7s}: mean regret {c(info,f'nregret_{a}').mean():.3f} vs "
          f"{c(info,f'nregret_{b}').mean():.3f}  |  p={p:.2e}")

# ===== FIG 0: overall regret boxplot =====
fig,ax=plt.subplots(figsize=(9,5))
data=[[r[f"nregret_{m}"] for r in info] for m in SEL]
bp=ax.boxplot(data,positions=range(len(SEL)),widths=.62,patch_artist=True,showfliers=False)
for p,m in zip(bp["boxes"],SEL): p.set_facecolor(COL[m]); p.set_alpha(.55)
for md in bp["medians"]: md.set_color("black")
rng=np.random.default_rng(0)
for i,m in enumerate(SEL):
    ys=[r[f"nregret_{m}"] for r in info]; ax.scatter(i+rng.uniform(-.16,.16,len(ys)),ys,s=8,color=COL[m],edgecolor="white",linewidth=.3,zorder=3)
ax.plot(range(len(SEL)),[np.mean([r[f"nregret_{m}"] for r in info]) for m in SEL],"D",color="black",ms=7,zorder=5,label="mean")
ax.axhline(0,ls="--",color="gray",lw=1,label="oracle (0 = perfect)")
ax.set_xticks(range(len(SEL))); ax.set_xticklabels(SEL); ax.set_ylim(-.05,1.7)
ax.set_ylabel("normalized regret (0 = oracle, 1 = average model)"); ax.set_xlabel("metric used to select the model")
ax.set_title(f"Model-selection regret across {ndat} datasets × up to 7 budgets ({len(info)} cases)\n"
             "log-loss and average precision tie for best; precision@K trails",fontsize=12)
ax.grid(alpha=.25,axis="y"); ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(f"{OUT}/figT0_overall.png",dpi=130,bbox_inches="tight"); plt.close(fig); print("saved figT0_overall.png")

# ===== FIG 5: optimism vs count =====
fig,ax=plt.subplots(figsize=(9,5.2))
xx=c(R,"pos_in_budget"); yy=100*c(R,"optimism_PK_rel")
ax.scatter(xx,yy,s=22,color="#c0392b",edgecolor="white",linewidth=.3); ax.set_xscale("log"); ax.axhline(0,color="#888",lw=1)
lo=np.polyfit(np.log10(xx),yy,1); xr=np.logspace(np.log10(xx.min()),np.log10(xx.max()),50)
ax.plot(xr,lo[0]*np.log10(xr)+lo[1],"--",color="#7d3c98",lw=2,label="trend")
ax.set_xlabel("positives inside the budget (count, log scale)"); ax.set_ylabel("optimism of validation precision@K (%)")
ax.set_title("Reported validation precision@K is inflated when few positives sit in the budget",fontsize=12)
ax.grid(alpha=.25,which="both"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figT5_optimism.png",dpi=130,bbox_inches="tight"); plt.close(fig); print("saved figT5_optimism.png")

def binned(rows,key,metrics,nb=6,logx=True):
    x=c(rows,key); e=np.unique(np.quantile(x,np.linspace(0,1,nb+1)))
    cen=[];ser={m:[] for m in metrics};ns=[]
    for b in range(len(e)-1):
        lo,hi=e[b],e[b+1]; mask=(x>=lo)&(x<hi if b<len(e)-2 else x<=hi)
        sub=[rows[i] for i in range(len(rows)) if mask[i]]
        if len(sub)<5: continue
        cen.append(np.median(x[mask])); ns.append(len(sub))
        for m in metrics: ser[m].append(np.mean([r[f"nregret_{m}"] for r in sub]))
    return cen,ser,ns

# ===== FIG 1: regret per metric vs n_pos/K (reader's axis) =====
cen,ser,ns=binned(info,"npos_over_K",M3)
fig,ax=plt.subplots(figsize=(9,5.2)); xs=range(len(cen))
for m in M3: ax.plot(xs,ser[m],"o-",color=COL[m],lw=2.2,ms=7,label=m)
ax.set_xticks(list(xs)); ax.set_xticklabels([f"{v:.2g}\n(n={n})" for v,n in zip(cen,ns)])
ax.set_xlabel("n_pos / K   =   positive rate ÷ reviewable rate   (positives per inspection slot)")
ax.set_ylabel("mean normalized regret (lower better)")
ax.set_title("Does the ratio n_pos/K decide the metric? Only loosely.\n"
             "log-loss / AP lead across the range; precision@K closes in only at the extremes",fontsize=12)
ax.grid(alpha=.25); ax.legend(title="selection metric")
fig.tight_layout(); fig.savefig(f"{OUT}/figT1_ratio.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("saved figT1_ratio.png")

# ===== FIG 2: the decisive test — ratio vs count, side by side =====
fig,(a1,a2)=plt.subplots(1,2,figsize=(12.5,5.2),sharey=True)
c1,s1,n1=binned(info,"npos_over_K",M3)
c2,s2,n2=binned(info,"pos_in_budget",M3)
for ax,cc,ss,nn,xl,ttl,rho in [
    (a1,c1,s1,n1,"n_pos / K  (reader's ratio)","Ratio axis (scale-free)","npos_over_K"),
    (a2,c2,s2,n2,"positives inside the budget (count)","Count axis (scale-dependent)","pos_in_budget")]:
    xs=range(len(cc))
    for m in M3: ax.plot(xs,ss[m],"o-",color=COL[m],lw=2.2,ms=7,label=m)
    ax.set_xticks(list(xs)); ax.set_xticklabels([f"{v:.2g}\n(n={n})" for v,n in zip(cc,nn)])
    rr=spearmanr(c(info,rho),mean_reg).correlation
    ax.set_title(f"{ttl}\n|ρ| with regret = {abs(rr):.2f}",fontsize=11); ax.set_xlabel(xl); ax.grid(alpha=.25)
a1.set_ylabel("mean normalized regret (lower better)"); a2.legend(title="metric",loc="upper right")
fig.text(0.5,-0.02,"each point = mean regret over its bin;  n = number of cases (dataset × budget K) in the bin",
         ha="center",fontsize=8.5,color="#666")
fig.suptitle("The ratio does NOT decide it — the absolute count of positives in the budget does",fontsize=12.5,y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/figT2_ratio_vs_count.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("saved figT2_ratio_vs_count.png")

# ===== FIG 3: controlled — hold ratio ~1, vary absolute count =====
band=[r for r in info if 0.5<=r["npos_over_K"]<=2.0]
cb,sb,nb2=binned(band,"pos_in_budget",M3,nb=4)
fig,ax=plt.subplots(figsize=(9,5.2)); xs=range(len(cb))
for m in M3: ax.plot(xs,sb[m],"o-",color=COL[m],lw=2.2,ms=7,label=m)
ax.set_xticks(list(xs)); ax.set_xticklabels([f"~{v:.0f}\n(n={n})" for v,n in zip(cb,nb2)])
ax.set_xlabel("positives inside the budget (count)")
ax.set_ylabel("mean normalized regret (lower better)")
ax.set_title("Proof the ratio isn't enough: fixing n_pos/K ≈ 1 (budget ≈ positives),\n"
             f"regret still falls as the absolute count grows  ({len(band)} cases)",fontsize=12)
ax.grid(alpha=.25); ax.legend(title="metric")
fig.tight_layout(); fig.savefig(f"{OUT}/figT3_controlled.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("saved figT3_controlled.png")

# ===== FIG 4: win-rate by n_pos/K regime =====
reg=[("< 0.5\n(budget > positives)",lambda r:r["npos_over_K"]<0.5),
     ("0.5 – 2",lambda r:0.5<=r["npos_over_K"]<2),
     ("2 – 8",lambda r:2<=r["npos_over_K"]<8),
     ("≥ 8\n(tiny budget)",lambda r:r["npos_over_K"]>=8)]
fig,ax=plt.subplots(figsize=(9.5,5.2)); w=0.2; xb=np.arange(len(reg))
for i,m in enumerate(SEL):
    vals=[]
    for _,f in reg:
        sub=[r for r in info if f(r)]
        wins=sum(1 for r in sub if min(SEL,key=lambda mm:r[f"nregret_{mm}"])==m)
        vals.append(100*wins/max(len(sub),1))
    ax.bar(xb+(i-1.5)*w,vals,w,color=COL[m],label=m,alpha=.85)
counts=[sum(1 for r in info if f(r)) for _,f in reg]
ax.set_xticks(xb); ax.set_xticklabels([f"{l}\n(n={c2})" for (l,_),c2 in zip(reg,counts)])
ax.set_xlabel("n_pos / K   (positives per inspection slot)")
ax.set_ylabel("% of cases where the metric selected the best model")
ax.set_title("Win-rate by the ratio n_pos/K — no clean regime where precision@K rules\n"
             "log-loss and AP stay ahead almost everywhere",fontsize=12)
ax.grid(alpha=.25,axis="y"); ax.legend(ncol=4,loc="upper center")
fig.tight_layout(); fig.savefig(f"{OUT}/figT4_winrate_ratio.png",dpi=130,bbox_inches="tight"); plt.close(fig)
print("saved figT4_winrate_ratio.png")

summ={"ndat":ndat,"nrows":len(R),"ninfo":len(info),
      "rho_meanreg_ratio":float(spearmanr(c(info,"npos_over_K"),mean_reg).correlation),
      "rho_meanreg_count":float(spearmanr(c(info,"pos_in_budget"),mean_reg).correlation),
      "means":{m:float(np.mean(c(info,f"nregret_{m}"))) for m in SEL},
      "wins":{m:int(sum(1 for r in info if min(SEL,key=lambda mm:r[f"nregret_{mm}"])==m)) for m in SEL},
      "wilcoxon_p":sig}
json.dump(summ,open(f"{OUT}/meta3_summary.json","w"),indent=2)
print("\n",json.dumps(summ,indent=2))
