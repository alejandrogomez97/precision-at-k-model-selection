"""Figure 1: precision@k is piecewise-constant / non-differentiable vs smooth log-loss.
Regenerates fig1_step_function.png."""
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

OUT="/home/agomez/proyectos/precision-at-k-study"
rng=np.random.default_rng(0)
n=400; k=20
y=(rng.random(n)<0.15).astype(float)
base=rng.normal(size=n)+1.2*y                  # mildly informative scores
thr=np.sort(base)[-k]                           # k-th largest (the cutoff)
cand=np.where((y==0)&(base<thr))[0]
j=cand[np.argmax(base[cand])]                   # a negative just below the cutoff
deltas=np.linspace(-1.5,3.5,600); prec=[]; ll=[]
for d in deltas:
    s=base.copy(); s[j]=base[j]+d
    top=np.argpartition(-s,k-1)[:k]
    prec.append(y[top].mean())
    p=np.clip(1/(1+np.exp(-(s-s.mean()))),1e-6,1-1e-6)
    ll.append(-(y*np.log(p)+(1-y)*np.log(1-p)).mean())
prec=np.array(prec); ll=np.array(ll)

fig,ax=plt.subplots(1,2,figsize=(11,4.2))
ax[0].plot(deltas,prec,color="#c0392b",lw=2)
ax[0].set_title(f"precision@k as we shift ONE score",fontsize=12)
ax[0].set_xlabel("shift added to a single instance's score"); ax[0].set_ylabel(f"precision@{k}")
ax[0].text(0.03,0.9,"flat almost everywhere\n(gradient = 0)\n\njumps only when the\ntop-k set changes",
           transform=ax[0].transAxes,va="top",fontsize=9,
           bbox=dict(boxstyle="round",fc="#fdf0ee",ec="#c0392b"))
ax[1].plot(deltas,ll,color="#2471a3",lw=2)
ax[1].set_title("log-loss as we shift the same score",fontsize=12)
ax[1].set_xlabel("shift added to a single instance's score"); ax[1].set_ylabel("log-loss")
ax[1].text(0.5,0.1,"smooth: non-zero gradient\neverywhere",transform=ax[1].transAxes,va="bottom",
           fontsize=9,bbox=dict(boxstyle="round",fc="#eaf2f8",ec="#2471a3"))
for a in ax: a.grid(alpha=0.25)
fig.suptitle("Why precision@k cannot be a training loss (piecewise-constant, gradient 0)",fontsize=13,y=1.02)
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_step_function.png",dpi=130,bbox_inches="tight")
plt.close(fig); print("saved fig1_step_function.png")
