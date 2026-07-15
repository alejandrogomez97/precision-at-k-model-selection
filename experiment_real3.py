"""
K-sweep study. For each dataset, fix the model bank and sweep the budget K to
hit target ratios  r = n_pos / K  (positives per inspection slot) in
{0.25, 0.5, 1, 2, 4, 8, 16}.  This lets us test the reader's proposed axis
(prevalence / reviewable-rate = n_pos/K) directly, and check whether that ratio
alone decides which selection metric to use — or whether the absolute count of
positives that land in the budget still matters (scale-dependence).
"""
import json, warnings, time
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.datasets import make_classification, fetch_openml
from sklearn.model_selection import train_test_split
from imblearn.datasets import fetch_datasets
import lightgbm as lgb

OUT="/home/agomez/proyectos/precision-at-k-study"; CACHE=f"{OUT}/data_cache"
C_CONFIGS=40; R_BOOT=120; MAX_N=30000
RATIOS=[0.25,0.5,1,2,4,8,16]           # target n_pos / K
N_OPENML_TARGET=70
SEL=["P@K","AP","AUC","logloss"]

def precision_at_k(S,y,k):
    C,n=S.shape; k=int(min(max(k,1),n))
    idx=np.argpartition(-S,kth=k-1,axis=1)[:,:k]; return y[idx].mean(1)
def auc_vec(S,y):
    C,n=S.shape; order=np.argsort(S,axis=1); ranks=np.empty_like(order,np.float64)
    ar=np.arange(1,n+1,dtype=np.float64); np.put_along_axis(ranks,order,np.broadcast_to(ar,(C,n)),1)
    pos=y.astype(bool); npos=pos.sum(); nneg=n-npos
    if npos==0 or nneg==0: return np.full(C,0.5)
    return (ranks[:,pos].sum(1)-npos*(npos+1)/2.0)/(npos*nneg)
def ap_vec(S,y):
    C,n=S.shape; order=np.argsort(-S,axis=1); ys=y[order]
    prec=np.cumsum(ys,axis=1)/np.arange(1,n+1,dtype=np.float64); npos=y.sum()
    return np.zeros(C) if npos==0 else (prec*ys).sum(1)/npos
def logloss_vec(P,y,eps=1e-7):
    P=np.clip(P,eps,1-eps); return -(y*np.log(P)+(1-y)*np.log(1-P)).mean(1)

def clean_Xy(X,y):
    if hasattr(y,"values"): y=y.values
    y=np.asarray(y).ravel().astype(str)
    classes=pd.unique(y[y!="nan"])
    if len(classes)!=2: return None,None
    vals,cnts=np.unique(y,return_counts=True); minority=vals[np.argmin(cnts)]
    yb=(y==minority).astype(np.float64)
    if isinstance(X,pd.DataFrame):
        Xn=np.empty((len(X),X.shape[1]),np.float32)
        for j,c in enumerate(X.columns):
            s=X[c]; Xn[:,j]=(s.astype(np.float32) if s.dtype.kind in "biufc"
                             else pd.factorize(s,use_na_sentinel=True)[0].astype(np.float32))
        X=Xn
    else: X=np.asarray(X,np.float32)
    keep=np.nanstd(X,0)>0
    if keep.sum()>=2: X=X[:,keep]
    return X,yb

def load_all():
    seen=set()
    def sig(X,y): return (len(y),int(y.sum()),X.shape[1])
    for name,b in fetch_datasets(data_home=CACHE).items():
        X,y=clean_Xy(pd.DataFrame(b.data),b.target)
        if X is None: continue
        s=sig(X,y)
        if s in seen: continue
        seen.add(s); yield name,"imblearn",X,y
    cand=pd.read_csv(f"{OUT}/openml_candidates.csv"); cand=cand[cand.NumberOfInstances<=100000].sort_values("prev")
    got=0
    for _,row in cand.iterrows():
        if got>=N_OPENML_TARGET: break
        did=int(row.did); nm=str(row["name"])[:40]
        try:
            d=fetch_openml(data_id=did,as_frame=True,cache=True,data_home=f"{CACHE}/openml")
            X,y=clean_Xy(d.data,d.target)
            if X is None or len(y)<800 or y.sum()<40 or (len(y)-y.sum())<40: continue
            s=sig(X,y)
            if s in seen: continue
            seen.add(s); got+=1; yield nm,"openml",X,y
        except Exception as e:
            print(f"  skip {nm}: {type(e).__name__}"); continue
    for nm,prev,sep in [("synth_2pct",0.02,1.0),("synth_0.5pct",0.005,0.8),("synth_5pct",0.05,1.2)]:
        X,y=make_classification(n_samples=20000,n_features=30,n_informative=8,n_redundant=6,
            n_clusters_per_class=3,weights=[1-prev],flip_y=0.01,class_sep=sep,random_state=7)
        yield nm,"synthetic",X.astype(np.float32),y.astype(np.float64)

def random_config(rng):
    return dict(objective="binary",n_estimators=int(rng.integers(60,180)),
        learning_rate=float(10**rng.uniform(-2.0,-0.6)),num_leaves=int(rng.integers(8,80)),
        max_depth=int(rng.choice([-1,3,4,5,6])),min_child_samples=int(rng.integers(5,80)),
        subsample=float(rng.uniform(0.6,1.0)),subsample_freq=1,colsample_bytree=float(rng.uniform(0.5,1.0)),
        reg_lambda=float(10**rng.uniform(-3,1)),reg_alpha=float(10**rng.uniform(-3,1)),
        n_jobs=2,verbosity=-1,random_state=int(rng.integers(1,1e6)))

def run_one(name,X,y,seed=0):
    rng=np.random.default_rng(seed); n=len(y)
    if n>MAX_N:
        idx,_=train_test_split(np.arange(n),train_size=MAX_N,stratify=y,random_state=seed); X,y=X[idx],y[idx]
    Xtr,Xtmp,ytr,ytmp=train_test_split(X,y,train_size=0.5,stratify=y,random_state=seed)
    Xv,Xt,yv,yt=train_test_split(Xtmp,ytmp,train_size=0.5,stratify=ytmp,random_state=seed)
    if yv.sum()<12 or yt.sum()<12: return []
    vp=np.empty((C_CONFIGS,len(yv)),np.float32); tp=np.empty((C_CONFIGS,len(yt)),np.float32)
    crng=np.random.default_rng(seed+123)
    for c in range(C_CONFIGS):
        m=lgb.LGBMClassifier(**random_config(crng)); m.fit(Xtr,ytr)
        vp[c]=m.predict_proba(Xv)[:,1]; tp[c]=m.predict_proba(Xt)[:,1]
    nt=len(yt); nv=len(yv); npos_t=int(yt.sum()); prevalence=y.sum()/len(y)

    # build the set of test-K values from target ratios, dedup
    Kset={}
    for r in RATIOS:
        Kt=int(np.clip(round(npos_t/r),5,nt-1))
        Kset[Kt]=min(Kset.get(Kt,r),r)     # remember a representative ratio
    # precompute targets per K
    info={}
    for Kt in Kset:
        tq=precision_at_k(tp,yt,Kt)
        info[Kt]=dict(tq=tq,oracle=float(tq.max()),meancfg=float(tq.mean()),
                      Kv=int(np.clip(round(Kt*nv/nt),3,nv-1)))
    sel={Kt:{m:[] for m in SEL} for Kt in Kset}
    optv={Kt:[] for Kt in Kset}; optt={Kt:[] for Kt in Kset}
    npool=len(yv)
    for _ in range(R_BOOT):
        bi=rng.integers(0,npool,size=npool); S=vp[:,bi]; ys=yv[bi]
        ap=ap_vec(S,ys); auc=auc_vec(S,ys); ll=-logloss_vec(S,ys)
        for Kt in Kset:
            Kv=info[Kt]["Kv"]; tq=info[Kt]["tq"]; pk=precision_at_k(S,ys,Kv)
            mv={"P@K":pk,"AP":ap,"AUC":auc,"logloss":ll}
            for mm in SEL: sel[Kt][mm].append(float(tq[int(np.argmax(mv[mm]))]))
            bp=int(np.argmax(pk)); optv[Kt].append(float(pk[bp])); optt[Kt].append(float(tq[bp]))
    out=[]
    for Kt in Kset:
        it=info[Kt]; oracle=it["oracle"]; denom=max(oracle-it["meancfg"],1e-6)
        rec=dict(name=name,K=Kt,n=nt,n_pos=npos_t,prevalence=prevalence,
                 npos_over_K=npos_t/Kt, K_over_n=Kt/nt,
                 oracle=oracle,meancfg=it["meancfg"],pos_in_budget=oracle*Kt,
                 optimism_PK_rel=float((np.mean(optv[Kt])-np.mean(optt[Kt]))/max(np.mean(optt[Kt]),1e-6)))
        for mm in SEL:
            a=np.array(sel[Kt][mm]); rec[f"nregret_{mm}"]=float((oracle-a.mean())/denom)
        out.append(rec)
    return out

if __name__=="__main__":
    t0=time.time(); results=[]; ndat=0
    for name,src,X,y in load_all():
        try: rows=run_one(name,X,y)
        except Exception as e:
            print(f"  ERROR {name}: {type(e).__name__} {str(e)[:50]}"); continue
        if not rows: print(f"  drop {name}"); continue
        for r in rows: r["source"]=src
        results+=rows; ndat+=1
        print(f"[{ndat:3d}] {name:26s} prev={rows[0]['prevalence']:.3f} "
              f"Ks={sorted(set(r['K'] for r in rows))}")
    json.dump(results,open(f"{OUT}/results_real3.json","w"),indent=2)
    print(f"\n{ndat} datasets, {len(results)} rows, {time.time()-t0:.0f}s -> results_real3.json")
