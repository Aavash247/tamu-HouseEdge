import numpy as np, itertools, math
PAIRS=list(itertools.combinations(range(1,9),2))
SIZE_PRIOR={'steady':[.62,.30,.075,.002],'balanced':[.40,.42,.15,.025],'jackpot':[.12,.33,.41,.14]}

def machine_arrays(ser,cu):
    ticks=np.array(sorted(t for t in ser if t<cu)); pay=np.array([ser[t] for t in ticks])
    win=(pay>0).astype(float); pos={t:k for k,t in enumerate(ticks)}
    L=np.full((len(ticks),9),np.nan)
    for k in range(1,9):
        for idx,t in enumerate(ticks):
            p=pos.get(t-k)
            if p is not None: L[idx,k]=win[p]
    qlag=[(win[pos[cu-k]] if (cu-k) in pos else np.nan) for k in range(9)]
    return ticks,pay,win,L,qlag

def run_bocpd(cells, ys, gaps, ncell, qcell, hazard, a, m0):
    """cells: int cell index per obs; returns (log evidence, predictive P(win) for query cell).
    Run-length posterior over segments; each segment has independent Beta(a*m0,a*(1-m0)) per cell."""
    # state arrays over run lengths: weights logR, counts n[r,c], s[r,c]
    logR=np.array([0.0]); N=np.zeros((1,ncell)); S=np.zeros((1,ncell)); ev=0.0
    for c,y,g in zip(cells,ys,gaps):
        # changepoint transition over gap
        hz=1-(1-hazard)**g
        R=np.exp(logR-logR.max()); R/=R.sum()
        cp=R.sum()*hz; R=R*(1-hz)
        R=np.concatenate([[cp],R]); N=np.vstack([np.zeros((1,ncell)),N]); S=np.vstack([np.zeros((1,ncell)),S])
        p=(S[:,c]+a*m0)/(N[:,c]+a)
        lik=p if y else 1-p
        tot=(R*lik).sum(); ev+=math.log(tot)
        R=R*lik/tot
        N[:,c]+=1; S[:,c]+=y
        keep=R>1e-10
        R=R[keep];N=N[keep];S=S[keep]
        logR=np.log(R)
    R=np.exp(logR-logR.max()); R/=R.sum()
    if qcell is None:
        p=(S.sum(1)+a*m0)/(N.sum(1)+a)
    else:
        p=(S[:,qcell]+a*m0)/(N[:,qcell]+a)
    return ev,float((R*p).sum())

def predict(q,H,hazard=1/150,a=2.0,temp=1.0,pair_prior=None,size_hl=1e9,size_b=5.0,eps=0.003,size_prior=SIZE_PRIOR,pairs=None):
    pairs=pairs or PAIRS
    out={}
    for r in q.itertuples():
        ticks,pay,win,L,qlag=machine_arrays(H[r.machine_id],r.cutoff); cu=r.cutoff
        m0=float(np.clip(win.mean(),0.05,0.95))
        gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
        logs=[];preds=[]
        ev,p=run_bocpd(np.zeros(len(win),int),win,gaps,1,0,hazard,a,m0)
        logs.append(temp*ev+math.log(pair_prior.get('null',1.0) if pair_prior else 1.0)); preds.append(p)
        for (i,j) in pairs:
            m=~np.isnan(L[:,i])&~np.isnan(L[:,j])
            cells=(L[m,i]==L[m,j]).astype(int); tk=ticks[m]
            g=np.diff(np.concatenate([[tk[0]-1],tk])) if len(tk) else []
            qc=None if (np.isnan(qlag[i]) or np.isnan(qlag[j])) else int(qlag[i]==qlag[j])
            ev,p=run_bocpd(cells,win[m],g,2,qc,hazard,a,m0)
            # add evidence of unusable rows under null-ish (approx: ignore; same across pairs mostly)
            lp=math.log(pair_prior.get((i,j),1e-3)) if pair_prior else 0.0
            logs.append(temp*ev+lp); preds.append(p)
        logs=np.array(logs); pw=np.exp(logs-logs.max()); pw/=pw.sum()
        pwin=float(np.clip((pw*np.array(preds)).sum(),eps,1-eps))
        wm=win>0; sw=2.0**(-(cu-1-ticks[wm])/size_hl); sp=pay[wm]
        cnt=np.array([sw[sp==v].sum() for v in (1,2,5,20)])
        pr=np.array(size_prior[r.machine_type]); sz=(cnt+size_b*pr)/(cnt.sum()+size_b)
        out[r.query_id]=[1-pwin]+list(pwin*sz)
    return out
