import numpy as np, itertools, math
from scipy.special import betaln
PAIRS=list(itertools.combinations(range(1,9),2))
SIZE_PRIOR={'steady':[.62,.30,.075,.002],'balanced':[.40,.42,.15,.025],'jackpot':[.12,.33,.41,.14]}
def bml(s,n,a,b): return betaln(s+a*b,n-s+a*(1-b))-betaln(a*b,a*(1-b))
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
def pair_post(win,L,w,a,temp,pair_prior,base):
    logs=[];stats=[]
    n=w.sum(); s=(w*win).sum()
    logs.append(temp*bml(s,n,a,base)+math.log(pair_prior.get('null',1.0) if pair_prior else 1.0)); stats.append(None)
    for (i,j) in PAIRS:
        m=~np.isnan(L[:,i])&~np.isnan(L[:,j]); same=np.zeros(len(win),bool); same[m]=L[m,i]==L[m,j]
        ll=0
        for mm in (m&same, m&~same, ~m):
            ll+=bml((w[mm]*win[mm]).sum(), w[mm].sum(), a, base)
        lp=math.log(pair_prior.get((i,j),1e-3)) if pair_prior else 0.0
        logs.append(temp*ll+lp); stats.append((i,j,m,same))
    logs=np.array(logs); pw=np.exp(logs-logs.max()); return pw/pw.sum(), stats
def predict(q,H,hl=40,hl_sel=None,a=2.0,temp=1.0,pair_prior=None,size_hl=1e9,size_b=5.0,eps=0.003,size_prior=SIZE_PRIOR,ret_pwin=False):
    out={}
    for r in q.itertuples():
        ticks,pay,win,L,qlag=machine_arrays(H[r.machine_id],r.cutoff); cu=r.cutoff
        w=2.0**(-(cu-1-ticks)/hl); ws=w if hl_sel is None else 2.0**(-(cu-1-ticks)/hl_sel)
        base=float(np.clip((w*win).sum()/w.sum(),0.02,0.98)); bsel=float(np.clip((ws*win).sum()/ws.sum(),0.02,0.98))
        pw,stats=pair_post(win,L,ws,a,temp,pair_prior,bsel)
        preds=[]
        for st in stats:
            if st is None: preds.append(((w*win).sum()+a*base)/(w.sum()+a)); continue
            i,j,m,same=st
            if np.isnan(qlag[i]) or np.isnan(qlag[j]): preds.append(base); continue
            mm=m&(same if qlag[i]==qlag[j] else ~same)
            preds.append(((w[mm]*win[mm]).sum()+a*base)/(w[mm].sum()+a))
        pwin=float(np.clip((pw*np.array(preds)).sum(),eps,1-eps))
        wm=win>0; sw=2.0**(-(cu-1-ticks[wm])/size_hl); sp=pay[wm]
        cnt=np.array([sw[sp==v].sum() for v in (1,2,5,20)])
        pr=np.array(size_prior[r.machine_type]); sz=(cnt+size_b*pr)/(cnt.sum()+size_b)
        out[r.query_id]=[1-pwin]+list(pwin*sz)
    return out
