import pandas as pd, numpy as np, itertools
P='public/'
q=pd.read_csv(P+'validation/queries.csv'); t=pd.read_csv(P+'validation/targets.csv'); c=pd.read_csv(P+'validation/conditions.csv')
h=pd.read_csv(P+'validation/history.csv')
d=q.merge(t,on=['query_id','episode_id']).merge(c,on='query_id')
H={m:dict(zip(g.tick,g.payout)) for m,g in h.groupby('machine_id')}
rows=[]
for _,r in d.iterrows():
    ser=H[r.machine_id]; cu=r.cutoff
    ts=[x for x in ser if all((x-k) in ser for k in range(1,9))]
    if len(ts)<20: continue
    Y=np.array([ser[x]>0 for x in ts],float)
    X=np.array([[ser[x-k]>0 for k in range(1,9)] for x in ts],float)
    xq=np.array([ser.get(cu-k,-1)>0 for k in range(1,9)],float)
    best=None
    for i,j in itertools.combinations(range(8),2):
        cell=X[:,i]*2+X[:,j]; ll=0; rates={}
        for v in range(4):
            m=cell==v; n=m.sum(); s=Y[m].sum(); p=(s+.5)/(n+1); rates[v]=p
            ll+= s*np.log(p)+(n-s)*np.log(1-p)
        if best is None or ll>best[0]: best=(ll,i,j,rates)
    ll,i,j,rates=best
    pred=rates[int(xq[i]*2+xq[j])]
    # single lag best
    rows.append(dict(cond=r.condition,typ=r.machine_type,true=1-r.p_0,pair=pred,freq=Y.mean(),i=i+1,j=j+1,ll=ll/len(Y)))
R=pd.DataFrame(rows)
print(len(R))
print('MSE pair',((R.true-R.pair)**2).mean(),'freq',((R.true-R.freq)**2).mean())
print(R.groupby('cond').apply(lambda x: pd.Series(dict(pair=((x.true-x.pair)**2).mean(),freq=((x.true-x.freq)**2).mean()))))
print(R[['i','j']].value_counts().head(15))
print(R.head(30).round(3))
