import pandas as pd, numpy as np, itertools
P='public/'
q=pd.read_csv(P+'validation/queries.csv'); t=pd.read_csv(P+'validation/targets.csv'); c=pd.read_csv(P+'validation/conditions.csv')
h=pd.read_csv(P+'validation/history.csv')
d=q.merge(t,on=['query_id','episode_id']).merge(c,on='query_id')
d=d[d.condition=='stable']
H={m:dict(zip(g.tick,g.payout)) for m,g in h.groupby('machine_id')}
for _,r in d.head(24).iterrows():
    ser=H[r.machine_id]; cu=r.cutoff
    ts=[x for x in ser if x>=8]
    Y=np.array([ser[x]>0 for x in ts],float)
    out=[]
    for i,j in [(1,3),(2,5),(4,7)]:
        A=np.array([ser[x-i]>0 for x in ts]); B=np.array([ser[x-j]>0 for x in ts])
        rates=[Y[(A==a)&(B==b)].mean() for a in (0,1) for b in (0,1)]
        out.append(f"({i},{j}):"+" ".join(f"{x:.2f}" for x in rates))
    print(r.machine_type[:4], r.signal, f"p0={r.p_0:.3f}", " | ".join(out))
