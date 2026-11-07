import pandas as pd, numpy as np
P='public/'
q=pd.read_csv(P+'validation/queries.csv'); t=pd.read_csv(P+'validation/targets.csv'); c=pd.read_csv(P+'validation/conditions.csv')
h=pd.read_csv(P+'validation/history.csv')
d=q.merge(t,on=['query_id','episode_id']).merge(c,on='query_id')
H={(m):dict(zip(g.tick,g.payout)) for m,g in h.groupby('machine_id')}
for k in range(1,9):
    d[f'l{k}']=[H[m].get(cu-k,np.nan) for m,cu in zip(d.machine_id,d.cutoff)]
d['logit']=np.log((1-d.p_0)/d.p_0)
for k in range(1,9):
    w=(d[f'l{k}']>0).astype(float); 
    print(k, round(np.corrcoef(w[d[f'l{k}'].notna()], d.logit[d[f'l{k}'].notna()])[0,1],3))
pd.set_option('display.width',200)
print(d[['machine_type','signal','condition','p_0','p_1','p_2','p_5','p_20']+[f'l{k}' for k in range(1,9)]].head(40).round(3).to_string())
