import pandas as pd, numpy as np
P='public/'
q=pd.read_csv(P+'validation/queries.csv'); t=pd.read_csv(P+'validation/targets.csv'); c=pd.read_csv(P+'validation/conditions.csv')
h=pd.read_csv(P+'validation/history.csv')
d=q.merge(t,on=['query_id','episode_id']).merge(c,on='query_id')
cols=['p_0','p_1','p_2','p_5','p_20']
print(d.groupby('condition')[cols].mean().round(3))
print(d.groupby('machine_type')[cols].mean().round(3))
print(d.groupby(['machine_type','signal'])[cols].mean().round(3))
print('p0 distribution quantiles', d.p_0.quantile([.05,.25,.5,.75,.95]).round(3).tolist())
# history count per machine & ticks
g=h.groupby('machine_id').tick.agg(['count','min','max'])
d=d.merge(g,left_on='machine_id',right_index=True)
print(d.groupby('condition')[['count','cutoff','max']].describe().round(1).T)
# conditional win rate given lag1 in history
h=h.sort_values(['machine_id','tick'])
h['win']=(h.payout>0).astype(int)
for lag in range(1,9):
    h[f'l{lag}']=h.groupby('machine_id').win.shift(lag)
print(h[[f'l{k}' for k in range(1,9)]+['win']].corr()['win'].round(3))
# per machine lag-1 effect variety
def eff(x):
    x=x.dropna(subset=['l1'])
    a=x[x.l1==1].win.mean(); b=x[x.l1==0].win.mean(); return a-b
e=h.groupby('machine_id').apply(eff)
print('per machine lag1 effect quantiles',e.quantile([.05,.25,.5,.75,.95]).round(3).tolist())
