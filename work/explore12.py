from exp import *
import pandas as pd
from sklearn.metrics import roc_auc_score
prof=cfg['size_profiles']
rows=[]
for q in vq:
    t=T.loc[q['query_id']]; s=np.array([t.p_1,t.p_2,t.p_5,t.p_20]); s/=s.sum()
    st=int(np.abs(s-np.array(prof[q['machine_type']][1])).sum()<np.abs(s-np.array(prof[q['machine_type']][0])).sum())
    ser=vh[(q['episode_id'],q['machine_id'])]; cu=q['cutoff']
    f=dict(st=st,typ=q['machine_type'],sig=q['signal'],cu=cu)
    for N in [1,2,3,4,6,8,12,16,32,64]:
        vals=[ser[t] for t in range(cu-N,cu) if t in ser]
        f[f'sum{N}']=np.mean(vals) if vals else np.nan
        f[f'win{N}']=np.mean([v>0 for v in vals]) if vals else np.nan
    # episode neighbors same tick features
    nb=[]
    for o in vq:
        if o['episode_id']==q['episode_id'] and o['machine_id']!=q['machine_id']:
            so=vh[(o['episode_id'],o['machine_id'])]
            nb+= [so[t] for t in range(cu-8,cu) if t in so]
    f['nb8']=np.mean(nb) if nb else np.nan
    f['nbw8']=np.mean([v>0 for v in nb]) if nb else np.nan
    f['tickmod']=cu%2; f['cu']=cu
    rows.append(f)
X=pd.DataFrame(rows)
for c in X.columns:
    if c in ('st','typ'): continue
    m=X[c].notna()
    print(c, round(roc_auc_score(X.st[m],X[c][m]),3))
