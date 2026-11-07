import sys; sys.path.insert(0,'work')
from harness import *
from explore7 import d
g=d.groupby('episode_id').st.agg(['mean','count'])
print(np.histogram(g['mean'],bins=[0,.01,.2,.4,.6,.8,.99,1.01])[0])
# compare to binomial expectation
print('overall',d.st.mean())
# cross-machine lag features: neighbor wins at lags
rows=[]
for ep,gg in d.reset_index().groupby('episode_id'):
    for r in gg.itertuples():
        f={}
        for o in gg.itertuples():
            if o.machine_id==r.machine_id: continue
            ser=VH[o.machine_id]
            for k in range(1,9):
                f.setdefault(k,[]).append(ser.get(r.cutoff-k,np.nan))
        rows.append(dict(st=r.st,typ=r.machine_type,**{f'nw{k}':np.nanmean(np.array(v)>0) for k,v in f.items()},**{f'np{k}':np.nanmean(v) for k,v in f.items()}))
X=pd.DataFrame(rows)
print(X.drop(columns='typ').corr()['st'].round(3).sort_values())
