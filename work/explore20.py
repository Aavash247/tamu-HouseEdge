import sys; sys.path.insert(0,'work')
import pandas as pd, numpy as np, json
from harness import T,COLS,C,VQ
cfg=json.load(open('entry-v2/model.json')); prof=cfg['size_profiles']
p=pd.read_csv('work/v3/validation_predictions.csv').set_index('query_id').loc[T.index]
typ=VQ.set_index('query_id').machine_type.loc[T.index]
sz=T[COLS[1:]].div(1-T.p_0,axis=0)
st=np.array([int(abs(s[0]-prof[t][1][0])<abs(s[0]-prof[t][0][0])) for s,t in zip(sz.values,typ)])
res=(1-T.p_0)-(1-p.p_0)
from scipy.special import logit
lres=logit((1-T.p_0).clip(1e-4,1-1e-4))-logit((1-p.p_0).clip(1e-4,1-1e-4))
print('corr resid vs st', np.corrcoef(res,st)[0,1].round(3), 'logit', np.corrcoef(lres,st)[0,1].round(3))
for t in ['steady','balanced','jackpot']:
    m=(typ==t).values; print(t, np.corrcoef(res[m],st[m])[0,1].round(3), np.corrcoef(lres[m],st[m])[0,1].round(3))
