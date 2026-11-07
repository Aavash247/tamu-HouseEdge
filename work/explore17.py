import sys; sys.path.insert(0,'work')
import pandas as pd, numpy as np, harness
from harness import T,COLS,C
p=pd.read_csv('work/v3/validation_predictions.csv').set_index('query_id').loc[T.index]
d=pd.DataFrame({'pp':1-p.p_0,'tp':1-T.p_0,'cond':C.loc[T.index]})
d['e']=(d.pp-d.tp)
d['se_total']=((p[COLS].values-T[COLS].values)**2).sum(1)
d['se_win']=2*d.e**2  # approx contribution (p0 and mass)
for c,g in d.groupby('cond'):
    print(c, 'mean se_total',g.se_total.mean().round(4),'win part ~',(g.e**2).mean().round(4),
          'true intermediate frac(0.2<tp<0.8)',((g.tp>0.2)&(g.tp<0.8)).mean().round(2),
          'err on intermediate',(g.e[(g.tp>0.2)&(g.tp<0.8)]**2).mean().round(4),'err extremes',(g.e[(g.tp<=0.2)|(g.tp>=0.8)]**2).mean().round(4))
