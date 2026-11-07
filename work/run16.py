import sys; sys.path.insert(0,'work')
from contextlib import redirect_stdout; import io
with redirect_stdout(io.StringIO()):
    import combine as cb
from harness import score, T
import pandas as pd, numpy as np, math
eps_ids=sorted(T.episode_id.unique()); half=set(eps_ids[::2])
h1=lambda e: e.ep.isin(half); h2=lambda e: ~e.ep.isin(half)
v2=pd.read_csv('submissions/v2/validation_predictions.csv').set_index('query_id')
v3=pd.read_csv('submissions/v3/validation_predictions.csv').set_index('query_id')
base=pd.read_csv('winner-blend/validation_predictions.csv').set_index('query_id')
for name,p in [('orig blend',base),('v2',v2),('v3',v3)]:
    print(name); score(p,h1); score(p,h2)
def combine_eps(eps,fw):
    out={}
    for qid,rec in cb.comp.items():
        pf=[]
        for f in ['2','4b']:
            l=[math.log(0.01)+rec['null'][0]];p=[rec['null'][1]]
            for pr,v in cb.PR.items(): l.append(rec[(f,)+pr][0]+math.log(v)); p.append(rec[(f,)+pr][1])
            l=np.array(l); w=np.exp(l-l.max()); w/=w.sum(); pf.append((w*np.array(p)).sum())
        pw=min(max(fw*pf[0]+(1-fw)*pf[1],eps),1-eps); out[qid]=[1-pw]+list(pw*rec['size'])
    return out
print('eps/fw grid (half1 | half2)')
for eps in [0.0005,0.002,0.005]:
    for fw in [0.3,0.5,0.7]:
        p=combine_eps(eps,fw); a=score(p,h1,False)[0]; b=score(p,h2,False)[0]; print(eps,fw,round(a,3),round(b,3))
