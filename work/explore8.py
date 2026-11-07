import sys,itertools; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
from explore7 import PROF,IDX,d
from sklearn.metrics import roc_auc_score
res={N:[] for N in [3,5,10,20,40,1000]}; st=[]; cond=[]
for r in d.reset_index().itertuples():
    ticks,pay,win,L,qlag=bp.machine_arrays(VH[r.machine_id],r.cutoff)
    w=win>0; P=pay[w]
    llr=np.log(np.array([PROF[r.machine_type][1][IDX[p]] for p in P]))-np.log(np.array([PROF[r.machine_type][0][IDX[p]] for p in P]))
    for N in res: res[N].append(llr[-N:].sum() if len(llr) else 0)
    st.append(r.st); cond.append(r.cond)
st=np.array(st); cond=np.array(cond)
for N,v in res.items():
    v=np.array(v); print(N, round(roc_auc_score(st,v),3), {c:round(roc_auc_score(st[cond==c],v[cond==c]),3) for c in set(cond)})
