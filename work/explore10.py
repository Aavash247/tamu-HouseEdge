import sys,itertools; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
import contextlib,io
with contextlib.redirect_stdout(io.StringIO()):
    from explore7 import d,PROF,IDX
rows=[]
for r in d.reset_index().itertuples():
    ticks,pay,win,L,qlag=bp.machine_arrays(VH[r.machine_id],r.cutoff)
    ser=VH[r.machine_id]
    pl={k:ser.get(r.cutoff-k,np.nan) for k in range(1,9)}
    rows.append(dict(st=r.st,cond=r.cond,typ=r.machine_type,sig=r.signal,pwin=1-T.p_0[r.query_id],
      **{f'x{i}{j}':float(qlag[i]==qlag[j]) for i,j in [(1,3),(2,5),(4,7)]},
      **{f'g{k}':float(pl[k]>=2) for k in range(1,9)},
      cnt=sum(pl[k]>0 for k in range(1,9)), sm=sum(pl[k] for k in range(1,9) if pl[k]==pl[k])))
X=pd.DataFrame(rows)
print(X.drop(columns=['cond','typ']).corr()['st'].round(3).sort_values())
print(X.groupby(pd.cut(X.pwin,[0,.1,.5,.9,1])).st.mean())
