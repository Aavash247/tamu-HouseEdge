import sys; sys.path.insert(0,'work')
from harness import *
d=VQ.set_index('query_id').loc[T.index]; d['pw']=1-T.p_0; d['cond']=C
for typ,g in d.groupby('machine_type'):
    print(typ, np.histogram(g.pw,bins=[0,.01,.02,.05,.1,.2,.4,.6,.8,.9,.95,.98,.99,1])[0])
    print('  sig', g.groupby('signal').pw.apply(lambda x: np.round(np.sort(x.values)[[0,len(x)//10,len(x)//2,-len(x)//10,-1]],3).tolist()).to_dict())
