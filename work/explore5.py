import sys; sys.path.insert(0,'work')
from harness import *; import pairmodel as pm
q=VQ.merge(C.rename('cond'),left_on='query_id',right_index=True)
for cond in ['shared_change','drift']:
  print('=====',cond)
  for ep,g in list(q[q.cond==cond].groupby('episode_id'))[:4]:
    print('--',ep, g.cutoff.iloc[0])
    for r in g.itertuples():
        ticks,pay,win,L,qlag=pm.machine_arrays(VH[r.machine_id],r.cutoff)
        segs=[]
        for lo in range(0,r.cutoff,40):
            m=(ticks>=lo)&(ticks<lo+40)
            res=[]
            for (i,j) in [(1,3),(2,5),(4,7)]:
                mm=m&~np.isnan(L[:,i])&~np.isnan(L[:,j]); same=L[mm,i]==L[mm,j]
                res.append(f"{win[mm][same].mean():.1f}/{win[mm][~same].mean():.1f}")
            sz=pay[m&(win>0)]; segs.append(" ".join(res)+f" s{np.mean(sz):.1f}")
        print(r.machine_type[:4],' | '.join(segs))
