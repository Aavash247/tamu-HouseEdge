import sys; sys.path.insert(0,'work')
from harness import *; import bocpd as bp, sizemodel as sm
pp={(1,3):1.0,(2,5):1.0,(4,7):1.0,'null':0.3}; P3=[(1,3),(2,5),(4,7)]
base=pd.DataFrame(bp.predict(VQ,VH,hazard=1/400,pair_prior=pp,pairs=P3)).T; base.columns=COLS
pw=1-base.p_0
def with_size(fn):
    out={}
    for r in VQ.itertuples():
        ticks,pay,win,L,qlag=bp.machine_arrays(VH[r.machine_id],r.cutoff)
        m=pay>0; sz=fn(pay[m],r.machine_type,r.cutoff-1-ticks[m])
        out[r.query_id]=[base.p_0[r.query_id]]+list(pw[r.query_id]*sz)
    return out
score(base)
print('iid',end=' ');score(with_size(lambda p,t,a: sm.iid_mix(p,t)))
for hl in [30,60,120]:
    print('iid hl',hl,end=' ');score(with_size(lambda p,t,a: sm.iid_mix(p,t,a,hl=hl)))
for rho in [0.01,0.03,0.1,0.2]:
    print('hmm',rho,end=' ');score(with_size(lambda p,t,a: sm.hmm(p,t,rho)))
