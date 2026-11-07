import numpy as np, math, itertools
LEV=np.array([0.01,0.04,0.1,0.2,0.35,0.5,0.65,0.8,0.9,0.96,0.99])
class Grid4:
    def __init__(self,lev=LEV):
        g=np.array(list(itertools.product(range(len(lev)),repeat=4)))
        self.R=lev[g]  # K x 4 ; index cell = 2*a+b
        self.K=len(g)
def run_flip4(G,A,B,ys,gaps,qa,qb,rho,logprior=None,flip='b',return_post=False):
    R=G.R; K=G.K
    if flip=='b': perm=[1,0,3,2]   # (a,b)->(a,1-b)
    else: perm=[2,3,0,1]           # (a,b)->(1-a,b)
    bel=np.full(K,0.5); ll=np.zeros(K)
    for a,b,y,g in zip(A,B,ys,gaps):
        f=(1-(1-2*rho)**g)/2; bel=bel*(1-f)+(1-bel)*f
        c=2*int(a)+int(b); p0=R[:,c]; p1=R[:,perm[c]]
        if not y: p0=1-p0; p1=1-p1
        tot=bel*p0+(1-bel)*p1; ll+=np.log(tot); bel=bel*p0/tot
    bel=bel*(1-rho)+(1-bel)*rho
    ll=ll+(logprior if logprior is not None else -math.log(K))
    mx=ll.max(); w=np.exp(ll-mx); ev=mx+math.log(w.sum()); w/=w.sum()
    if qa is None or qb is None or (isinstance(qa,float) and np.isnan(qa)) or (isinstance(qb,float) and np.isnan(qb)):
        pq=R.mean(1)
    else:
        c=2*int(qa)+int(qb); pq=bel*R[:,c]+(1-bel)*R[:,perm[c]]
    p=float((w*pq).sum())
    return (ev,p,w) if return_post else (ev,p)
