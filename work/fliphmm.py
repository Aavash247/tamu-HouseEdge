import numpy as np, math
G=np.linspace(0.01,0.99,25)
Q1,Q0=np.meshgrid(G,G,indexing='ij'); Q1=Q1.ravel(); Q0=Q0.ravel()
def run_flip(cells,ys,gaps,qcell,rho,logprior=None):
    """2-state orientation HMM over a grid of (q_same,q_diff). Returns log evidence, predictive P(win)."""
    K=len(Q1)
    b=np.full(K,0.5)  # P(sigma=0) per grid point
    ll=np.zeros(K)
    for c,y,g in zip(cells,ys,gaps):
        f=(1-(1-2*rho)**g)/2
        b=b*(1-f)+(1-b)*f
        p0=Q1 if c==1 else Q0   # sigma=0
        p1=Q0 if c==1 else Q1   # sigma=1 swaps
        l0=p0 if y else 1-p0; l1=p1 if y else 1-p1
        tot=b*l0+(1-b)*l1
        ll+=np.log(tot); b=b*l0/tot
    # advance one tick to the query
    f=rho; b=b*(1-f)+(1-b)*f
    lp=ll+(logprior if logprior is not None else 0)
    mx=lp.max(); w=np.exp(lp-mx); ev=mx+math.log(w.sum())-math.log(K); w/=w.sum()
    if qcell is None: pq=0.5*(Q1+Q0)
    else:
        pq=b*(Q1 if qcell==1 else Q0)+(1-b)*(Q0 if qcell==1 else Q1)
    return ev,float((w*pq).sum())
