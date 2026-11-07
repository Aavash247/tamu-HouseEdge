import numpy as np, math
def fb_orientation(qs,qd,cells,ys,gaps,rho):
    """forward-backward for one (qs,qd) point; returns smoothed P(sigma=0) per obs and filtered P(sigma=0) for next tick."""
    n=len(cells); fw=np.zeros(n); b=0.5; L0=np.zeros(n); L1=np.zeros(n); F=np.zeros(n)
    for t,(c,y,g) in enumerate(zip(cells,ys,gaps)):
        f=(1-(1-2*rho)**g)/2; F[t]=f
        b=b*(1-f)+(1-b)*f
        p0=qs if c==1 else qd; p1=qd if c==1 else qs
        l0=p0 if y else 1-p0; l1=p1 if y else 1-p1; L0[t]=l0; L1[t]=l1
        b=b*l0/(b*l0+(1-b)*l1); fw[t]=b
    nxt=b*(1-rho)+(1-b)*rho
    # backward
    sm=np.zeros(n); sm[-1]=fw[-1] if n else 0
    be0,be1=1.0,1.0
    for t in range(n-1,-1,-1):
        s0=fw[t]*be0; s1=(1-fw[t])*be1; sm[t]=s0/(s0+s1)
        # propagate backward beta to t-1: beta_{t-1}(s)=sum_s' T(s->s') L_t(s') beta_t(s')
        f=F[t]; a0=L0[t]*be0; a1=L1[t]*be1
        nb0=(1-f)*a0+f*a1; nb1=f*a0+(1-f)*a1; z=nb0+nb1; be0,be1=nb0/z,nb1/z
    return sm,nxt
def refine(grid,post_w,A,B,ys,gaps,qa,qb,rho,strength=4.0,flip='b'):
    cells=(A==B).astype(int)
    k=int(np.argmax(post_w)); qs=grid.q_same[k]; qd=grid.q_diff[k]
    # use posterior-mean rates as anchor
    qs_m=(post_w*grid.q_same).sum(); qd_m=(post_w*grid.q_diff).sum()
    sm,nxt=fb_orientation(qs,qd,cells,ys,gaps,rho)
    # aligned counts in orientation-0 frame
    N=np.zeros((2,2)); S=np.zeros((2,2))
    for a,b_,y,p0 in zip(A.astype(int),B.astype(int),ys,sm):
        if flip=='b': a1,b1=a,1-b_
        else: a1,b1=1-a,b_
        N[a,b_]+=p0; S[a,b_]+=p0*y; N[a1,b1]+=1-p0; S[a1,b1]+=(1-p0)*y
    prior=np.array([[qs,qd],[qd,qs]])
    R=(S+strength*prior)/(N+strength)
    if qa is None: return None
    qa=int(qa); qb=int(qb)
    if flip=='b': alt=R[qa,1-qb]
    else: alt=R[1-qa,qb]
    return float(nxt*R[qa,qb]+(1-nxt)*alt)
