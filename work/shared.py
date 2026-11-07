from exp import *
import harness
from scipy.special import logsumexp
pt=make_prior(train_posts(0.001),lambda q:q['machine_type'])
def machine_tau_ll(q,hist,rho=0.001):
    ticks,pay,win,lags,qlag=he.machine_arrays(hist[(q['episode_id'],q['machine_id'])],q['cutoff'])
    best=None
    for (i,j),prior in zip(cfg['pairs'],cfg['pair_prior']):
        m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
        ev,p,w=he.run_flip(grid,cells,win[m],g,qc,rho,logprior=pt[q['machine_type']],return_post=True)
        if best is None or ev>best[0]: best=(ev,i,j,m,cells,qc,w)
    ev,i,j,m,cells,qc,w=best
    k=int(np.argmax(w)); qs,qd=grid.q_same[k],grid.q_diff[k]
    y=win[m]; tk=ticks[m]
    p0=np.where(cells==1,qs,qd); p1=np.where(cells==1,qd,qs)
    l0=np.where(y==1,np.log(p0),np.log(1-p0)); l1=np.where(y==1,np.log(p1),np.log(1-p1))
    T_=q['cutoff']; taus=np.arange(0,T_+1)  # tau=T_ means no change before cutoff
    c0=np.concatenate([[0],np.cumsum(l0)]); c1=np.concatenate([[0],np.cumsum(l1)])
    nb=np.searchsorted(tk,taus)  # number of obs before tau
    A=c0[nb]+(c1[-1]-c1[nb])   # sigma0=0, flip at tau -> after: sigma=1
    B=c1[nb]+(c0[-1]-c0[nb])
    LL=np.logaddexp(A,B)-math.log(2)
    # P(sigma_now=0 | tau): if tau<T_: current = flipped of sigma0
    post_s0=np.exp(A-np.logaddexp(A,B))  # P(sigma0=0|tau)
    s_now0=np.where(taus<T_,1-post_s0,post_s0)
    if qc is None: pq=np.full(len(taus),0.5*(qs+qd))
    else: pq=s_now0*(qs if qc==1 else qd)+(1-s_now0)*(qd if qc==1 else qs)
    return taus,LL,pq,(i,j)
if __name__=='__main__':
    for ep in sorted({q['episode_id'] for q in vq})[:40]:
        qs_=[q for q in vq if q['episode_id']==ep]; cond=harness.C[qs_[0]['query_id']]
        if cond!='shared_change' and cond!='drift': continue
        res=[machine_tau_ll(q,vh) for q in qs_]
        taus=res[0][0]
        tot=sum(r[1] for r in res)
        am=[int(taus[np.argmax(r[1])]) for r in res]
        print(cond[:5],ep[-3:],qs_[0]['cutoff'],'joint tau',int(taus[np.argmax(tot)]),'indiv',am)
