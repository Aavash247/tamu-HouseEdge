from exp import *
import four
posts=train_posts(); pt=make_prior(posts,lambda q:q['machine_type'])
def run4(strength=4.0,flip='b',rho=0.001):
    preds={}
    for q in vq:
        ticks,pay,win,lags,qlag=he.machine_arrays(vh[(q['episode_id'],q['machine_id'])],q['cutoff'])
        m0=float(np.clip(win.mean(),0.05,0.95)); gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
        logs=[];ps=[]
        ev,p=he.run_null(win,gaps,cfg['null_hazard'],cfg['null_a'],m0); logs.append(ev+math.log(cfg['null_prior'])); ps.append(p)
        lp=pt[q['machine_type']]
        for (i,j),prior in zip(cfg['pairs'],cfg['pair_prior']):
            m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
            ev,p,w=he.run_flip(grid,cells,win[m],g,qc,rho,logprior=lp,return_post=True)
            if strength is not None and qc is not None:
                p2=four.refine(grid,w,lags[m,i],lags[m,j],win[m],g,qlag[i],qlag[j],rho,strength,flip)
                if p2 is not None: p=p2
            logs.append(ev+math.log(prior)); ps.append(p)
        logs=np.array(logs); w=np.exp(logs-logs.max()); w/=w.sum(); pwin=float(np.clip((w*np.array(ps)).sum(),cfg['eps'],1-cfg['eps']))
        mw=pay>0; sz=he.size_distribution(pay[mw],q['cutoff']-1-ticks[mw],cfg['size_profiles'][q['machine_type']],cfg['size_half_life'])
        preds[q['query_id']]=[1-pwin]+list(pwin*np.asarray(sz))
    return preds
if __name__=='__main__':
    print('none',end=' '); score(run4(None))
    for s in [2.0,5.0,12.0]:
        for fl in ['b']:
            print(s,fl,end=' '); score(run4(s,fl))
