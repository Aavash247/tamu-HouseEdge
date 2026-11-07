from exp import *
import flip4, time
G4=flip4.Grid4(); print('K',G4.K)
def p4inputs(ticks,lags,qlag,i,j):
    m=~np.isnan(lags[:,i])&~np.isnan(lags[:,j]); tk=ticks[m]
    g=np.diff(np.concatenate([[tk[0]-1],tk]))
    return m,lags[m,i],lags[m,j],g,qlag[i],qlag[j]
def swapsym(w,flip):
    # posterior symmetric under orientation swap: R -> R[:,perm]
    return w  # handled via averaging below
def train_prior4(rho,flip,mix=0.9):
    d=defaultdict(list); t=time.time()
    lev_n=len(flip4.LEV); idx=np.arange(G4.K).reshape([lev_n]*4)
    perm_axes=[1,0,3,2] if flip=='b' else [2,3,0,1]
    swap=np.transpose(idx,perm_axes).ravel()
    for q in tq:
        ticks,pay,win,lags,qlag=he.machine_arrays(th[(q['episode_id'],q['machine_id'])],q['cutoff'])
        best=None
        for (i,j) in cfg['pairs']:
            m,A,B,g,qa,qb=p4inputs(ticks,lags,qlag,i,j)
            ev,p,w=flip4.run_flip4(G4,A,B,win[m],g,qa,qb,rho,flip=flip,return_post=True)
            if best is None or ev>best[0]: best=(ev,w)
        w=best[1]; w=0.5*(w+w[swap]); d[q['machine_type']].append(w)
    print('prior time',time.time()-t)
    return {k:np.log(mix*np.mean(v,0)+(1-mix)/G4.K) for k,v in d.items()}
def run4(pri,rho=0.001,flip='b',queries=None,hist=None):
    queries=queries or vq; hist=hist or vh; preds={}
    for q in queries:
        ticks,pay,win,lags,qlag=he.machine_arrays(hist[(q['episode_id'],q['machine_id'])],q['cutoff'])
        m0=float(np.clip(win.mean(),0.05,0.95)); gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
        logs=[];ps=[]
        ev,p=he.run_null(win,gaps,cfg['null_hazard'],cfg['null_a'],m0); logs.append(ev+math.log(cfg['null_prior'])); ps.append(p)
        for (i,j),prior in zip(cfg['pairs'],cfg['pair_prior']):
            m,A,B,g,qa,qb=p4inputs(ticks,lags,qlag,i,j)
            ev,p=flip4.run_flip4(G4,A,B,win[m],g,qa,qb,rho,logprior=pri[q['machine_type']] if pri else None,flip=flip)
            logs.append(ev+math.log(prior)); ps.append(p)
        logs=np.array(logs); w=np.exp(logs-logs.max()); w/=w.sum(); pwin=float(np.clip((w*np.array(ps)).sum(),cfg['eps'],1-cfg['eps']))
        mw=pay>0; sz=he.size_distribution(pay[mw],q['cutoff']-1-ticks[mw],cfg['size_profiles'][q['machine_type']],cfg['size_half_life'])
        preds[q['query_id']]=[1-pwin]+list(pwin*np.asarray(sz))
    return preds
if __name__=='__main__':
    t=time.time(); print('4cell flat b',end=' '); score(run4(None)); print(time.time()-t)
    pri=train_prior4(0.001,'b')
    import pickle; pickle.dump(pri,open('work/pri4b.pkl','wb'))
    print('4cell prior b',end=' '); score(run4(pri))
