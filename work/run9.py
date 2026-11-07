import sys; sys.path.insert(0,'entry-v2'); sys.path.insert(0,'work')
from harness import score, T, COLS
import houseedge as he, numpy as np, math, json
from collections import defaultdict
P='public/'
cfg=json.load(open('entry-v2/model.json')); grid=he.FlipGrid(cfg['grid_size'])
tq=he.read_queries(P+'train/queries.csv'); th=he.read_history(P+'train/history.csv')
vq=he.read_queries(P+'validation/queries.csv'); vh=he.read_history(P+'validation/history.csv')
post=defaultdict(list)
for q in tq:
    ticks,pay,win,lags,qlag=he.machine_arrays(th[(q['episode_id'],q['machine_id'])],q['cutoff'])
    best=None
    for (i,j) in cfg['pairs']:
        m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
        ev,p,w=he.run_flip(grid,cells,win[m],g,qc,cfg['rho'],return_post=True)
        if best is None or ev>best[0]: best=(ev,w)
    w=best[1]; w=0.5*(w+w.reshape(25,25).T.ravel())  # symmetrize over swap
    post[q['machine_type']].append(w); post['all'].append(w)
priors={k:np.mean(v,0) for k,v in post.items()}
def run(mix, by_type=True):
    preds={}
    for q in vq:
        pr=priors[q['machine_type'] if by_type else 'all']; pr=mix*pr+(1-mix)/len(pr); lp=np.log(pr)
        ticks,pay,win,lags,qlag=he.machine_arrays(vh[(q['episode_id'],q['machine_id'])],q['cutoff'])
        m0=float(np.clip(win.mean(),0.05,0.95)); gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
        logs=[];ps=[]
        ev,p=he.run_null(win,gaps,cfg['null_hazard'],cfg['null_a'],m0); logs.append(ev+math.log(cfg['null_prior'])); ps.append(p)
        for (i,j),prior in zip(cfg['pairs'],cfg['pair_prior']):
            m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
            ev,p=he.run_flip(grid,cells,win[m],g,qc,cfg['rho'],logprior=lp); logs.append(ev+math.log(prior)); ps.append(p)
        logs=np.array(logs); w=np.exp(logs-logs.max()); w/=w.sum(); pwin=float(np.clip((w*np.array(ps)).sum(),cfg['eps'],1-cfg['eps']))
        mw=pay>0; sz=he.size_distribution(pay[mw],q['cutoff']-1-ticks[mw],cfg['size_profiles'][q['machine_type']],cfg['size_half_life'])
        preds[q['query_id']]=[1-pwin]+list(pwin*np.asarray(sz))
    return preds
if __name__=='__main__':
    for mix in [0.0,0.5,0.9,1.0]:
        print(mix,end=' '); score(run(mix))
    print('all-type 0.9',end=' '); score(run(0.9,False))
