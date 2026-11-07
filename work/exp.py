import sys; sys.path.insert(0,'entry-v2'); sys.path.insert(0,'work')
from harness import score, T, COLS
import houseedge as he, numpy as np, math, json
from collections import defaultdict
P='public/'
cfg=json.load(open('entry-v2/model.json')); grid=he.FlipGrid(cfg['grid_size'])
tq=he.read_queries(P+'train/queries.csv'); th=he.read_history(P+'train/history.csv')
vq=he.read_queries(P+'validation/queries.csv'); vh=he.read_history(P+'validation/history.csv')
def train_posts(rho=None):
    rho=rho or cfg['rho']; out=[]
    for q in tq:
        ticks,pay,win,lags,qlag=he.machine_arrays(th[(q['episode_id'],q['machine_id'])],q['cutoff'])
        best=None
        for (i,j) in cfg['pairs']:
            m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
            ev,p,w=he.run_flip(grid,cells,win[m],g,qc,rho,return_post=True)
            if best is None or ev>best[0]: best=(ev,w)
        w=best[1]; w=0.5*(w+w.reshape(grid_n(),grid_n()).T.ravel()); out.append((q,w))
    return out
def grid_n(): return int(round(math.sqrt(len(grid.q_same))))
def make_prior(posts,keyfn,mix=0.9):
    d=defaultdict(list)
    for q,w in posts: d[keyfn(q)].append(w)
    K=len(grid.q_same)
    return {k:np.log(mix*np.mean(v,0)+(1-mix)/K) for k,v in d.items()}
def run(prior_fn=lambda q:None, queries=None, hist=None, rho=None, null_prior=None, size_fn=None, eps=None, pairs=None):
    queries=queries or vq; hist=hist or vh; rho=rho or cfg['rho']; null_prior=null_prior or cfg['null_prior']; eps=eps or cfg['eps']
    pairs=pairs or list(zip(cfg['pairs'],cfg['pair_prior']))
    preds={}
    for q in queries:
        ticks,pay,win,lags,qlag=he.machine_arrays(hist[(q['episode_id'],q['machine_id'])],q['cutoff'])
        m0=float(np.clip(win.mean(),0.05,0.95)); gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
        logs=[];ps=[]
        ev,p=he.run_null(win,gaps,cfg['null_hazard'],cfg['null_a'],m0); logs.append(ev+math.log(null_prior)); ps.append(p)
        lp=prior_fn(q)
        for (i,j),prior in pairs:
            m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
            ev,p=he.run_flip(grid,cells,win[m],g,qc,rho,logprior=lp); logs.append(ev+math.log(prior)); ps.append(p)
        logs=np.array(logs); w=np.exp(logs-logs.max()); w/=w.sum(); pwin=float(np.clip((w*np.array(ps)).sum(),eps,1-eps))
        mw=pay>0
        if size_fn: sz=size_fn(q,pay[mw],q['cutoff']-1-ticks[mw])
        else: sz=he.size_distribution(pay[mw],q['cutoff']-1-ticks[mw],cfg['size_profiles'][q['machine_type']],cfg['size_half_life'])
        preds[q['query_id']]=[1-pwin]+list(pwin*np.asarray(sz))
    return preds
