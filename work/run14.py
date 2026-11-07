from run13 import *
import pickle, flip4
pt=make_prior(train_posts(0.001),lambda q:q['machine_type'])
pri_b=pickle.load(open('work/pri4b.pkl','rb'))
pri_a=train_prior4(0.001,'a'); pickle.dump(pri_a,open('work/pri4a.pkl','wb'))
comp={}
for q in vq:
    ticks,pay,win,lags,qlag=he.machine_arrays(vh[(q['episode_id'],q['machine_id'])],q['cutoff'])
    m0=float(np.clip(win.mean(),0.05,0.95)); gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
    rec={'null':he.run_null(win,gaps,cfg['null_hazard'],cfg['null_a'],m0)}
    for (i,j) in cfg['pairs']:
        m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
        rec[('2',i,j)]=he.run_flip(grid,cells,win[m],g,qc,0.001,logprior=pt[q['machine_type']])
        m,A,B,g,qa,qb=p4inputs(ticks,lags,qlag,i,j)
        rec[('4b',i,j)]=flip4.run_flip4(G4,A,B,win[m],g,qa,qb,0.001,logprior=pri_b[q['machine_type']],flip='b')
        rec[('4a',i,j)]=flip4.run_flip4(G4,A,B,win[m],g,qa,qb,0.001,logprior=pri_a[q['machine_type']],flip='a')
    mw=pay>0; sz=he.size_distribution(pay[mw],q['cutoff']-1-ticks[mw],cfg['size_profiles'][q['machine_type']],cfg['size_half_life'])
    rec['size']=np.asarray(sz); comp[q['query_id']]=rec
pickle.dump(comp,open('work/comp_val.pkl','wb'))
