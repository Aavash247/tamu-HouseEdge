from exp import *
import pickle, combine as cb
from contextlib import redirect_stdout
import io
prof=cfg['size_profiles']; IDX={1:0,2:1,5:2,20:3}
SG=np.linspace(0.005,0.995,100)
def post_pi(pays,typ,ages,hl,logprior=None):
    A=np.array(prof[typ][0]);B=np.array(prof[typ][1])
    if len(pays)==0: ll=np.zeros(len(SG))
    else:
        la=A[[IDX[p] for p in pays]]; lb=B[[IDX[p] for p in pays]]; w=2.0**(-np.asarray(ages)/hl)
        ll=(w[:,None]*np.log(SG[None,:]*lb[:,None]+(1-SG[None,:])*la[:,None])).sum(0)
    if logprior is not None: ll=ll+logprior
    p=np.exp(ll-ll.max()); return p/p.sum()
# EB prior from train
def eb_prior(by_type=True,mix=0.9):
    d=defaultdict(list)
    for q in tq:
        ticks,pay,win,lags,qlag=he.machine_arrays(th[(q['episode_id'],q['machine_id'])],q['cutoff'])
        mw=pay>0; d[q['machine_type'] if by_type else 'all'].append(post_pi(pay[mw],q['machine_type'],q['cutoff']-1-ticks[mw],1e9))
    return {k:np.log(mix*np.mean(v,0)+(1-mix)/len(SG)) for k,v in d.items()}
pri=eb_prior()
def sizes(hl,use_prior):
    out={}
    for q in vq:
        ticks,pay,win,lags,qlag=he.machine_arrays(vh[(q['episode_id'],q['machine_id'])],q['cutoff'])
        mw=pay>0; p=post_pi(pay[mw],q['machine_type'],q['cutoff']-1-ticks[mw],hl,pri[q['machine_type']] if use_prior else None)
        pi=(p*SG).sum(); out[q['query_id']]=pi*np.array(prof[q['machine_type']][1])+(1-pi)*np.array(prof[q['machine_type']][0])
    return out
for hl in [40,60,100,1e9]:
    for up in [False,True]:
        sz=sizes(hl,up)
        for qid in cb.comp: cb.comp[qid]['size']=sz[qid]
        print(hl,up,end=' '); score(cb.combine(['2','4b'],avg=True))
