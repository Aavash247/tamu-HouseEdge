from exp import *
import itertools, pandas as pd
prof=cfg['size_profiles']; IDX={1:0,2:1,5:2,20:3}
G=np.linspace(0.01,0.99,50)
def mixll(la,lb):
    if len(la)==0: return 0.0
    return np.log(G[None,:]*lb[:,None]+(1-G[None,:])*la[:,None]).sum(0).max()
gains=defaultdict(list); winres=defaultdict(list)
eps_=defaultdict(list)
for q in tq: eps_[q['episode_id']].append(q)
n=0
for ep,qs_ in list(eps_.items()):
    if len(qs_)<2: continue
    for q in qs_:
        H=th; ser=H[(q['episode_id'],q['machine_id'])]
        ticks=np.array(sorted(t for t in ser if t<q['cutoff']))
        if len(ticks)<150: continue
        pay=np.array([ser[t] for t in ticks]); w=pay>0
        A=np.array(prof[q['machine_type']][0]); B=np.array(prof[q['machine_type']][1])
        la=A[[IDX[p] for p in pay[w]]]; lb=B[[IDX[p] for p in pay[w]]]
        base=mixll(la,lb); tw=ticks[w]
        best=[]
        for o in qs_:
            if o['machine_id']==q['machine_id']: continue
            so=H[(o['episode_id'],o['machine_id'])]
            for k in range(1,9):
                f=np.array([so.get(t-k,-1) for t in tw])
                ok=f>=0; fw=(f>0)&ok
                g=mixll(la[fw],lb[fw])+mixll(la[ok&~fw],lb[ok&~fw])+mixll(la[~ok],lb[~ok])-base
                best.append(g)
        gains['nbr_max'].append(max(best)); gains['nbr_mean'].append(np.mean(best))
        n+=1
    if n>150: break
print(n, {k:(round(np.mean(v),2),round(np.median(v),2)) for k,v in gains.items()})
