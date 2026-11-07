from exp import *
import itertools, pandas as pd
prof=cfg['size_profiles']; IDX={1:0,2:1,5:2,20:3}
G=np.linspace(0.01,0.99,50)
def mixll(la,lb):
    if len(la)==0: return 0.0
    return np.log(G[None,:]*lb[:,None]+(1-G[None,:])*la[:,None]).sum(0).max()
tot=defaultdict(float); n=0
allq=[(q,th) for q in tq]+[(q,vh) for q in vq]
for q,H in allq:
    ticks,pay,win,lags,qlag=he.machine_arrays(H[(q['episode_id'],q['machine_id'])],q['cutoff'])
    if len(ticks)<150: continue
    pos={int(t):k for k,t in enumerate(ticks)}
    PL=np.full((len(ticks),9),np.nan)
    for k in range(1,9):
        for idx,t in enumerate(ticks):
            p=pos.get(int(t)-k)
            if p is not None: PL[idx,k]=pay[p]
    w=pay>0; A=np.array(prof[q['machine_type']][0]); B=np.array(prof[q['machine_type']][1])
    la=A[[IDX[p] for p in pay[w]]]; lb=B[[IDX[p] for p in pay[w]]]
    base=mixll(la,lb); P=PL[w]
    feats={}
    for k in range(1,9):
        feats[f'w{k}']=P[:,k]>0; feats[f'b{k}']=P[:,k]>=2; feats[f'j{k}']=P[:,k]>=5
    for i,j in itertools.combinations(range(1,9),2):
        feats[f'xw{i}{j}']=(P[:,i]>0)==(P[:,j]>0); feats[f'xb{i}{j}']=(P[:,i]>=2)==(P[:,j]>=2)
    for name,f in feats.items():
        ok=~np.isnan(P[:,1:]).any(1); f=f&ok
        tot[name]+=mixll(la[f],lb[f])+mixll(la[~f],lb[~f])-base
    n+=1
s=pd.Series(tot)/n
print(n); print(s.sort_values(ascending=False).head(15).round(2)); print(s.describe().round(2))
