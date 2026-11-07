from exp import *
prof=cfg['size_profiles']; IDX={1:0,2:1,5:2,20:3}
G=np.linspace(0.01,0.99,50)
def mixll(la,lb):
    if len(la)==0: return 0.0
    return np.log(G[None,:]*lb[:,None]+(1-G[None,:])*la[:,None]).sum(0).max()
res=defaultdict(list); n=0
for q in tq:
    ser=th[(q['episode_id'],q['machine_id'])]; ticks=np.array(sorted(t for t in ser if t<q['cutoff']))
    if len(ticks)<150: continue
    pay=np.array([ser[t] for t in ticks]); w=pay>0; tw=ticks[w]
    A=np.array(prof[q['machine_type']][0]); B=np.array(prof[q['machine_type']][1])
    la=A[[IDX[p] for p in pay[w]]]; lb=B[[IDX[p] for p in pay[w]]]; base=mixll(la,lb)
    for k in range(2,13):
        g=sum(mixll(la[tw%k==r],lb[tw%k==r]) for r in range(k))-base
        res[k].append(g/(k-1))  # per extra parameter
    # time blocks (slow switching): 4 blocks
    for nb in [2,4,8]:
        edges=np.linspace(0,q['cutoff'],nb+1)
        g=sum(mixll(la[(tw>=edges[b])&(tw<edges[b+1])],lb[(tw>=edges[b])&(tw<edges[b+1])]) for b in range(nb))-base
        res[f'blk{nb}'].append(g/(nb-1))
    n+=1
print(n,{k:round(np.mean(v),2) for k,v in res.items()})
