from exp import *
import harness
cnt=0
for q in vq:
    if harness.C[q['query_id']] not in ('shared_change','drift'): continue
    ticks,pay,win,lags,qlag=he.machine_arrays(vh[(q['episode_id'],q['machine_id'])],q['cutoff'])
    best=None
    for (i,j) in cfg['pairs']:
        m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
        ev,p=he.run_flip(grid,cells,win[m],g,qc,0.001)
        if best is None or ev>best[0]: best=(ev,i,j)
    _,i,j=best
    out=[]
    for lo,hi in [(0,70),(70,140),(140,q['cutoff'])]:
        mm=(ticks>=lo)&(ticks<hi)&~np.isnan(lags[:,i])&~np.isnan(lags[:,j])
        r=[win[mm&(lags[:,i]==a)&(lags[:,j]==b)].mean() for a in (0,1) for b in (0,1)]
        out.append(' '.join(f'{x:.2f}' for x in r))
    print(harness.C[q['query_id']][:5],q['machine_type'][:4],(i,j),' | '.join(out))
    cnt+=1
    if cnt>30: break
