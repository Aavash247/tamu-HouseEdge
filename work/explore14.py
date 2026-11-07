from exp import *
import pandas as pd
pred=pd.read_pickle('work/cur_pred.pkl')
import harness
for qid in ['main-validation-e00087-m006-next','main-validation-e00005-m007-next','main-validation-e00017-m003-next','main-validation-e00059-m006-next']:
    q=[x for x in vq if x['query_id']==qid][0]
    ticks,pay,win,lags,qlag=he.machine_arrays(vh[(q['episode_id'],q['machine_id'])],q['cutoff'])
    print(qid,q['machine_type'],q['signal'],len(ticks),'pred',round(pred.p_0[qid],3),'true',round(T.p_0[qid],3), 'qlag',qlag[1:])
    for (i,j) in [(1,3),(2,5),(4,7)]:
        m,cells,g,qc=he.pair_inputs(ticks,lags,qlag,i,j)
        ev,p,w=he.run_flip(grid,cells,win[m],g,qc,0.001,return_post=True)
        # segment rates
        segs=[]
        for lo in range(0,q['cutoff'],50):
            mm=m&(ticks>=lo)&(ticks<lo+50); c=(lags[mm,i]==lags[mm,j]); y=win[mm]
            segs.append(f"{y[c].mean():.2f}/{y[~c].mean():.2f}")
        print(f"  ({i},{j}) ev={ev:.1f} p={p:.3f} qc={qc}", ' '.join(segs))
