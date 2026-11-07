import numpy as np, math
import bocpd as bp, fliphmm as fh, sizemodel as sm
def predict(q,H,pairs=((1,3),(2,5),(4,7)),pair_prior=None,rho=0.005,use_bocpd=True,use_flip=True,hazard=1/1000,a=1.0,eps=0.002,fam_prior=(1.0,1.0),null_prior=0.01,size_fn=None,ret_parts=False):
    out={}
    for r in q.itertuples():
        ticks,pay,win,L,qlag=bp.machine_arrays(H[r.machine_id],r.cutoff)
        m0=float(np.clip(win.mean(),0.05,0.95))
        gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
        logs=[];preds=[]
        ev,p=bp.run_bocpd(np.zeros(len(win),int),win,gaps,1,0,hazard,a,m0)
        logs.append(ev+math.log(null_prior)); preds.append(p)
        for (i,j) in pairs:
            m=~np.isnan(L[:,i])&~np.isnan(L[:,j]); tk=ticks[m]
            g=np.diff(np.concatenate([[tk[0]-1],tk])); cells=(L[m,i]==L[m,j]).astype(int)
            qc=None if (np.isnan(qlag[i]) or np.isnan(qlag[j])) else int(qlag[i]==qlag[j])
            lpp=math.log(pair_prior.get((i,j),1.0)) if pair_prior else 0.0
            if use_bocpd:
                ev,p=bp.run_bocpd(cells,win[m],g,2,qc,hazard,a,m0); logs.append(ev+lpp+math.log(fam_prior[0])); preds.append(p)
            if use_flip:
                ev,p=fh.run_flip(cells,win[m],g,qc,rho); logs.append(ev+lpp+math.log(fam_prior[1])); preds.append(p)
        logs=np.array(logs); pw=np.exp(logs-logs.max()); pw/=pw.sum()
        pwin=float(np.clip((pw*np.array(preds)).sum(),eps,1-eps))
        mw=pay>0
        sz=sm.iid_mix(pay[mw],r.machine_type,r.cutoff-1-ticks[mw],hl=60) if size_fn is None else size_fn(pay[mw],r.machine_type,r.cutoff-1-ticks[mw])
        out[r.query_id]=[1-pwin]+list(pwin*sz)
    return out
