import sys,math,collections; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
TQ,TH=load('train')
cnt=collections.Counter(); margins=[]
for r in TQ.itertuples():
    ticks,pay,win,L,qlag=bp.machine_arrays(TH[r.machine_id],r.cutoff+1)  # include label? no: history only <cutoff
    m0=float(np.clip(win.mean(),0.05,0.95)); gaps=np.diff(np.concatenate([[ticks[0]-1],ticks]))
    evn,_=bp.run_bocpd(np.zeros(len(win),int),win,gaps,1,0,1/400,2.0,m0)
    evs={}
    for (i,j) in bp.PAIRS:
        m=~np.isnan(L[:,i])&~np.isnan(L[:,j]); tk=ticks[m]
        g=np.diff(np.concatenate([[tk[0]-1],tk]))
        evs[(i,j)],_=bp.run_bocpd((L[m,i]==L[m,j]).astype(int),win[m],g,2,0,1/400,2.0,m0)
    best=max(evs,key=evs.get); s=sorted(evs.values())
    cnt[best if evs[best]-evn>3 else 'null']+=1; margins.append(evs[best]-evn)
print(cnt.most_common(12))
print(np.percentile(margins,[5,25,50,75,95]))
