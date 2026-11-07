import sys,itertools; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
PROF={'steady':[[.7540,.1968,.0482,.0010],[.4551,.4359,.1068,.0022]],
'balanced':[[.4999,.3479,.1305,.0217],[.2141,.5467,.2050,.0342]],
'jackpot':[[.3351,.2449,.3149,.1050],[.1208,.3239,.4165,.1388]]}
IDX={1:0,2:1,5:2,20:3}
d=VQ.set_index('query_id').loc[T.index]; d['cond']=C.loc[T.index]
tv=T[COLS]; sz=tv[COLS[1:]].div(1-tv.p_0,axis=0)
d['st']=[int(abs(s[0]-PROF[t][1][0])<abs(s[0]-PROF[t][0][0])) for s,t in zip(sz.values,d.machine_type)]
def mixll(lA,lB,mask):
    # max over pi of sum log(pi*B+(1-pi)*A) grid
    best=-1e18
    for pi in np.linspace(0.01,0.99,50):
        v=np.log(pi*lB[mask]+(1-pi)*lA[mask]).sum(); best=max(best,v)
    return best
res=[]
for r in d[d.cond=='stable'].reset_index().head(60).itertuples():
    ticks,pay,win,L,qlag=bp.machine_arrays(VH[r.machine_id],r.cutoff)
    ser=VH[r.machine_id]
    w=win>0; A=np.array([PROF[r.machine_type][0][IDX[p]] for p in pay[w]]); B=np.array([PROF[r.machine_type][1][IDX[p]] for p in pay[w]])
    base=mixll(A,B,np.ones(w.sum(),bool))
    # raw payout lags
    PL=np.full((len(ticks),9),np.nan); pos={t:k for k,t in enumerate(ticks)}
    for k in range(1,9):
        for idx,t in enumerate(ticks):
            p=pos.get(t-k); 
            if p is not None: PL[idx,k]=pay[p]
    PLw=PL[w]; Lw=L[w]
    feats={}
    for k in range(1,9):
        feats[f'w{k}']=Lw[:,k]; feats[f'g{k}']=(PLw[:,k]>=2).astype(float); feats[f'p{k}']=(PLw[:,k]>=5).astype(float)
    for i,j in itertools.combinations(range(1,9),2):
        feats[f'x{i}{j}']=(Lw[:,i]==Lw[:,j]).astype(float)
    gains={}
    for name,f in feats.items():
        g=mixll(A,B,f==1)+mixll(A,B,f==0)-base; gains[name]=g
    top=sorted(gains.items(),key=lambda x:-x[1])[:3]
    res.append((r.machine_type,r.st,int(w.sum()),round(base,1),[(n,round(g,1)) for n,g in top]))
for x in res[:30]: print(x)
