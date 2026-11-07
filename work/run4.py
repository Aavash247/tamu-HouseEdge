import sys,time; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
pp={(1,3):1.0,(2,5):1.0,(4,7):1.0,'null':0.3}
P3=[(1,3),(2,5),(4,7)]
pred=pd.DataFrame(bp.predict(VQ,VH,hazard=1/400,pair_prior=pp,pairs=P3)).T; pred.columns=COLS
pred=pred.loc[T.index]; tv=T[COLS]
score(pred)
# perfect win prob, model size
pw_t=1-tv.p_0; sz_m=pred[COLS[1:]].div(1-pred.p_0,axis=0)
A=pd.concat([tv.p_0, sz_m.mul(pw_t,axis=0)],axis=1); print('perfect win:',end=''); score(A)
sz_t=tv[COLS[1:]].div(pw_t,axis=0); pw_m=1-pred.p_0
B=pd.concat([pred.p_0, sz_t.mul(pw_m,axis=0)],axis=1); print('perfect size:',end=''); score(B)
# look at true size distributions
d=VQ.set_index('query_id').loc[T.index]
sz_t['typ']=d.machine_type; sz_t['cond']=C.loc[T.index]
print(sz_t.round(2).groupby('typ').head(12).sort_values('typ').to_string())
