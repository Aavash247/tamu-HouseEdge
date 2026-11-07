from exp import *
import pandas as pd
posts=train_posts(); pt=make_prior(posts,lambda q:q['machine_type'])
pred=pd.DataFrame(run(lambda q:pt[q['machine_type']],rho=0.001)).T; pred.columns=COLS; pred=pred.loc[T.index]; tv=T[COLS]
score(pred)
A=pd.concat([tv.p_0, pred[COLS[1:]].div(1-pred.p_0,axis=0).mul(1-tv.p_0,axis=0)],axis=1); print('perfect win',end=' '); score(A)
B=pd.concat([pred.p_0, tv[COLS[1:]].div(1-tv.p_0,axis=0).mul(1-pred.p_0,axis=0)],axis=1); print('perfect size',end=' '); score(B)
pred.to_pickle('work/cur_pred.pkl')
# biggest win errors
e=(pred.p_0-tv.p_0); import harness
d=pd.DataFrame({'err':e,'p0':pred.p_0,'t0':tv.p_0,'cond':harness.C.loc[T.index]})
print(d.assign(ae=e.abs()).sort_values('ae',ascending=False).head(25).round(3))
print(d.groupby('cond').err.apply(lambda x:(x**2).mean()))
