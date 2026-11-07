import pandas as pd, numpy as np
P='public/'
COLS=['p_0','p_1','p_2','p_5','p_20']; PAY=[0,1,2,5,20]
def load(split):
    q=pd.read_csv(P+f'{split}/queries.csv'); h=pd.read_csv(P+f'{split}/history.csv')
    H={}
    for (m),g in h.groupby('machine_id'):
        H[m]=dict(zip(g.tick.to_numpy().tolist(),g.payout.to_numpy().tolist()))
    return q,H
VQ,VH=load('validation')
T=pd.read_csv(P+'validation/targets.csv').set_index('query_id')
C=pd.read_csv(P+'validation/conditions.csv').set_index('query_id').condition
REF=pd.read_csv(P+'validation/reference.csv').set_index('query_id')[COLS]
def _err(pred):
    df=pd.DataFrame(pred).T if isinstance(pred,dict) else pred
    df=df.loc[T.index]; df.columns=COLS
    se=((df.values-T[COLS].values)**2).sum(1)
    return pd.DataFrame({'se':se,'ep':T.episode_id.values,'cond':C.loc[T.index].values},index=T.index)
REFE=_err(REF)
def score(pred, mask=None, verbose=True):
    e=_err(pred); r=REFE
    if mask is not None: e=e[mask(e)]; r=r.loc[e.index]
    out={}
    for c in ['stable','short_history','drift','shared_change']:
        a=e[e.cond==c].groupby('ep').se.mean().mean(); b=r[r.cond==c].groupby('ep').se.mean().mean()
        out[c]=a/b
    R=np.mean(list(out.values())); pts=60/(1+R)
    if verbose: print(f"pts={pts:.3f} R={R:.4f} "+" ".join(f"{k[:6]}={v:.3f}" for k,v in out.items()))
    return pts,out
if __name__=='__main__':
    b=pd.read_csv('winner-blend/validation_predictions.csv').set_index('query_id')
    score(b); score(REF)
