import sys,itertools; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
tv=T[COLS]; pw=1-tv.p_0; sz=tv[COLS[1:]].div(pw,axis=0)
d=VQ.set_index('query_id').loc[T.index]
sz['typ']=d.machine_type
sz['state']=0
for typ,g in sz.groupby('typ'):
    print(typ); print(g[COLS[1:]].round(4).value_counts().head(6))
# define state = p_1 > midpoint
mid={'steady':0.6,'balanced':0.35,'jackpot':0.23}
st=np.array([s.p_1<mid[s.typ] for s in sz.itertuples()]).astype(int)  # 1= high-size
d['st']=st; d['cond']=C.loc[T.index]
print(d.groupby(['machine_type','signal']).st.mean())
# relate to lag payouts (raw payout size, and win) at query
rows=[]
for r in d.reset_index().itertuples():
    ser=VH[r.machine_id]; cu=r.cutoff
    rows.append([ser.get(cu-k,np.nan) for k in range(1,9)])
Lg=np.array(rows,float)
for k in range(8):
    m=~np.isnan(Lg[:,k])
    for f,name in [(lambda x:x>0,'win'),(lambda x:x>=2,'ge2'),(lambda x:x>=5,'ge5')]:
        x=f(Lg[m,k]).astype(float); print(k+1,name, round(np.corrcoef(x,st[m])[0,1],3), end='; ')
    print()
