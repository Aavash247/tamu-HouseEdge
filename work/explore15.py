from exp import *
import itertools
for qid,(pi,pj) in [('main-validation-e00087-m006-next',(4,7)),('main-validation-e00017-m003-next',(1,3)),('main-validation-e00059-m006-next',(1,3))]:
    q=[x for x in vq if x['query_id']==qid][0]
    ticks,pay,win,lags,qlag=he.machine_arrays(vh[(q['episode_id'],q['machine_id'])],q['cutoff'])
    ok=~np.isnan(lags[:,1:]).any(1)
    L=lags[ok]; y=win[ok]
    on=(L[:,pi]==L[:,pj]) if np.nanmean(y[(L[:,pi]==L[:,pj])])>np.nanmean(y[(L[:,pi]!=L[:,pj])]) else (L[:,pi]!=L[:,pj])
    print(qid, 'on-cell n',on.sum(),'rate',y[on].mean().round(2), 'query on?', (qlag[pi]==qlag[pj])==(on[0]==(L[0,pi]==L[0,pj])))
    res=[]
    for k in range(1,9):
        f=L[on,k]==1; res.append((f'w{k}',round(y[on][f].mean(),2),round(y[on][~f].mean(),2)))
    for i,j in itertools.combinations(range(1,9),2):
        f=L[on,i]==L[on,j]; res.append((f'x{i}{j}',round(y[on][f].mean(),2),round(y[on][~f].mean(),2)))
    res.sort(key=lambda r:-abs(r[1]-r[2])); print(res[:6])
    # payout size of lags
    PLs={}
