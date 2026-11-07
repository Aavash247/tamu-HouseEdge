from exp import *
import pickle
comp=pickle.load(open('work/comp_val.pkl','rb'))
PR=dict(zip(map(tuple,cfg['pairs']),cfg['pair_prior']))
def combine(fams,fam_w=None,null_prior=0.01,avg=False,temp=1.0):
    preds={}
    for qid,rec in comp.items():
        logs=[math.log(null_prior)+rec['null'][0]];ps=[rec['null'][1]]
        if avg:
            # per-family BMA then average families
            pf=[]
            for f in fams:
                l=[math.log(null_prior)+rec['null'][0]];p=[rec['null'][1]]
                for pr,v in PR.items(): l.append(rec[(f,)+pr][0]+math.log(v)); p.append(rec[(f,)+pr][1])
                l=np.array(l); w=np.exp(l-l.max()); w/=w.sum(); pf.append((w*np.array(p)).sum())
            pwin=float(np.dot(fam_w or [1/len(fams)]*len(fams),pf))
        else:
            for f in fams:
                for pr,v in PR.items():
                    logs.append(temp*rec[(f,)+pr][0]+math.log(v)+math.log((fam_w or {}).get(f,1.0))); ps.append(rec[(f,)+pr][1])
            l=np.array(logs); w=np.exp(l-l.max()); w/=w.sum(); pwin=float((w*np.array(ps)).sum())
        pwin=min(max(pwin,0.002),0.998)
        preds[qid]=[1-pwin]+list(pwin*rec['size'])
    return preds
for fams in [['2'],['4b'],['4a'],['2','4b'],['4a','4b'],['2','4a','4b']]:
    print(fams,'bma',end=' '); score(combine(fams))
    if len(fams)>1: print(fams,'avg',end=' '); score(combine(fams,avg=True))
