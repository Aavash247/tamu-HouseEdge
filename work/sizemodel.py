import numpy as np
PROF={'steady':[[.7540,.1968,.0482,.0010],[.4551,.4359,.1068,.0022]],
'balanced':[[.4999,.3479,.1305,.0217],[.2141,.5467,.2050,.0342]],
'jackpot':[[.3351,.2449,.3149,.1050],[.1208,.3239,.4165,.1388]]}
IDX={1:0,2:1,5:2,20:3}
GRID=np.linspace(0.005,0.995,100)
def iid_mix(pays,typ,ages=None,hl=1e9,prior_a=1.0,prior_b=1.0):
    A=np.array(PROF[typ][0]);B=np.array(PROF[typ][1])
    if len(pays)==0: pi=prior_a/(prior_a+prior_b)
    else:
        la=A[[IDX[p] for p in pays]]; lb=B[[IDX[p] for p in pays]]
        w=np.ones(len(pays)) if ages is None else 2.0**(-np.asarray(ages)/hl)
        ll=(w[:,None]*np.log(GRID[None,:]*lb[:,None]+(1-GRID[None,:])*la[:,None])).sum(0)
        ll+=(prior_a-1)*np.log(GRID)+(prior_b-1)*np.log(1-GRID)
        p=np.exp(ll-ll.max()); p/=p.sum(); pi=(p*GRID).sum()
    return pi*B+(1-pi)*A
def hmm(pays,typ,rho=0.05,p0=0.5):
    A=np.array(PROF[typ][0]);B=np.array(PROF[typ][1]); b=p0
    for p in pays:
        la=A[IDX[p]]; lb=B[IDX[p]]
        b=b*lb/(b*lb+(1-b)*la)
        b=b*(1-rho)+(1-b)*rho
    return b*B+(1-b)*A
