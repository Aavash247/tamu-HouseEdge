from exp import *
posts=train_posts()
pt=make_prior(posts,lambda q:q['machine_type'])
pts=make_prior(posts,lambda q:(q['machine_type'],q['signal']))
print('type',end=' '); score(run(lambda q:pt[q['machine_type']]))
print('type+sig',end=' '); score(run(lambda q:pts[(q['machine_type'],q['signal'])]))
print('avg',end=' '); score(run(lambda q:np.logaddexp(pt[q['machine_type']],pts[(q['machine_type'],q['signal'])])-math.log(2)))
for rho in [0.001,0.0005]:
    print('type rho',rho,end=' '); score(run(lambda q:pt[q['machine_type']],rho=rho))
