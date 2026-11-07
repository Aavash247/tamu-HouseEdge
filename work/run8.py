import sys,time; sys.path.insert(0,'work')
from harness import *; import combo
t=time.time()
print('bocpd only',end=' '); score(combo.predict(VQ,VH,use_flip=False)); print(time.time()-t)
for rho in [0.002,0.005,0.01]:
    t=time.time(); print('flip only',rho,end=' '); score(combo.predict(VQ,VH,use_bocpd=False,rho=rho)); print(time.time()-t)
print('both',end=' '); score(combo.predict(VQ,VH,rho=0.005))
