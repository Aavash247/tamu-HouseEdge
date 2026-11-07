import sys,time; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
pp={(1,3):1.0,(2,5):1.0,(4,7):1.0,'null':0.3}
P3=[(1,3),(2,5),(4,7)]
for hz in [1/100,1/200,1/400]:
    t=time.time(); print(hz,end=' '); score(bp.predict(VQ,VH,hazard=hz,pair_prior=pp,pairs=P3)); print(time.time()-t)
