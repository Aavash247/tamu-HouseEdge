import sys; sys.path.insert(0,'work')
from harness import *; import bocpd as bp
pp={(1,3):1.0,(2,5):1.0,(4,7):1.0,'null':0.01}; P3=[(1,3),(2,5),(4,7)]
for a in [0.5,1.0,2.0]:
  for hz in [1/400,1/1000]:
    print(a,round(1/hz),end=' '); score(bp.predict(VQ,VH,hazard=hz,a=a,pair_prior=pp,pairs=P3,eps=0.002))
