import sys; sys.path.insert(0,'work')
from harness import *; import pairmodel as pm, time
t=time.time()
pp={(1,3):1.0,(2,5):1.0,(4,7):1.0,'null':0.3}
pred=pm.predict(VQ,VH,pair_prior=pp); print(time.time()-t); score(pred)
pred=pm.predict(VQ,VH); score(pred)
pred=pm.predict(VQ,VH,pair_prior=pp,hl=40); score(pred)
