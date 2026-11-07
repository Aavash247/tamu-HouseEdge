import sys; sys.path.insert(0,'work')
from harness import *; import pairmodel as pm
pp={(1,3):1.0,(2,5):1.0,(4,7):1.0,'null':0.3}
for hl,hs in [(40,None),(40,1e9),(25,1e9),(60,1e9),(25,80),(40,80)]:
    print(hl,hs,end=' '); score(pm.predict(VQ,VH,hl=hl,hl_sel=hs,pair_prior=pp))
