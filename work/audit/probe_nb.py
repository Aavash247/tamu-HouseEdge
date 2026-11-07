import random
from common import *
model=he.Forecaster(FINAL_CFG); vq,vh,_,_=load("validation"); rng=random.Random(1)
mx=0
for q in rng.sample(vq,60):
    base=model.predict([q],vh)[q["query_id"]]
    h={k:(v if (k[0]!=q["episode_id"] or k[1]==q["machine_id"]) else {t:rng.choice(PAY) for t in v}) for k,v in vh.items()}
    mx=max(mx,np.abs(model.predict([q],h)[q["query_id"]]-base).max())
print("per-query neighbour scramble, max |diff|:",mx)
