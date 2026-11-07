"""Per-machine, per-tick residuals on TRAIN folds 0-2 (fold 3 locked, unused).
Episodes in fold f are predicted by a model fit on the other two unlocked folds,
with train-only (EM) size profiles. Residuals for ticks cutoff-24 .. cutoff."""
import json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "audit"))
from folds import fold_map, LOCKED_FOLD
NT = 25

def run_fold(f):
    from common import he, load, build_cfg, size_profiles_em
    tq, th, tcond, tlab = load("train")
    fm, _ = fold_map("train")
    tr = [q for q in tq if fm[q["episode_id"]] not in (f, LOCKED_FOLD)]
    te = [q for q in tq if fm[q["episode_id"]] == f]
    model = he.Forecaster(build_cfg(tr, th, size_profiles_em(tr, th)))
    rows = []
    for k in range(NT):
        qs, hs = [], {}
        for q in te:
            c = q["cutoff"] - k
            s = th[(q["episode_id"], q["machine_id"])]
            y = tlab[q["query_id"]] if k == 0 else s.get(c)
            nq = dict(q, cutoff=c, query_id=f'{q["query_id"]}:k{k}')
            nq["_y"] = y
            qs.append(nq); hs[(q["episode_id"], q["machine_id"])] = {t: v for t, v in s.items() if t < c}
        pred = model.predict(qs, hs)
        for q in qs:
            p = pred[q["query_id"]]
            rows.append({"episode": q["episode_id"], "machine": q["machine_id"], "type": q["machine_type"],
                         "cond": tcond[q["query_id"].split(":k")[0]], "fold": f, "tick": q["cutoff"], "k": k,
                         "y": q["_y"], "p": [float(x) for x in p]})
    return rows

if __name__ == "__main__":
    with ProcessPoolExecutor(3) as ex:
        allrows = [r for rs in ex.map(run_fold, (0, 1, 2)) for r in rs]
    json.dump(allrows, open(HERE / "residual_rows.json", "w"))
    print("rows", len(allrows), "observed", sum(r["y"] is not None for r in allrows))
