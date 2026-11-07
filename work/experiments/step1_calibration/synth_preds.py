"""Save the base model's synthetic-test predictions per seed (fit on synthetic train only), for post-processing checks."""
import sys, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
SA = Path(__file__).resolve().parents[2] / "synthetic_audit"
def run(seed):
    sys.path.insert(0, str(SA))
    from evaluate import he, build_cfg, size_profiles_from_targets
    d = SA / f"data/seed{seed}"
    tq = he.read_queries(d/"train/queries.csv"); th = he.read_history(d/"train/history.csv")
    m = he.Forecaster(build_cfg(tq, th, size_profiles_from_targets(d/"train")))
    vq = he.read_queries(d/"test/queries.csv"); vh = he.read_history(d/"test/history.csv")
    p = {k: [float(x) for x in v] for k, v in m.predict(vq, vh).items()}
    n = {q["query_id"]: len(vh[(q["episode_id"], q["machine_id"])]) for q in vq}
    json.dump({"pred": p, "n": n, "type": {q["query_id"]: q["machine_type"] for q in vq}}, open(Path(__file__).parent / f"synth_seed{seed}.json", "w"))
    return seed
if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex: print(list(ex.map(run, [1,2,3,4,5])))
