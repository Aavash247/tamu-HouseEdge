"""Diagnostic: house-edge error split by whether a query-time lag (1..8) is unobserved."""
import sys, json, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import he, build_cfg, size_profiles_from_targets, read_probs
seed = sys.argv[1] if len(sys.argv) > 1 else "1"
d = Path(f"data/seed{seed}")
tq = he.read_queries(d/"train/queries.csv"); th = he.read_history(d/"train/history.csv")
model = he.Forecaster(build_cfg(tq, th, size_profiles_from_targets(d/"train")))
vq = he.read_queries(d/"test/queries.csv"); vh = he.read_history(d/"test/history.csv")
cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(d/"test/conditions.csv"))}
truth = read_probs(d/"test/targets.csv")
sub = [q for q in vq if cond[q["query_id"]] in ("stable", "drift", "shared_change")]
pred = model.predict(sub, vh)
res = {True: [], False: []}
for q in sub:
    s = vh[(q["episode_id"], q["machine_id"])]
    miss = any((q["cutoff"] - k) not in s for k in range(1, 9))
    res[miss].append(float(((pred[q["query_id"]] - truth[q["query_id"]]) ** 2).sum()))
for k, v in res.items():
    print(f"query lag 1-8 missing={k}: n={len(v)} mean error={np.mean(v):.4f}")
