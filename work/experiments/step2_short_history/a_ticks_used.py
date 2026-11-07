"""2.0a: how many observed ticks each part of the final model actually uses for short-history machines."""
import sys, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "entry-final"))
import houseedge as he
cfg = json.loads((ROOT / "entry-final/model.json").read_text())
for split in ("validation", "test"):
    q = he.read_queries(ROOT / f"public/{split}/queries.csv"); h = he.read_history(ROOT / f"public/{split}/history.csv")
    stats = {"n_observed": [], "recent_block": [], "older_sampled": [], "null_model_rows": [], "size_model_wins": []}
    pair_rows = {tuple(p): [] for p in cfg["pairs"]}; pair_rows_old = {tuple(p): [] for p in cfg["pairs"]}
    for x in q:
        s = h[(x["episode_id"], x["machine_id"])]
        if len(s) != 64:
            continue
        ticks, pay, win, lags, qlag = he.machine_arrays(s, x["cutoff"])
        recent = [t for t in ticks if t >= x["cutoff"] - 40]
        stats["n_observed"].append(len(ticks)); stats["recent_block"].append(len(recent))
        stats["older_sampled"].append(len(ticks) - len(recent)); stats["null_model_rows"].append(len(ticks))
        stats["size_model_wins"].append(int((pay > 0).sum()))
        for (i, j) in cfg["pairs"]:
            m, *_ = he.pair_inputs(ticks, lags, qlag, i, j)
            pair_rows[(i, j)].append(int(m.sum()))
            pair_rows_old[(i, j)].append(int((m & (ticks < x["cutoff"] - 40)).sum()))
    print(f"{split}: {len(stats['n_observed'])} short-history machines (n == 64)")
    for k, v in stats.items():
        print(f"   {k:18s} mean={np.mean(v):6.2f} min={min(v)} max={max(v)}")
    for p in pair_rows:
        print(f"   pair {p}: rows usable by the pair likelihood mean={np.mean(pair_rows[p]):.2f} "
              f"(of which from the 24 older sampled ticks: mean={np.mean(pair_rows_old[p]):.2f}, max={max(pair_rows_old[p])})")
