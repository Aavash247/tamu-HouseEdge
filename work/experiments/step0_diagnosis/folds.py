"""Episode-level fold assignment (identical to work/audit/part2.py). Fold 3 of TRAIN is LOCKED."""
import csv
from pathlib import Path
ROOT = Path(__file__).resolve().parents[3]
CONDS = ["stable", "short_history", "drift", "shared_change"]
LOCKED_FOLD = 3

def fold_map(split="train", nfold=4):
    ep_cond = {}
    q = {r["query_id"]: r["episode_id"] for r in csv.DictReader(open(ROOT / f"public/{split}/queries.csv"))}
    for r in csv.DictReader(open(ROOT / f"public/{split}/conditions.csv")):
        ep_cond[q[r["query_id"]]] = r["condition"]
    out = {}
    for c in CONDS:
        for n, e in enumerate(sorted(e for e, cc in ep_cond.items() if cc == c)):
            out[e] = n % nfold
    return out, ep_cond
