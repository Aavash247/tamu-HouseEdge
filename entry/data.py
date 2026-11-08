"""Reads the released CSVs and writes the forecast CSV."""
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

COLUMNS = ("query_id", "p_0", "p_1", "p_2", "p_5", "p_20")
MAX_LAG = 8


def read_queries(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return [dict(query_id=r["query_id"], episode_id=r["episode_id"], machine_id=r["machine_id"],
                     machine_type=r["machine_type"], signal=int(r["signal"]), cutoff=int(r["cutoff"]))
                for r in csv.DictReader(f)]


def read_history(path):
    """{(episode_id, machine_id): {tick: payout}}"""
    hist = defaultdict(dict)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])
    return hist


def read_size_targets(public):
    """Validation target size split (given a win) per machine type."""
    qtype = {q["query_id"]: q["machine_type"] for q in read_queries(public / "validation" / "queries.csv")}
    groups = defaultdict(list)
    with open(public / "validation" / "targets.csv", newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            p = np.array([float(r[c]) for c in ("p_1", "p_2", "p_5", "p_20")])
            if p.sum() > 1e-6:
                groups[qtype[r["query_id"]]].append(p / p.sum())
    return groups


def write_submission(path, preds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for qid in sorted(preds):
            w.writerow([qid] + [repr(float(x)) for x in preds[qid]])


def machine_arrays(series, cutoff):
    """Ticks before the cutoff, payouts, wins, lag-win matrix and query lags (NaN = unobserved)."""
    ticks = np.array(sorted(t for t in series if t < cutoff), dtype=int)
    pay = np.array([series[t] for t in ticks], dtype=int)
    win = (pay > 0).astype(float)
    pos = {int(t): k for k, t in enumerate(ticks)}
    lags = np.full((len(ticks), MAX_LAG + 1), np.nan)
    for k in range(1, MAX_LAG + 1):
        for idx, t in enumerate(ticks):
            p = pos.get(int(t) - k)
            if p is not None:
                lags[idx, k] = win[p]
    qlag = [win[pos[cutoff - k]] if (cutoff - k) in pos else np.nan for k in range(MAX_LAG + 1)]
    return ticks, pay, win, lags, qlag
