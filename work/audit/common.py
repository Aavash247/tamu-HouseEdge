"""Shared helpers for the overfitting audit. Imports the final model code read-only."""
import copy, csv, itertools, json, math, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "entry-final"))
sys.path.insert(0, str(ROOT))
import houseedge as he  # noqa: E402

PUB = ROOT / "public"
FINAL_CFG = json.loads((ROOT / "entry-final" / "model.json").read_text())
CONDS = ["stable", "short_history", "drift", "shared_change"]
PAY = (0, 1, 2, 5, 20)


def load(split):
    q = he.read_queries(PUB / split / "queries.csv")
    h = he.read_history(PUB / split / "history.csv")
    cond = {}
    p = PUB / split / "conditions.csv"
    if p.exists():
        with open(p) as f:
            cond = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}
    labels = {}
    p = PUB / split / "labels.csv"
    if p.exists():
        with open(p) as f:
            labels = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(f)}
    return q, h, cond, labels


def fit_pairs(queries, hist, cfg, grid):
    """Same procedure as Forecaster.fit: best-evidence pair per machine over all 28 pairs."""
    counts = defaultdict(float)
    best_by_machine = {}
    for q in queries:
        ticks, pay, win, lags, qlag = he.machine_arrays(hist[(q["episode_id"], q["machine_id"])], q["cutoff"])
        if len(ticks) < 20:
            continue
        evs = {}
        for (i, j) in he.ALL_PAIRS:
            m, cells, gaps, qc = he.pair_inputs(ticks, lags, qlag, i, j)
            if m.sum() < 10:
                continue
            evs[(i, j)], _ = he.run_flip(grid, cells, win[m], gaps, qc, cfg["rho"])
        if evs:
            best = max(evs, key=evs.get)
            srt = sorted(evs.values())
            counts[best] += 1.0
            best_by_machine[q["query_id"]] = (best, srt[-1] - srt[-2])
    total = sum(counts.values())
    pairs = sorted([p for p, c in counts.items() if c / total >= cfg["min_pair_share"]])
    return pairs, [counts[p] / sum(counts[x] for x in pairs) for p in pairs], counts, best_by_machine


def size_profiles_em(queries, hist, iters=300):
    """Train-only alternative to validation-target clustering: 2-profile EM over winning payouts."""
    by_type = defaultdict(list)
    for q in queries:
        s = hist[(q["episode_id"], q["machine_id"])]
        pays = [he.WIN_PAYOUTS.index(v) for t, v in s.items() if t < q["cutoff"] and v > 0]
        if pays:
            by_type[q["machine_type"]].append(np.array(pays))
    out = {}
    for typ, machines in by_type.items():
        allp = np.concatenate(machines)
        base = np.bincount(allp, minlength=4) / len(allp)
        A = base * np.array([1.3, 0.8, 0.8, 0.8]); A /= A.sum()
        B = base * np.array([0.7, 1.2, 1.2, 1.2]); B /= B.sum()
        pis = np.full(len(machines), 0.5)
        for _ in range(iters):
            nA = np.zeros(4); nB = np.zeros(4)
            for k, p in enumerate(machines):
                lb = pis[k] * B[p]; la = (1 - pis[k]) * A[p]
                r = lb / (lb + la)
                pis[k] = (r.sum() + 1) / (len(p) + 2)
                nB += np.bincount(p, weights=r, minlength=4)
                nA += np.bincount(p, weights=1 - r, minlength=4)
            A = (nA + 1e-3) / (nA + 1e-3).sum(); B = (nB + 1e-3) / (nB + 1e-3).sum()
        prof = sorted([A.tolist(), B.tolist()], key=lambda v: -v[0])
        out[typ] = prof
    return out


def build_cfg(train_q, train_h, size_profiles):
    """Fit a config exactly like fit.py, but on an arbitrary set of training queries."""
    cfg = {k: v for k, v in FINAL_CFG.items() if k not in ("pairs", "pair_prior", "prior2", "prior4", "size_profiles")}
    grid = he.FlipGrid(cfg["grid_size"])
    pairs, prior, _, _ = fit_pairs(train_q, train_h, cfg, grid)
    cfg["pairs"] = [list(p) for p in pairs]
    cfg["pair_prior"] = prior
    cfg["prior2"], cfg["prior4"] = he.Forecaster._rate_priors(train_q, train_h, cfg, grid)
    cfg["size_profiles"] = size_profiles
    return cfg


def brier(p, y):
    onehot = np.array([1.0 if v == y else 0.0 for v in PAY])
    return float(((np.asarray(p) - onehot) ** 2).sum())


def cond_table(rows, key):
    """rows: dicts with episode, cond, and metric `key`. Episode-mean then condition-mean (official weighting)."""
    out = {}
    for c in CONDS:
        eps = defaultdict(list)
        for r in rows:
            if r["cond"] == c:
                eps[r["episode"]].append(r[key])
        out[c] = float(np.mean([np.mean(v) for v in eps.values()])) if eps else float("nan")
    return out
