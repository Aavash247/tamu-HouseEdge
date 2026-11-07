"""House Edge forecaster: lag-pair XOR win model with orientation-flip HMM + size-profile mixture.

Win chance: each machine's win probability depends on whether the win/loss outcomes at
two specific lags agree (an XOR-style interaction). The relationship can flip sign over time
(drift / shared change), so a two-state orientation HMM tracks the current sign while the
per-cell rates are integrated over a grid (Bayesian). Candidate lag pairs and their prior
come from the training split; models are combined by Bayesian model averaging on each
machine's own history (strictly before the cutoff).

Payout size given a win: each machine type has a small set of size profiles; the machine's
past winning payouts give a posterior over the profile mixture weight.
"""

import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

PAYOUTS = (0, 1, 2, 5, 20)
WIN_PAYOUTS = (1, 2, 5, 20)
COLUMNS = ("query_id", "p_0", "p_1", "p_2", "p_5", "p_20")
ALL_PAIRS = list(itertools.combinations(range(1, 9), 2))


# ----------------------------------------------------------------------------- data

def read_queries(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    return [dict(query_id=r["query_id"], episode_id=r["episode_id"], machine_id=r["machine_id"],
                 machine_type=r["machine_type"], signal=int(r["signal"]), cutoff=int(r["cutoff"]))
            for r in rows]


def read_history(path):
    hist = defaultdict(dict)
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])
    return hist


def write_submission(path, preds):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for qid in sorted(preds):
            w.writerow([qid] + [repr(float(x)) for x in preds[qid]])


# ----------------------------------------------------------------------------- features

def machine_arrays(series, cutoff):
    """Observed ticks strictly before cutoff, payouts, win flags, lag-win matrix (NaN = unobserved)."""
    ticks = np.array(sorted(t for t in series if t < cutoff), dtype=int)
    pay = np.array([series[t] for t in ticks], dtype=int)
    win = (pay > 0).astype(float)
    pos = {int(t): k for k, t in enumerate(ticks)}
    lags = np.full((len(ticks), 9), np.nan)
    for k in range(1, 9):
        for idx, t in enumerate(ticks):
            p = pos.get(int(t) - k)
            if p is not None:
                lags[idx, k] = win[p]
    qlag = [win[pos[cutoff - k]] if (cutoff - k) in pos else np.nan for k in range(9)]
    return ticks, pay, win, lags, qlag


# ----------------------------------------------------------------------------- win model

class FlipGrid:
    def __init__(self, n=25):
        g = np.linspace(0.01, 0.99, n)
        a, b = np.meshgrid(g, g, indexing="ij")
        self.q_same = a.ravel()
        self.q_diff = b.ravel()


def run_flip(grid, cells, ys, gaps, qcell, rho):
    """Orientation HMM. sigma=0: P(win|same)=q_same, P(win|diff)=q_diff; sigma=1 swaps them.
    Returns (log evidence, predictive win probability for the query cell)."""
    qs, qd = grid.q_same, grid.q_diff
    k = len(qs)
    b = np.full(k, 0.5)
    ll = np.zeros(k)
    for c, y, g in zip(cells, ys, gaps):
        f = (1.0 - (1.0 - 2.0 * rho) ** g) / 2.0
        b = b * (1.0 - f) + (1.0 - b) * f
        p0 = qs if c == 1 else qd
        p1 = qd if c == 1 else qs
        l0 = p0 if y else 1.0 - p0
        l1 = p1 if y else 1.0 - p1
        tot = b * l0 + (1.0 - b) * l1
        ll += np.log(tot)
        b = b * l0 / tot
    b = b * (1.0 - rho) + (1.0 - b) * rho
    mx = ll.max()
    w = np.exp(ll - mx)
    ev = mx + math.log(w.sum()) - math.log(k)
    w /= w.sum()
    if qcell is None:
        pq = 0.5 * (qs + qd)
    elif qcell == 1:
        pq = b * qs + (1.0 - b) * qd
    else:
        pq = b * qd + (1.0 - b) * qs
    return ev, float((w * pq).sum())


def run_null(ys, gaps, hazard, a, m0):
    """Single-rate Beta-Bernoulli with Bayesian online change points (no lag dependence)."""
    log_r = np.array([0.0])
    n = np.zeros(1)
    s = np.zeros(1)
    ev = 0.0
    for y, g in zip(ys, gaps):
        hz = 1.0 - (1.0 - hazard) ** g
        r = np.exp(log_r - log_r.max())
        r /= r.sum()
        cp = hz
        r = np.concatenate([[cp], r * (1.0 - hz)])
        n = np.concatenate([[0.0], n])
        s = np.concatenate([[0.0], s])
        p = (s + a * m0) / (n + a)
        lik = p if y else 1.0 - p
        tot = (r * lik).sum()
        ev += math.log(tot)
        r = r * lik / tot
        n += 1.0
        s += y
        keep = r > 1e-10
        r, n, s = r[keep], n[keep], s[keep]
        log_r = np.log(r)
    r = np.exp(log_r - log_r.max())
    r /= r.sum()
    return ev, float((r * (s + a * m0) / (n + a)).sum())


def pair_inputs(ticks, lags, qlag, i, j):
    m = ~np.isnan(lags[:, i]) & ~np.isnan(lags[:, j])
    tk = ticks[m]
    gaps = np.diff(np.concatenate([[tk[0] - 1], tk])) if len(tk) else np.array([], dtype=int)
    cells = (lags[m, i] == lags[m, j]).astype(int)
    qc = None if (np.isnan(qlag[i]) or np.isnan(qlag[j])) else int(qlag[i] == qlag[j])
    return m, cells, gaps, qc


# ----------------------------------------------------------------------------- size model

SIZE_GRID = np.linspace(0.005, 0.995, 100)


def size_distribution(pays, ages, profiles, half_life):
    """Posterior-mean mixture of the type's size profiles given past winning payouts."""
    prof = np.array(profiles)  # K x 4
    k = len(prof)
    if k == 1:
        return prof[0]
    idx = [WIN_PAYOUTS.index(int(p)) for p in pays]
    w = 2.0 ** (-np.asarray(ages, dtype=float) / half_life) if len(pays) else np.zeros(0)
    if k == 2:
        if len(pays) == 0:
            return prof.mean(0)
        la = prof[0][idx]
        lb = prof[1][idx]
        ll = (w[:, None] * np.log(SIZE_GRID[None, :] * lb[:, None]
                                  + (1 - SIZE_GRID[None, :]) * la[:, None])).sum(0)
        p = np.exp(ll - ll.max())
        p /= p.sum()
        pi = float((p * SIZE_GRID).sum())
        return pi * prof[1] + (1 - pi) * prof[0]
    # K > 2: weighted EM for mixture weights, with a flat Dirichlet(1) prior
    pi = np.full(k, 1.0 / k)
    if len(pays):
        lik = prof[:, idx].T  # n x K
        for _ in range(50):
            r = lik * pi
            r /= r.sum(1, keepdims=True)
            pi = ((w[:, None] * r).sum(0) + 1.0) / (w.sum() + k)
    return pi @ prof


# ----------------------------------------------------------------------------- model

class Forecaster:
    def __init__(self, config):
        self.cfg = config
        self.grid = FlipGrid(config["grid_size"])

    # --- fitting -----------------------------------------------------------------------
    @staticmethod
    def fit(public, config):
        """Learn candidate lag pairs (train histories) and size profiles (validation targets)."""
        public = Path(public)
        cfg = dict(config)
        grid = FlipGrid(cfg["grid_size"])
        queries = read_queries(public / "train" / "queries.csv")
        hist = read_history(public / "train" / "history.csv")
        counts = defaultdict(float)
        for q in queries:
            ticks, pay, win, lags, qlag = machine_arrays(hist[(q["episode_id"], q["machine_id"])], q["cutoff"])
            if len(ticks) < 20:
                continue
            evs = {}
            for (i, j) in ALL_PAIRS:
                m, cells, gaps, qc = pair_inputs(ticks, lags, qlag, i, j)
                if m.sum() < 10:
                    continue
                evs[(i, j)], _ = run_flip(grid, cells, win[m], gaps, qc, cfg["rho"])
            if evs:
                best = max(evs, key=evs.get)
                counts[best] += 1.0
        total = sum(counts.values())
        pairs = sorted([p for p, c in counts.items() if c / total >= cfg["min_pair_share"]])
        cfg["pairs"] = [list(p) for p in pairs]
        cfg["pair_prior"] = [counts[p] / sum(counts[x] for x in pairs) for p in pairs]
        cfg["size_profiles"] = Forecaster._size_profiles(public)
        return Forecaster(cfg)

    @staticmethod
    def _size_profiles(public):
        """Cluster validation target size distributions (given a win) per machine type."""
        qtype = {q["query_id"]: q["machine_type"] for q in read_queries(public / "validation" / "queries.csv")}
        groups = defaultdict(list)
        with open(public / "validation" / "targets.csv", newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                p = np.array([float(r[c]) for c in ("p_1", "p_2", "p_5", "p_20")])
                if p.sum() > 1e-6:
                    groups[qtype[r["query_id"]]].append(p / p.sum())
        out = {}
        for typ, vecs in groups.items():
            vecs = np.array(vecs)
            clusters = []  # greedy clustering of near-identical vectors
            for v in vecs:
                for c in clusters:
                    if np.abs(c["sum"] / c["n"] - v).max() < 0.02:
                        c["sum"] += v
                        c["n"] += 1
                        break
                else:
                    clusters.append({"sum": v.copy(), "n": 1})
            keep = [c for c in clusters if c["n"] >= 0.05 * len(vecs)]
            if not keep:
                keep = [{"sum": vecs.sum(0), "n": len(vecs)}]
            keep.sort(key=lambda c: -(c["sum"] / c["n"])[0])  # profile 0 = most 1x-heavy
            out[typ] = [list(map(float, c["sum"] / c["n"])) for c in keep]
        return out

    # --- persistence -------------------------------------------------------------------
    def save(self, path):
        Path(path).write_text(json.dumps(self.cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    # --- prediction --------------------------------------------------------------------
    def predict_one(self, q, series):
        cfg = self.cfg
        cutoff = q["cutoff"]
        ticks, pay, win, lags, qlag = machine_arrays(series, cutoff)
        if len(ticks) == 0:
            pwin = 0.5
        else:
            m0 = float(np.clip(win.mean(), 0.05, 0.95))
            gaps = np.diff(np.concatenate([[ticks[0] - 1], ticks]))
            logs, preds = [], []
            ev, p = run_null(win, gaps, cfg["null_hazard"], cfg["null_a"], m0)
            logs.append(ev + math.log(cfg["null_prior"]))
            preds.append(p)
            for (i, j), prior in zip(cfg["pairs"], cfg["pair_prior"]):
                m, cells, g, qc = pair_inputs(ticks, lags, qlag, i, j)
                if m.sum() == 0:
                    continue
                ev, p = run_flip(self.grid, cells, win[m], g, qc, cfg["rho"])
                logs.append(ev + math.log(prior))
                preds.append(p)
            logs = np.array(logs)
            w = np.exp(logs - logs.max())
            w /= w.sum()
            pwin = float((w * np.array(preds)).sum())
        pwin = min(max(pwin, cfg["eps"]), 1.0 - cfg["eps"])
        profiles = cfg["size_profiles"].get(q["machine_type"])
        if profiles is None:
            profiles = [list(np.mean([v for vs in cfg["size_profiles"].values() for v in vs], axis=0))]
        mw = pay > 0
        size = size_distribution(pay[mw], cutoff - 1 - ticks[mw], profiles, cfg["size_half_life"])
        size = np.asarray(size, dtype=float)
        size = size / size.sum()
        probs = np.concatenate([[1.0 - pwin], pwin * size])
        probs = np.clip(probs, 1e-9, None)
        return probs / probs.sum()

    def predict(self, queries, hist):
        return {q["query_id"]: self.predict_one(q, hist.get((q["episode_id"], q["machine_id"]), {}))
                for q in queries}


DEFAULT_CONFIG = {
    "grid_size": 25,
    "rho": 0.002,
    "null_prior": 0.01,
    "null_hazard": 0.001,
    "null_a": 1.0,
    "eps": 0.002,
    "size_half_life": 60.0,
    "min_pair_share": 0.05,
}
