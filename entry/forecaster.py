"""Fits and runs the forecaster. Settings and learned values live in model.json."""
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from data import machine_arrays, read_history, read_queries, read_size_targets
from size import fit_profiles, size_distribution
from win import FlipGrid, FourGrid, pair_inputs, run_flip, run_flip4, run_null

ALL_PAIRS = list(itertools.combinations(range(1, 9), 2))
MIN_TICKS = 20

DEFAULT_CONFIG = {
    "grid_size": 25,
    "rho": 0.001,               # orientation flip rate per tick
    "prior_mix": 0.9,           # learned rate prior vs flat
    "family_weights": {"2": 0.5, "4": 0.5},
    "null_prior": 0.01,         # prior on the no-lag model
    "null_hazard": 0.001,
    "null_a": 1.0,
    "eps": 0.002,               # win chance clipped to [eps, 1 - eps]
    "size_half_life": 60.0,
    "min_pair_share": 0.05,     # pairs best for fewer train machines are dropped
}


class Forecaster:
    def __init__(self, config):
        self.cfg = config
        self.grid = FlipGrid(config["grid_size"])
        self.grid4 = FourGrid()
        self.prior2 = {t: np.array(v) for t, v in config.get("prior2", {}).items()}
        self.prior4 = {t: np.array(v) for t, v in config.get("prior4", {}).items()}

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path):
        Path(path).write_text(json.dumps(self.cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # fitting

    @classmethod
    def fit(cls, public, config):
        """Lag pairs and rate priors from train histories, size profiles from validation targets."""
        public = Path(public)
        cfg = dict(config)
        grid = FlipGrid(cfg["grid_size"])
        queries = read_queries(public / "train" / "queries.csv")
        hist = read_history(public / "train" / "history.csv")
        cfg["pairs"], cfg["pair_prior"] = _best_pairs(queries, hist, cfg, grid)
        cfg["size_profiles"] = fit_profiles(read_size_targets(public))
        cfg["prior2"], cfg["prior4"] = _rate_priors(queries, hist, cfg, grid)
        return cls(cfg)

    # prediction

    def predict(self, queries, hist):
        return {q["query_id"]: self.predict_one(q, hist.get((q["episode_id"], q["machine_id"]), {}))
                for q in queries}

    def predict_one(self, q, series):
        cfg = self.cfg
        ticks, pay, win, lags, qlag = machine_arrays(series, q["cutoff"])
        pwin = self._win_chance(q["machine_type"], ticks, win, lags, qlag) if len(ticks) else 0.5
        pwin = min(max(pwin, cfg["eps"]), 1.0 - cfg["eps"])
        wins = pay > 0
        size = size_distribution(pay[wins], q["cutoff"] - 1 - ticks[wins],
                                 self._profiles(q["machine_type"]), cfg["size_half_life"])
        size = np.asarray(size, dtype=float)
        size = size / size.sum()
        probs = np.clip(np.concatenate([[1.0 - pwin], pwin * size]), 1e-9, None)
        return probs / probs.sum()

    def _win_chance(self, machine_type, ticks, win, lags, qlag):
        """Bayesian average over the no-lag model and every lag pair, per family, then blended."""
        cfg = self.cfg
        m0 = float(np.clip(win.mean(), 0.05, 0.95))
        gaps = np.diff(np.concatenate([[ticks[0] - 1], ticks]))
        null_ev, null_p = run_null(win, gaps, cfg["null_hazard"], cfg["null_a"], m0)
        null_log = null_ev + math.log(cfg["null_prior"])
        lp2 = _type_prior(self.prior2, machine_type)
        lp4 = _type_prior(self.prior4, machine_type)
        fam = {"2": ([null_log], [null_p]), "4": ([null_log], [null_p])}
        for (i, j), prior in zip(cfg["pairs"], cfg["pair_prior"]):
            m, cells, g, qc = pair_inputs(ticks, lags, qlag, i, j)
            if m.sum() == 0:
                continue
            ev, p, _ = run_flip(self.grid, cells, win[m], g, qc, cfg["rho"], logprior=lp2)
            fam["2"][0].append(ev + math.log(prior))
            fam["2"][1].append(p)
            ev, p, _ = run_flip4(self.grid4, lags[m, i], lags[m, j], win[m], g, qlag[i], qlag[j],
                                 cfg["rho"], logprior=lp4)
            fam["4"][0].append(ev + math.log(prior))
            fam["4"][1].append(p)
        pwin = 0.0
        for name, weight in cfg["family_weights"].items():
            logs = np.array(fam[name][0])
            w = np.exp(logs - logs.max())
            w /= w.sum()
            pwin += weight * float((w * np.array(fam[name][1])).sum())
        return pwin

    def _profiles(self, machine_type):
        profiles = self.cfg["size_profiles"]
        if machine_type in profiles:
            return profiles[machine_type]
        return [list(np.mean([v for vs in profiles.values() for v in vs], axis=0))]


def _type_prior(priors, machine_type):
    """The type's log prior; an unseen type gets the average of the known ones."""
    if not priors:
        return None
    if machine_type in priors:
        return priors[machine_type]
    return np.logaddexp.reduce(np.array(list(priors.values())), axis=0) - math.log(len(priors))


def _best_pairs(queries, hist, cfg, grid):
    """Pairs that best explain enough train machines, and their shares as a prior."""
    counts = defaultdict(float)
    for q in queries:
        ticks, _, win, lags, qlag = machine_arrays(hist[(q["episode_id"], q["machine_id"])], q["cutoff"])
        if len(ticks) < MIN_TICKS:
            continue
        evs = {}
        for i, j in ALL_PAIRS:
            m, cells, gaps, qc = pair_inputs(ticks, lags, qlag, i, j)
            if m.sum() < 10:
                continue
            evs[(i, j)] = run_flip(grid, cells, win[m], gaps, qc, cfg["rho"])[0]
        if evs:
            counts[max(evs, key=evs.get)] += 1.0
    total = sum(counts.values())
    pairs = sorted(p for p, c in counts.items() if c / total >= cfg["min_pair_share"])
    kept = sum(counts[p] for p in pairs)
    return [list(p) for p in pairs], [counts[p] / kept for p in pairs]


def _rate_priors(queries, hist, cfg, grid):
    """Empirical-Bayes log priors over each rate grid, per machine type, symmetric under a flip."""
    grid4 = FourGrid()
    post2, post4 = defaultdict(list), defaultdict(list)
    for q in queries:
        ticks, _, win, lags, qlag = machine_arrays(hist[(q["episode_id"], q["machine_id"])], q["cutoff"])
        if len(ticks) < MIN_TICKS:
            continue
        best2 = best4 = None
        for i, j in cfg["pairs"]:
            m, cells, g, qc = pair_inputs(ticks, lags, qlag, i, j)
            if m.sum() == 0:
                continue
            ev, _, w = run_flip(grid, cells, win[m], g, qc, cfg["rho"])
            if best2 is None or ev > best2[0]:
                best2 = (ev, w)
            ev, _, w = run_flip4(grid4, lags[m, i], lags[m, j], win[m], g, qlag[i], qlag[j], cfg["rho"])
            if best4 is None or ev > best4[0]:
                best4 = (ev, w)
        if best2 is not None:
            post2[q["machine_type"]].append(0.5 * (best2[1] + best2[1][grid.swap]))
            post4[q["machine_type"]].append(0.5 * (best4[1] + best4[1][grid4.swap]))
    mix = cfg["prior_mix"]
    n2 = cfg["grid_size"] ** 2
    prior2 = {t: list(map(float, np.log(mix * np.mean(v, 0) + (1 - mix) / n2))) for t, v in post2.items()}
    prior4 = {t: list(map(float, np.log(mix * np.mean(v, 0) + (1 - mix) / grid4.k))) for t, v in post4.items()}
    return prior2, prior4
