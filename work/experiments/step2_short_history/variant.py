"""Variant predictor for Step 2. Replicates entry-final Forecaster.predict_one exactly when no option is
on (checked in d_cv.py), and adds options that apply ONLY to short-history machines, detected from a
prediction-time input: number of observed ticks n < 100 (n == 64 exactly at test time; long histories
have n >= 190).

Options
  marg         2.0-fix: use every observed tick in the pair likelihoods; an unobserved lag outcome is
               marginalised with the machine's win rate instead of dropping the row.
  all_pairs    2.1a: average over all 28 lag pairs; prior = (train best-pair count + alpha) normalised.
  prior_power  2.1b: posterior uses the type-level rate prior raised to this power (strength of pooling).
  null_prior   override of the no-lag model's prior weight for short-history machines.
"""
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "entry-final"))
import houseedge as he  # noqa: E402

SHORT_N = 100


def _cell_probs_2(a, b, m):
    """P(lags agree) for each row; a/b may be NaN (unobserved)."""
    pa1 = np.where(np.isnan(a), m, a); pb1 = np.where(np.isnan(b), m, b)
    return pa1 * pb1 + (1 - pa1) * (1 - pb1)


def run_flip_marg(grid, a, b, ys, gaps, qcell, rho, m, logprior=None):
    qs, qd = grid.q_same, grid.q_diff
    k = len(qs); bel = np.full(k, 0.5); ll = np.zeros(k)
    ps_all = _cell_probs_2(a, b, m)
    for ps, y, g in zip(ps_all, ys, gaps):
        f = (1 - (1 - 2 * rho) ** g) / 2; bel = bel * (1 - f) + (1 - bel) * f
        p0 = ps * qs + (1 - ps) * qd; p1 = ps * qd + (1 - ps) * qs
        if not y:
            p0 = 1 - p0; p1 = 1 - p1
        tot = bel * p0 + (1 - bel) * p1; ll += np.log(tot); bel = bel * p0 / tot
    bel = bel * (1 - rho) + (1 - bel) * rho
    ll = ll + (logprior if logprior is not None else -math.log(k))
    mx = ll.max(); w = np.exp(ll - mx); ev = mx + math.log(w.sum()); w /= w.sum()
    if qcell is None:
        pq = 0.5 * (qs + qd)
    elif qcell == 1:
        pq = bel * qs + (1 - bel) * qd
    else:
        pq = bel * qd + (1 - bel) * qs
    return ev, float((w * pq).sum())


def run_flip4_marg(grid, a, b, ys, gaps, qa, qb, rho, m, logprior=None):
    r = grid.rates; k = grid.k; bel = np.full(k, 0.5); ll = np.zeros(k)
    rf = r[:, list(he.FLIP_B)]
    pa1 = np.where(np.isnan(a), m, a); pb1 = np.where(np.isnan(b), m, b)
    for x, z, y, g in zip(pa1, pb1, ys, gaps):
        f = (1 - (1 - 2 * rho) ** g) / 2; bel = bel * (1 - f) + (1 - bel) * f
        pc = np.array([(1 - x) * (1 - z), (1 - x) * z, x * (1 - z), x * z])
        p0 = r @ pc; p1 = rf @ pc
        if not y:
            p0 = 1 - p0; p1 = 1 - p1
        tot = bel * p0 + (1 - bel) * p1; ll += np.log(tot); bel = bel * p0 / tot
    bel = bel * (1 - rho) + (1 - bel) * rho
    ll = ll + (logprior if logprior is not None else -math.log(k))
    mx = ll.max(); w = np.exp(ll - mx); ev = mx + math.log(w.sum()); w /= w.sum()
    if np.isnan(qa) or np.isnan(qb):
        pq = r.mean(1)
    else:
        c = 2 * int(qa) + int(qb); pq = bel * r[:, c] + (1 - bel) * r[:, he.FLIP_B[c]]
    return ev, float((w * pq).sum())


def _temper(lp, power):
    if lp is None or power == 1.0:
        return lp
    x = lp * power
    return x - np.logaddexp.reduce(x)


def predict_one(model, q, series, opts=None):
    opts = opts or {}
    cfg = model.cfg
    cutoff = q["cutoff"]
    ticks, pay, win, lags, qlag = he.machine_arrays(series, cutoff)
    short = len(ticks) < SHORT_N
    o = opts if short else {}
    if len(ticks) == 0:
        pwin = 0.5
    else:
        m0 = float(np.clip(win.mean(), 0.05, 0.95))
        gaps_all = np.diff(np.concatenate([[ticks[0] - 1], ticks]))
        null_ev, null_p = he.run_null(win, gaps_all, cfg["null_hazard"], cfg["null_a"], m0)
        lp2 = _temper(model._type_prior(model.prior2, q["machine_type"]), o.get("prior_power", 1.0))
        lp4 = _temper(model._type_prior(model.prior4, q["machine_type"]), o.get("prior_power", 1.0))
        nprior = o.get("null_prior", cfg["null_prior"])
        fam = {"2": ([null_ev + math.log(nprior)], [null_p]), "4": ([null_ev + math.log(nprior)], [null_p])}
        if o.get("all_pairs"):
            counts = cfg["pair_counts"]; alpha = o.get("pair_alpha", 0.5)
            tot = sum(counts.get(f"{i},{j}", 0.0) + alpha for (i, j) in he.ALL_PAIRS)
            pairs = [([i, j], (counts.get(f"{i},{j}", 0.0) + alpha) / tot) for (i, j) in he.ALL_PAIRS]
        else:
            pairs = list(zip(cfg["pairs"], cfg["pair_prior"]))
        for (i, j), prior in pairs:
            if o.get("marg"):
                a = lags[:, i]
                b = lags[:, j]
                qa, qb = qlag[i], qlag[j]
                qc = None if (np.isnan(qa) or np.isnan(qb)) else int(qa == qb)
                ev, p = run_flip_marg(model.grid, a, b, win, gaps_all, qc, cfg["rho"], m0, lp2)
                fam["2"][0].append(ev + math.log(prior)); fam["2"][1].append(p)
                ev, p = run_flip4_marg(model.grid4, a, b, win, gaps_all, qa, qb, cfg["rho"], m0, lp4)
                fam["4"][0].append(ev + math.log(prior)); fam["4"][1].append(p)
                continue
            mk, cells, g, qc = he.pair_inputs(ticks, lags, qlag, i, j)
            if mk.sum() == 0:
                continue
            ev, p = he.run_flip(model.grid, cells, win[mk], g, qc, cfg["rho"], logprior=lp2)
            fam["2"][0].append(ev + math.log(prior)); fam["2"][1].append(p)
            ev, p = he.run_flip4(model.grid4, lags[mk, i], lags[mk, j], win[mk], g, qlag[i], qlag[j],
                                 cfg["rho"], logprior=lp4)
            fam["4"][0].append(ev + math.log(prior)); fam["4"][1].append(p)
        pwin = 0.0
        for name, weight in cfg["family_weights"].items():
            logs = np.array(fam[name][0]); w = np.exp(logs - logs.max()); w /= w.sum()
            pwin += weight * float((w * np.array(fam[name][1])).sum())
    pwin = min(max(pwin, cfg["eps"]), 1.0 - cfg["eps"])
    profiles = cfg["size_profiles"].get(q["machine_type"])
    if profiles is None:
        profiles = [list(np.mean([v for vs in cfg["size_profiles"].values() for v in vs], axis=0))]
    mw = pay > 0
    size = np.asarray(he.size_distribution(pay[mw], cutoff - 1 - ticks[mw], profiles, cfg["size_half_life"]), float)
    size = size / size.sum()
    probs = np.concatenate([[1.0 - pwin], pwin * size])
    probs = np.clip(probs, 1e-9, None)
    return probs / probs.sum()
