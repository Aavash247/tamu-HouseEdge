"""Per-machine model components for ALL 28 lag pairs, so any pair-prior epsilon can be combined later.

combine(comp, cfg, eps) reproduces entry-final Forecaster.predict_one exactly when eps == 0:
  learned pairs keep prior (1 - eps) * train_prior; the other 25 pairs share eps equally.
power: type-level rate prior raised to this power (2.1b); used only for short-history (n < 100) rows.
"""
import math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "entry-final"))
import houseedge as he  # noqa: E402

SHORT_N = 100


def _temper(lp, power):
    if lp is None or power == 1.0:
        return lp
    x = lp * power
    return x - np.logaddexp.reduce(x)


def components(model, q, series, power=1.0):
    cfg = model.cfg
    cutoff = q["cutoff"]
    ticks, pay, win, lags, qlag = he.machine_arrays(series, cutoff)
    out = {"n": len(ticks)}
    if len(ticks) == 0:
        out["empty"] = True
    else:
        m0 = float(np.clip(win.mean(), 0.05, 0.95))
        gaps = np.diff(np.concatenate([[ticks[0] - 1], ticks]))
        out["null"] = he.run_null(win, gaps, cfg["null_hazard"], cfg["null_a"], m0)
        pw = power if len(ticks) < SHORT_N else 1.0
        lp2 = _temper(model._type_prior(model.prior2, q["machine_type"]), pw)
        lp4 = _temper(model._type_prior(model.prior4, q["machine_type"]), pw)
        f2, f4 = {}, {}
        for (i, j) in he.ALL_PAIRS:
            mk, cells, g, qc = he.pair_inputs(ticks, lags, qlag, i, j)
            if mk.sum() == 0:
                continue
            f2[f"{i},{j}"] = he.run_flip(model.grid, cells, win[mk], g, qc, cfg["rho"], logprior=lp2)
            f4[f"{i},{j}"] = he.run_flip4(model.grid4, lags[mk, i], lags[mk, j], win[mk], g, qlag[i], qlag[j],
                                          cfg["rho"], logprior=lp4)
        out["f2"], out["f4"] = f2, f4
    profiles = cfg["size_profiles"].get(q["machine_type"])
    if profiles is None:
        profiles = [list(np.mean([v for vs in cfg["size_profiles"].values() for v in vs], axis=0))]
    mw = pay > 0
    size = np.asarray(he.size_distribution(pay[mw], cutoff - 1 - ticks[mw], profiles, cfg["size_half_life"]), float)
    out["size"] = (size / size.sum()).tolist()
    return out


def combine(comp, cfg, eps):
    learned = {f"{i},{j}": p for (i, j), p in zip(cfg["pairs"], cfg["pair_prior"])}
    others = [f"{i},{j}" for (i, j) in he.ALL_PAIRS if f"{i},{j}" not in learned]
    if comp.get("empty"):
        pwin = 0.5
    else:
        pwin = 0.0
        for name, weight in cfg["family_weights"].items():
            fam = comp["f" + name]
            logs = [comp["null"][0] + math.log(cfg["null_prior"])]
            ps = [comp["null"][1]]
            for key, prior in learned.items():
                if key in fam:
                    logs.append(fam[key][0] + math.log(prior * (1 - eps))); ps.append(fam[key][1])
            if eps > 0:
                for key in others:
                    if key in fam:
                        logs.append(fam[key][0] + math.log(eps / len(others))); ps.append(fam[key][1])
            logs = np.array(logs); w = np.exp(logs - logs.max()); w /= w.sum()
            pwin += weight * float((w * np.array(ps)).sum())
    pwin = min(max(pwin, cfg["eps"]), 1.0 - cfg["eps"])
    probs = np.concatenate([[1.0 - pwin], pwin * np.asarray(comp["size"])])
    probs = np.clip(probs, 1e-9, None)
    return probs / probs.sum()
