"""Win chance models: lag-pair tables with an orientation flip HMM, plus a no-lag fallback."""
import itertools
import math

import numpy as np

FOUR_LEVELS = (0.01, 0.04, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.96, 0.99)
FLIP_B = (1, 0, 3, 2)  # a flip maps cell (a, b) to (a, 1 - b); cell = 2a + b


class FlipGrid:
    """Rate grid for the 2-cell model: P(win | lags agree), P(win | lags differ)."""

    def __init__(self, n=25):
        g = np.linspace(0.01, 0.99, n)
        a, b = np.meshgrid(g, g, indexing="ij")
        self.q_same = a.ravel()
        self.q_diff = b.ravel()
        self.swap = np.arange(n * n).reshape(n, n).T.ravel()


class FourGrid:
    """Rate grid for the full 2x2 lag table."""

    def __init__(self, levels=FOUR_LEVELS):
        lev = np.array(levels)
        idx = np.array(list(itertools.product(range(len(lev)), repeat=4)))
        self.rates = lev[idx]  # K x 4, cells (0,0) (0,1) (1,0) (1,1)
        self.k = len(idx)
        cube = np.arange(self.k).reshape([len(lev)] * 4)
        self.swap = np.transpose(cube, FLIP_B).ravel()


def _flip(belief, rho, gap):
    """Orientation belief after `gap` ticks of flip rate rho."""
    f = (1.0 - (1.0 - 2.0 * rho) ** gap) / 2.0
    return belief * (1.0 - f) + (1.0 - belief) * f


def _posterior(ll, logprior, k):
    """Log evidence and normalized grid weights."""
    ll = ll + (logprior if logprior is not None else -math.log(k))
    mx = ll.max()
    w = np.exp(ll - mx)
    ev = mx + math.log(w.sum())
    w /= w.sum()
    return ev, w


def run_flip(grid, cells, ys, gaps, qcell, rho, logprior=None):
    """2-cell model. Returns (log evidence, P(win) for the query, grid weights)."""
    qs, qd = grid.q_same, grid.q_diff
    b = np.full(len(qs), 0.5)
    ll = np.zeros(len(qs))
    for c, y, g in zip(cells, ys, gaps):
        b = _flip(b, rho, g)
        p0 = qs if c == 1 else qd
        p1 = qd if c == 1 else qs
        l0 = p0 if y else 1.0 - p0
        l1 = p1 if y else 1.0 - p1
        tot = b * l0 + (1.0 - b) * l1
        ll += np.log(tot)
        b = b * l0 / tot
    b = b * (1.0 - rho) + (1.0 - b) * rho
    ev, w = _posterior(ll, logprior, len(qs))
    if qcell is None:
        pq = 0.5 * (qs + qd)
    elif qcell == 1:
        pq = b * qs + (1.0 - b) * qd
    else:
        pq = b * qd + (1.0 - b) * qs
    return ev, float((w * pq).sum()), w


def run_flip4(grid, a_vals, b_vals, ys, gaps, qa, qb, rho, logprior=None):
    """Full 2x2 model. Returns (log evidence, P(win) for the query, grid weights)."""
    r = grid.rates
    bel = np.full(grid.k, 0.5)
    ll = np.zeros(grid.k)
    for a, b, y, g in zip(a_vals, b_vals, ys, gaps):
        bel = _flip(bel, rho, g)
        c = 2 * int(a) + int(b)
        p0 = r[:, c]
        p1 = r[:, FLIP_B[c]]
        if not y:
            p0 = 1.0 - p0
            p1 = 1.0 - p1
        tot = bel * p0 + (1.0 - bel) * p1
        ll += np.log(tot)
        bel = bel * p0 / tot
    bel = bel * (1.0 - rho) + (1.0 - bel) * rho
    ev, w = _posterior(ll, logprior, grid.k)
    if np.isnan(qa) or np.isnan(qb):
        pq = r.mean(1)
    else:
        c = 2 * int(qa) + int(qb)
        pq = bel * r[:, c] + (1.0 - bel) * r[:, FLIP_B[c]]
    return ev, float((w * pq).sum()), w


def run_null(ys, gaps, hazard, a, m0):
    """One win rate with Bayesian online change points. Returns (log evidence, P(win))."""
    log_r = np.array([0.0])
    n = np.zeros(1)
    s = np.zeros(1)
    ev = 0.0
    for y, g in zip(ys, gaps):
        hz = 1.0 - (1.0 - hazard) ** g
        r = np.exp(log_r - log_r.max())
        r /= r.sum()
        r = np.concatenate([[hz], r * (1.0 - hz)])
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
    """Rows where both lags are seen, agree flags, tick gaps and the query's agree flag."""
    m = ~np.isnan(lags[:, i]) & ~np.isnan(lags[:, j])
    tk = ticks[m]
    gaps = np.diff(np.concatenate([[tk[0] - 1], tk])) if len(tk) else np.array([], dtype=int)
    cells = (lags[m, i] == lags[m, j]).astype(int)
    qc = None if (np.isnan(qlag[i]) or np.isnan(qlag[j])) else int(qlag[i] == qlag[j])
    return m, cells, gaps, qc
