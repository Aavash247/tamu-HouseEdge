"""Payout size given a win: a mixture of each type's fixed size profiles."""
import numpy as np

WIN_PAYOUTS = (1, 2, 5, 20)
MIX_GRID = np.linspace(0.005, 0.995, 100)


def size_distribution(pays, ages, profiles, half_life):
    """Posterior-mean profile mixture given past winning payouts, newer ones weighted more."""
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
        ll = (w[:, None] * np.log(MIX_GRID[None, :] * lb[:, None]
                                  + (1 - MIX_GRID[None, :]) * la[:, None])).sum(0)
        p = np.exp(ll - ll.max())
        p /= p.sum()
        pi = float((p * MIX_GRID).sum())
        return pi * prof[1] + (1 - pi) * prof[0]
    # more than two profiles: weighted EM with a flat Dirichlet prior
    pi = np.full(k, 1.0 / k)
    if len(pays):
        lik = prof[:, idx].T
        for _ in range(50):
            r = lik * pi
            r /= r.sum(1, keepdims=True)
            pi = ((w[:, None] * r).sum(0) + 1.0) / (w.sum() + k)
    return pi @ prof


def fit_profiles(groups, tol=0.02, min_share=0.05):
    """Groups near-identical size vectors per type; profile 0 is the most 1x-heavy."""
    out = {}
    for typ, vecs in groups.items():
        vecs = np.array(vecs)
        clusters = []
        for v in vecs:
            for c in clusters:
                if np.abs(c["sum"] / c["n"] - v).max() < tol:
                    c["sum"] += v
                    c["n"] += 1
                    break
            else:
                clusters.append({"sum": v.copy(), "n": 1})
        keep = [c for c in clusters if c["n"] >= min_share * len(vecs)]
        if not keep:
            keep = [{"sum": vecs.sum(0), "n": len(vecs)}]
        keep.sort(key=lambda c: -(c["sum"] / c["n"])[0])
        out[typ] = [list(map(float, c["sum"] / c["n"])) for c in keep]
    return out
