"""Cross-machine residual analysis on TRAIN folds 0-2 (fold 3 locked).
e_win = 1[y>0] - P(win);  e_big = 1[y>=5] - (p_5 + p_20).
(1) Same-tick correlation of a machine's residual with its neighbours' mean residual, per condition,
    with a permutation null (neighbour means taken from a different episode of the same condition).
(2) Predictability: ridge regression of e_win(t) on neighbour residuals at t-1..t-8 (usable at
    prediction time), cross-validated by fold (train on 2 folds, test on the 3rd), vs. a
    shuffled-neighbour null and an own-lag-only control."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import RidgeCV

HERE = Path(__file__).resolve().parent
CONDS = ["stable", "short_history", "drift", "shared_change"]
rows = [r for r in json.load(open(HERE / "residual_rows.json")) if r["y"] is not None]
rng = np.random.default_rng(0)

E = {}  # (episode, machine, tick) -> (e_win, e_big)
meta = {}
for r in rows:
    p = r["p"]
    E[(r["episode"], r["machine"], r["tick"])] = (float(r["y"] > 0) - (1 - p[0]), float(r["y"] >= 5) - (p[3] + p[4]))
    meta[(r["episode"], r["machine"])] = (r["cond"], r["fold"], r["type"])
ep_machines = defaultdict(list)
for (e, m), v in meta.items():
    ep_machines[e].append(m)
ep_cond = {e: meta[(e, ms[0])][0] for e, ms in ep_machines.items()}
ep_fold = {e: meta[(e, ms[0])][1] for e, ms in ep_machines.items()}
ep_ticks = defaultdict(set)
for (e, m, t) in E:
    ep_ticks[e].add(t)

out = {}
print("(1) SAME-TICK correlation: own residual vs mean of the 7 neighbours' residuals")
print(f"{'condition':15s}{'n':>7}{'corr_win':>10}{'null95':>9}{'p':>7}{'corr_big':>10}{'null95':>9}{'p':>7}   per-fold corr_win")
for c in CONDS:
    eps = [e for e in ep_machines if ep_cond[e] == c]
    def collect(nb_ep_of):
        a, b, ab, bb, folds = [], [], [], [], []
        for e in eps:
            ne = nb_ep_of[e]
            for t in ep_ticks[e]:
                for m in ep_machines[e]:
                    if (e, m, t) not in E:
                        continue
                    nbs = [E[(ne, n, t2)] for n in ep_machines[ne] if n != m
                           for t2 in [t - max(ep_ticks[e]) + max(ep_ticks[ne])] if (ne, n, t2) in E]
                    if not nbs:
                        continue
                    a.append(E[(e, m, t)][0]); b.append(np.mean([x[0] for x in nbs]))
                    ab.append(E[(e, m, t)][1]); bb.append(np.mean([x[1] for x in nbs])); folds.append(ep_fold[e])
        return np.array(a), np.array(b), np.array(ab), np.array(bb), np.array(folds)
    a, b, ab, bb, fo = collect({e: e for e in eps})
    cw, cb = np.corrcoef(a, b)[0, 1], np.corrcoef(ab, bb)[0, 1]
    nulls_w, nulls_b = [], []
    for _ in range(100):
        perm = rng.permutation(eps)
        mp = {e: (p if p != e else perm[(i + 1) % len(perm)]) for i, (e, p) in enumerate(zip(eps, perm))}
        x = collect(mp)
        nulls_w.append(np.corrcoef(x[0], x[1])[0, 1]); nulls_b.append(np.corrcoef(x[2], x[3])[0, 1])
    pw = np.mean(np.abs(nulls_w) >= abs(cw)); pb = np.mean(np.abs(nulls_b) >= abs(cb))
    perfold = [np.corrcoef(a[fo == f], b[fo == f])[0, 1] for f in (0, 1, 2)]
    print(f"{c:15s}{len(a):7d}{cw:10.4f}{np.percentile(np.abs(nulls_w), 95):9.4f}{pw:7.2f}{cb:10.4f}"
          f"{np.percentile(np.abs(nulls_b), 95):9.4f}{pb:7.2f}   {[round(v, 4) for v in perfold]}")
    out.setdefault("same_tick", {})[c] = {"corr_win": cw, "corr_big": cb, "p_win": pw, "p_big": pb,
                                         "per_fold_corr_win": perfold}

print("\n(2) LAGGED predictability of e_win(t), ridge, CV by fold (fit on 2 folds, score the 3rd)")
print("    R2 is out-of-sample vs predicting 0. Features: neighbour mean e_win & e_big at lags 1-8 (+ same-type mean).")


def features(e, m, t, nb_map=None, own=False):
    ne = e if nb_map is None else nb_map[e]
    shift = 0 if nb_map is None else max(ep_ticks[ne]) - max(ep_ticks[e])
    typ = meta[(e, m)][2]
    f = []
    for k in range(1, 9):
        if own:
            v = E.get((e, m, t - k)); f += [v[0] if v else 0.0, v[1] if v else 0.0]; continue
        nbs = [E[(ne, n, t - k + shift)] for n in ep_machines[ne] if n != m and (ne, n, t - k + shift) in E]
        st = [E[(ne, n, t - k + shift)] for n in ep_machines[ne]
              if n != m and meta[(ne, n)][2] == typ and (ne, n, t - k + shift) in E]
        f += [np.mean([x[0] for x in nbs]) if nbs else 0.0, np.mean([x[1] for x in nbs]) if nbs else 0.0,
              np.mean([x[0] for x in st]) if st else 0.0]
    return f


for c in CONDS:
    eps = [e for e in ep_machines if ep_cond[e] == c]
    perm = list(rng.permutation(eps)); nb_null = {e: perm[(i + 1) % len(perm)] if perm[i] == e else perm[i] for i, e in enumerate(eps)}
    data = {"neighbour": [], "null": [], "own": []}
    for e in eps:
        lo = min(ep_ticks[e]) + 8
        for t in ep_ticks[e]:
            if t < lo:
                continue
            for m in ep_machines[e]:
                if (e, m, t) not in E:
                    continue
                y = E[(e, m, t)][0]; fo = ep_fold[e]
                data["neighbour"].append((features(e, m, t), y, fo))
                data["null"].append((features(e, m, t, nb_null), y, fo))
                data["own"].append((features(e, m, t, own=True), y, fo))
    line = f"{c:15s}"
    for name, d in data.items():
        X = np.array([x for x, _, _ in d]); Y = np.array([y for _, y, _ in d]); F = np.array([f for _, _, f in d])
        r2s = []
        for f in (0, 1, 2):
            tr, te = F != f, F == f
            mdl = RidgeCV(alphas=np.logspace(-1, 4, 12)).fit(X[tr], Y[tr])
            r2s.append(1 - ((Y[te] - mdl.predict(X[te])) ** 2).sum() / (Y[te] ** 2).sum())
        out.setdefault("lagged_r2", {}).setdefault(c, {})[name] = r2s
        line += f"  {name}: R2={np.mean(r2s):+.4f}±{np.std(r2s):.4f} {[round(v, 4) for v in r2s]}"
    print(line + f"  (n={len(data['neighbour'])})")
json.dump(out, open(HERE / "c_analyze.json", "w"), indent=1, default=float)
