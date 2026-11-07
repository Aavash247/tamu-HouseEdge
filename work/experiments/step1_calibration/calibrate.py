"""STEP 1 — post-processing calibration of the unchanged base model.

Evaluation rows: work/experiments/step0_diagnosis/residual_rows.json — base-model predictions for every
machine in TRAIN folds 0-2 at the last 25 ticks before cutoff, from models fitted on the OTHER two
unlocked folds (fold 3 locked, never used). Score: Brier vs observed outcome (differences between
methods are unbiased estimates of differences in squared error to the true probabilities).

Variants (settings depend only on prediction-time inputs: history length n, machine type):
  a) shrink:  p = (1-w) p_model + w p_base,  w = k/(k+n)   (p_base = pooled train payout distribution)
  b) floor:   p = max(p, f) per outcome, renormalised
  c) temp:    p ∝ p^(1/T)
Selection is nested: for each held-out fold f, the parameter is chosen on the other two folds.
"""
import csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE.parents[0] / "step0_diagnosis"))
from folds import fold_map, CONDS, LOCKED_FOLD  # noqa: E402
from slot_forecasting.schema import Query, Play  # noqa: E402
from slot_forecasting.baseline import FrequencyForecaster  # noqa: E402

PAY = (0, 1, 2, 5, 20)
GRIDS = {
    "shrink_global": [0, 1, 2, 5, 10, 20, 40, 80, 160],
    "shrink_type": [0, 1, 2, 5, 10, 20, 40, 80, 160],
    "floor": [0, 1e-4, 3e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2],
    "temp": [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.35, 1.5],
}


def load_train():
    q = list(csv.DictReader(open(ROOT / "public/train/queries.csv")))
    hist = defaultdict(dict)
    for r in csv.DictReader(open(ROOT / "public/train/history.csv")):
        hist[(r["episode_id"], r["machine_id"])][int(r["tick"])] = int(r["payout"])
    labels = {r["query_id"]: int(r["outcome"]) for r in csv.DictReader(open(ROOT / "public/train/labels.csv"))}
    return q, hist, labels


def base_rates(queries, hist, eps):
    tot = np.zeros(5); typ = defaultdict(lambda: np.zeros(5))
    for q in queries:
        if q["episode_id"] not in eps:
            continue
        for t, v in hist[(q["episode_id"], q["machine_id"])].items():
            if t < int(q["cutoff"]):
                tot[PAY.index(v)] += 1; typ[q["machine_type"]][PAY.index(v)] += 1
    return tot / tot.sum(), {k: v / v.sum() for k, v in typ.items()}


def apply(kind, theta, P, n, pb):
    if kind.startswith("shrink"):
        w = (theta / (theta + n))[:, None]
        return (1 - w) * P + w * pb
    if kind == "floor":
        X = np.maximum(P, theta); return X / X.sum(1, keepdims=True)
    if kind == "temp":
        L = np.log(np.clip(P, 1e-12, 1)) / theta
        X = np.exp(L - L.max(1, keepdims=True)); return X / X.sum(1, keepdims=True)
    if kind == "base":
        return P
    raise ValueError(kind)


def apply_combo(params, P, n, pb):
    X = P
    for kind, theta in params:
        X = apply(kind, theta, X, n, pb)
    return X


def brier(P, Y):
    return ((P - Y) ** 2).sum(1)


def ep_cond_mean(vals, eps, conds, cond=None):
    d = defaultdict(list)
    for v, e, c in zip(vals, eps, conds):
        if cond is None or c == cond:
            d[e].append(v)
    return np.mean([np.mean(x) for x in d.values()])


def overall(vals, eps, conds):  # equal condition weight, episode-weighted inside (official structure)
    return np.mean([ep_cond_mean(vals, eps, conds, c) for c in CONDS])


def main():
    queries, hist, labels = load_train()
    fm, ep_cond = fold_map("train")
    rows = [r for r in json.load(open(HERE.parents[0] / "step0_diagnosis/residual_rows.json")) if r["y"] is not None]
    qinfo = {q["machine_id"]: q for q in queries}
    P = np.array([r["p"] for r in rows])
    Y = np.array([[1.0 if r["y"] == v else 0.0 for v in PAY] for r in rows])
    n = np.array([sum(1 for t in hist[(r["episode"], r["machine"])] if t < r["tick"]) for r in rows], float)
    F = np.array([r["fold"] for r in rows]); E = np.array([r["episode"] for r in rows]); C = np.array([r["cond"] for r in rows])
    T = [r["type"] for r in rows]
    assert LOCKED_FOLD not in set(F)

    # pooled reference (competition reference: pooled, strength 20) per fold, trained on the other unlocked folds
    REF = np.zeros_like(P)
    PB_G = np.zeros_like(P); PB_T = np.zeros_like(P)
    for f in (0, 1, 2):
        train_eps = {e for e, k in fm.items() if k not in (f, LOCKED_FOLD)}
        g, bt = base_rates(queries, hist, train_eps)
        tq = [q for q in queries if q["episode_id"] in train_eps]
        sq = tuple(Query(q["query_id"], q["episode_id"], q["machine_id"], q["machine_type"], int(q["signal"]), int(q["cutoff"])) for q in tq)
        sh = tuple(Play(q["episode_id"], q["machine_id"], t, v) for q in tq for t, v in hist[(q["episode_id"], q["machine_id"])].items())
        ref = FrequencyForecaster("pooled", prior_strength=20.0).fit(sq, sh, {q["query_id"]: labels[q["query_id"]] for q in tq})
        idx = np.where(F == f)[0]
        for i in idx:
            r = rows[i]; PB_G[i] = g; PB_T[i] = bt[r["type"]]
            q = qinfo[r["machine"]]
            qq = Query(f'{q["query_id"]}-t{r["tick"]}', r["episode"], r["machine"], r["type"], int(q["signal"]), r["tick"])
            plays = tuple(Play(r["episode"], r["machine"], t, v) for t, v in hist[(r["episode"], r["machine"])].items() if t < r["tick"])
            REF[i] = list(ref.predict((qq,), plays).values())[0]
    base_b = brier(P, Y); ref_b = brier(REF, Y)
    np.savez(HERE / "eval_arrays.npz", P=P, Y=Y, n=n, F=F, E=E, C=C, REF=REF, PB_G=PB_G, PB_T=PB_T)

    def pb_for(kind):
        return PB_T if kind == "shrink_type" else PB_G

    def fit(kinds_fixed, kind, mask):
        best = None
        for th in GRIDS[kind]:
            X = apply_combo(kinds_fixed + [(kind, th)], P[mask], n[mask], pb_for(kind)[mask])
            s = overall(brier(X, Y[mask]), E[mask], C[mask])
            if best is None or s < best[0]:
                best = (s, th)
        return best[1]

    results = {}

    def evaluate(name, chooser):
        """chooser(mask) -> list of (kind, theta); nested: chosen on the other two folds."""
        per_fold = {}
        chosen = {}
        newb = np.zeros(len(rows))
        for f in (0, 1, 2):
            params = chooser(F != f)
            chosen[f] = params
            m = F == f
            pb = pb_for(params[-1][0] if params else "x")[m] if params else PB_G[m]
            X = P[m]
            for kind, th in params:
                X = apply(kind, th, X, n[m], pb_for(kind)[m])
            newb[m] = brier(X, Y[m])
        full = chooser(F >= 0)
        out = {"chosen_per_fold": {str(k): v for k, v in chosen.items()}, "chosen_full": full, "folds": {}}
        for f in (0, 1, 2):
            m = F == f
            out["folds"][f] = {
                "delta_vs_base": overall(base_b[m] - newb[m], E[m], C[m]),
                "gain_vs_ref": overall(ref_b[m] - newb[m], E[m], C[m]),
                "cond_delta": {c: ep_cond_mean(base_b[m] - newb[m], E[m], C[m], c) for c in CONDS},
            }
        results[name] = out
        return out

    evaluate("base", lambda m: [])
    for kind in ("shrink_global", "shrink_type", "floor", "temp"):
        evaluate(kind, lambda m, kind=kind: [(kind, fit([], kind, m))])

    def combo_chooser(m):
        params = []
        for kind in ("shrink_type", "temp", "floor"):  # greedy, each fitted given the previous ones
            th = fit(params, kind, m)
            params.append((kind, th))
        return params
    evaluate("combo_shrink_type+temp+floor", combo_chooser)

    json.dump(results, open(HERE / "cv_results.json", "w"), indent=1, default=float)
    # report
    print("Train CV, folds 0-2 (fold 3 LOCKED). Brier vs outcomes on 25 ticks/machine; "
          "delta = base - variant (positive = better). Parameters chosen on the other two folds.\n")
    for name, r in results.items():
        d = [r["folds"][f]["delta_vs_base"] for f in (0, 1, 2)]
        g = [r["folds"][f]["gain_vs_ref"] for f in (0, 1, 2)]
        print(f"{name:30s} chosen(full)={r['chosen_full']}  per-fold chosen={list(r['chosen_per_fold'].values())}")
        print(f"   delta vs base per fold: {[f'{x:+.5f}' for x in d]}  mean {np.mean(d):+.5f} ± {np.std(d):.5f}"
              f"   | gain vs pooled ref: {[f'{x:.4f}' for x in g]} mean {np.mean(g):.4f}")
        for c in CONDS:
            cd = [r["folds"][f]["cond_delta"][c] for f in (0, 1, 2)]
            print(f"      {c:14s} {[f'{x:+.5f}' for x in cd]}  mean {np.mean(cd):+.5f} ± {np.std(cd):.5f}")


if __name__ == "__main__":
    main()
