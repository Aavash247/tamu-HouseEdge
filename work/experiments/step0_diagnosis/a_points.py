"""Points lost per condition for the existing final model.
(a) Official metric on VALIDATION (scoring only, no fitting), split into 4 stratified episode folds.
(b) Train-CV (folds 0-2 only; fold 3 locked): Brier vs outcomes from work/audit/cv_rows.json."""
import csv, json
import numpy as np
from folds import ROOT, CONDS, LOCKED_FOLD, fold_map

COLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
def probs(path):
    return {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(path))}
pred = probs(ROOT / "submissions/final/validation_predictions.csv")
ref = probs(ROOT / "public/validation/reference.csv")
tgt = probs(ROOT / "public/validation/targets.csv")
qep = {r["query_id"]: r["episode_id"] for r in csv.DictReader(open(ROOT / "public/validation/queries.csv"))}
vfold, vcond = fold_map("validation")

def cond_err(p, ids):
    eps = {}
    for i in ids:
        eps.setdefault(qep[i], []).append(((p[i] - tgt[i]) ** 2).sum())
    return np.mean([np.mean(v) for v in eps.values()])

def points_breakdown(ids):
    r = {}
    for c in CONDS:
        cid = [i for i in ids if vcond[qep[i]] == c]
        r[c] = cond_err(pred, cid) / cond_err(ref, cid)
    R = np.mean(list(r.values())); pts = 60 / (1 + R)
    lost = {c: 60 / (1 + R - r[c] / 4) - pts for c in CONDS}   # points regained if condition c were perfect
    return pts, r, lost

out = {"validation_folds": {}}
allids = list(tgt)
rows = []
for f in list(range(4)) + ["all"]:
    ids = allids if f == "all" else [i for i in allids if vfold[qep[i]] == f]
    pts, r, lost = points_breakdown(ids)
    out["validation_folds"][str(f)] = {"points": pts, "ratio": r, "points_lost": lost}
    rows.append((f, pts, r, lost))
print("(a) VALIDATION, official metric (scoring only). points_lost[c] = points gained if condition c had zero error")
print(f"{'fold':>5} {'points':>7} | " + " ".join(f"{c[:12]:>13}" for c in CONDS) + " | lost: " + " ".join(f"{c[:6]:>7}" for c in CONDS) + "   total_lost")
for f, pts, r, lost in rows:
    print(f"{str(f):>5} {pts:7.3f} | " + " ".join(f"{r[c]:13.4f}" for c in CONDS) + " |       " + " ".join(f"{lost[c]:7.3f}" for c in CONDS) + f"   {60-pts:6.3f}")
per = np.array([[x[3][c] for c in CONDS] for x in rows[:4]])
print("fold mean ± sd of points lost: " + "  ".join(f"{c}={m:.3f}±{s:.3f}" for c, m, s in zip(CONDS, per.mean(0), per.std(0))))
print("fold points mean ± sd: %.3f ± %.3f" % (np.mean([x[1] for x in rows[:4]]), np.std([x[1] for x in rows[:4]])))

cv = json.load(open(ROOT / "work/audit/cv_rows.json"))
print("\n(b) TRAIN-CV (fold 3 LOCKED, excluded). Brier vs outcomes; gain = reference - model")
print(f"{'fold':>5} " + " ".join(f"{c[:12]:>26}" for c in CONDS))
gains = {}
for f in (0, 1, 2):
    cells = []
    for c in CONDS:
        eps = {}
        for r in cv:
            if r["fold"] == f and r["cond"] == c:
                eps.setdefault(r["episode"], []).append((r["new_trainonly"], r["reference_pooled"]))
        m = np.mean([np.mean([a for a, _ in v]) for v in eps.values()]); b = np.mean([np.mean([b for _, b in v]) for v in eps.values()])
        gains.setdefault(c, []).append(b - m); cells.append(f"model={m:.4f} gain={b-m:.4f}")
    print(f"{f:>5} " + " ".join(f"{x:>26}" for x in cells))
print("gain mean ± sd over folds 0-2: " + "  ".join(f"{c}={np.mean(v):.4f}±{np.std(v):.4f}" for c, v in gains.items()))
out["train_cv_gain_folds012"] = gains
json.dump(out, open("a_points.json", "w"), indent=1, default=float)
