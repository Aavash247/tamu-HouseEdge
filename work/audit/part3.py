"""Summarize CV rows: Brier gain over the pooled reference (= reduction in squared error to truth)."""
import json, random
from common import *

rows = json.load(open(ROOT / "work/audit/cv_rows.json"))
vrows = json.load(open(ROOT / "work/audit/val_brier_rows.json"))
M = ["reference_pooled", "original_blend", "new_trainonly"]


def gains(rs):
    for r in rs:
        r["gain_new"] = r["reference_pooled"] - r["new_trainonly"]
        r["gain_old"] = r["reference_pooled"] - r["original_blend"]
    return rs


def boot_ci(rs, key, n=2000, seed=0):
    eps = defaultdict(list)
    for r in rs:
        eps[r["episode"]].append(r[key])
    vals = np.array([np.mean(v) for v in eps.values()])
    rng = np.random.default_rng(seed)
    bs = [rng.choice(vals, len(vals)).mean() for _ in range(n)]
    return vals.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


gains(rows); gains(vrows)
print("Mean Brier (episode-weighted). Lower is better. Gain = reference - model = reduction in squared error to truth.\n")
hdr = f"{'set':22s}{'cond':15s}" + "".join(f"{m:>18s}" for m in M) + f"{'gain_new':>10s}{'gain_old':>10s}"
print(hdr)
for label, rs in [(f"train-CV fold {f}", [r for r in rows if r["fold"] == f]) for f in range(4)] + \
                 [("train-CV all folds", rows), ("validation (submitted)", vrows)]:
    for c in CONDS + ["ALL"]:
        sub = rs if c == "ALL" else [r for r in rs if r["cond"] == c]
        if c == "ALL":
            vals = {k: np.mean([cond_table(rs, k)[cc] for cc in CONDS]) for k in M + ["gain_new", "gain_old"]}
        else:
            vals = {k: cond_table(sub, k)[c] for k in M + ["gain_new", "gain_old"]}
        print(f"{label:22s}{c:15s}" + "".join(f"{vals[m]:18.4f}" for m in M) + f"{vals['gain_new']:10.4f}{vals['gain_old']:10.4f}")
    print()

print("95% bootstrap CI (over episodes) of Brier gain of the new model over the reference:")
for label, rs in (("train-CV", rows), ("validation", vrows)):
    for c in CONDS:
        m, lo, hi = boot_ci([r for r in rs if r["cond"] == c], "gain_new")
        print(f"  {label:11s} {c:14s} gain={m:.4f}  [{lo:.4f}, {hi:.4f}]")

print("\nGain by pseudo-cutoff offset k (k=0 is the real query), all conditions pooled:")
for label, rs in (("train-CV", rows), ("validation", vrows)):
    print("  " + label + ": " + "  ".join(
        f"k={k}:{np.mean([r['gain_new'] for r in rs if r['k'] == k]):.4f}" for k in range(7)))
print("\nrows: train-CV", len(rows), "| validation", len(vrows))
