"""Step 2 CV on TRAIN folds 0-2 (fold 3 LOCKED). Fold f is predicted by a model fitted on the other two
unlocked folds (same construction as Step 0/1). Only short-history machines can change (gate n < 100),
so rows are the short-history episodes of each fold at ticks cutoff-k, k = 0..11.
Score: Brier vs observed outcome; delta = base - variant (positive = better)."""
import csv, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "step0_diagnosis"))
sys.path.insert(0, str(HERE.parents[1] / "audit"))
KMAX = 12
PAY = (0, 1, 2, 5, 20)
VARIANTS = {
    "base": {},
    "2.0-fix_marg": {"marg": True},
    "2.1a_all_pairs": {"all_pairs": True},
    "2.1a_all_pairs+marg": {"all_pairs": True, "marg": True},
}
for pw in (0.5, 1.5, 2.0, 3.0):
    VARIANTS[f"2.1b_prior_power_{pw}"] = {"prior_power": pw}
    VARIANTS[f"2.1b_prior_power_{pw}+marg"] = {"prior_power": pw, "marg": True}


def fold_cfg(f):
    path = HERE / f"fold_cfgs/f{f}.json"
    if path.exists():
        return json.loads(path.read_text())
    from common import he, load, build_cfg, size_profiles_em, fit_pairs
    from folds import fold_map, LOCKED_FOLD
    tq, th, _, _ = load("train")
    fm, _ = fold_map("train")
    tr = [q for q in tq if fm[q["episode_id"]] not in (f, LOCKED_FOLD)]
    cfg = build_cfg(tr, th, size_profiles_em(tr, th))
    _, _, counts, _ = fit_pairs(tr, th, cfg, he.FlipGrid(cfg["grid_size"]))
    cfg["pair_counts"] = {f"{i},{j}": c for (i, j), c in counts.items()}
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(cfg))
    return cfg


def job(args):
    f, vname = args
    import variant
    from common import load
    from folds import fold_map
    cfg = fold_cfg(f)
    model = variant.he.Forecaster(cfg)
    tq, th, tcond, tlab = load("train")
    fm, _ = fold_map("train")
    te = [q for q in tq if fm[q["episode_id"]] == f and tcond[q["query_id"]] == "short_history"]
    rows = []
    for q in te:
        s = th[(q["episode_id"], q["machine_id"])]
        for k in range(KMAX):
            c = q["cutoff"] - k
            y = tlab[q["query_id"]] if k == 0 else s.get(c)
            if y is None:
                continue
            hist = {t: v for t, v in s.items() if t < c}
            p = variant.predict_one(model, dict(q, cutoff=c), hist, VARIANTS[vname])
            onehot = np.array([1.0 if y == v else 0.0 for v in PAY])
            rows.append({"episode": q["episode_id"], "k": k, "brier": float(((p - onehot) ** 2).sum()), "p": p.tolist()})
    return f, vname, rows


if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(3) as ex:
        list(ex.map(fold_cfg, (0, 1, 2)))
    print(f"fold configs ready ({time.time()-t0:.0f}s)", flush=True)
    jobs = [(f, v) for v in VARIANTS for f in (0, 1, 2)]
    res = {}
    with ProcessPoolExecutor(10) as ex:
        for f, v, rows in ex.map(job, jobs):
            res.setdefault(v, {})[f] = rows
    json.dump(res, open(HERE / "d_cv_rows.json", "w"))
    print(f"done ({time.time()-t0:.0f}s)")
