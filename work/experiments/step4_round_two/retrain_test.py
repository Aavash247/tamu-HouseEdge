"""STEP 4.1 — retrain test on a NEW-PATTERN synthetic world.

The simulator (work/synthetic_audit/sim.py, unmodified; its TRAIN_PAIRS constant is overridden in this
process only) generates a full kit whose recurring lag pairs are NEW_PAIRS instead of (1,3),(2,5),(4,7).
Then the FULL training pipeline runs from scratch, exactly as fit.py does:
    Forecaster.fit(public, DEFAULT_CONFIG)  -> pairs discovered from train over all 28 pairs,
                                               rate priors from train, size profiles from validation targets
Both the base model and the CANDIDATE (2.1b power 2.0 for n<100 + eps 0.005) are scored on test against the
simulator's true probabilities. Sizes: per metadata, round kits have 24 episodes/condition for train and
validation (test is 48 in the real kit; 24 here to bound runtime). "half" = 12 train episodes/condition.
"""
import csv, json, sys, time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
SA = ROOT / "work/synthetic_audit"
NEW_PAIRS = ((1, 4), (2, 6), (3, 8))
SCEN = ["stable", "short_history", "drift", "shared_change"]
COLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
SIZES = {"full": 24, "half": 12}
VAL_PER, TEST_PER = 24, 24


def generate(seed, size):
    sys.path.insert(0, str(SA))
    import sim
    sim.TRAIN_PAIRS = NEW_PAIRS                     # override in this process only
    out = HERE / f"data/seed{seed}_{size}/public"
    if (out / "test/targets.csv").exists():
        return out
    rng = np.random.default_rng(1000 + seed)        # same world for full and half (train differs in count only)
    prof = sim.size_profiles(rng)
    rng_tr = np.random.default_rng(2000 + seed); rng_va = np.random.default_rng(3000 + seed); rng_te = np.random.default_rng(4000 + seed)
    sim.write_split(out, "train", f"r2s{seed}", SCEN, SIZES[size], rng_tr, prof)
    sim.write_split(out, "validation", f"r2s{seed}", SCEN, VAL_PER, rng_va, prof)
    sim.write_split(out, "test", f"r2s{seed}", SCEN, TEST_PER, rng_te, prof)
    return out


def run(args):
    seed, size = args
    t0 = time.time()
    public = generate(seed, size)
    sys.path.insert(0, str(ROOT / "entry-final")); sys.path.insert(0, str(ROOT / "work/experiments/step3_unseen_insurance"))
    sys.path.insert(0, str(ROOT))
    import houseedge as he
    import components as C
    from slot_forecasting.io import load_inputs, read_labels
    from slot_forecasting.baseline import FrequencyForecaster
    model = he.Forecaster.fit(public, he.DEFAULT_CONFIG)          # full pipeline, from scratch
    t_fit = time.time() - t0
    sq, sh = load_inputs(public / "train")
    pooled = FrequencyForecaster("pooled", prior_strength=20.0).fit(sq, sh, read_labels(public / "train/labels.csv", sq))
    vq = he.read_queries(public / "test/queries.csv"); vh = he.read_history(public / "test/history.csv")
    tq_sf, th_sf = load_inputs(public / "test")
    rd = lambda p: {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(p))}
    truth = rd(public / "test/targets.csv"); orc = rd(public / "test/param_oracle.csv")
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(public / "test/conditions.csv"))}
    ep = {q["query_id"]: q["episode_id"] for q in vq}
    preds = {"pooled": {k: np.array(v) for k, v in pooled.predict(tq_sf, th_sf).items()}, "base": {}, "candidate": {}, "oracle": orc}
    maxdiff = 0.0
    for n, q in enumerate(vq):
        s = vh[(q["episode_id"], q["machine_id"])]
        c1 = C.components(model, q, s, 1.0)
        preds["base"][q["query_id"]] = C.combine(c1, model.cfg, 0.0)
        c2 = C.components(model, q, s, 2.0) if c1["n"] < C.SHORT_N else c1
        preds["candidate"][q["query_id"]] = C.combine(c2, model.cfg, 0.005)
        if n < 5:  # base via components must equal the shipped Forecaster exactly
            maxdiff = max(maxdiff, float(np.abs(preds["base"][q["query_id"]] - model.predict_one(q, s)).max()))
    err = {}
    for name, P in preds.items():
        err[name] = {}
        for c in SCEN:
            e = defaultdict(list)
            for k, v in P.items():
                if cond[k] == c:
                    e[ep[k]].append(float(((v - truth[k]) ** 2).sum()))
            err[name][c] = float(np.mean([np.mean(x) for x in e.values()]))
    gap = {m: {c: (err["pooled"][c] - err[m][c]) / (err["pooled"][c] - err["oracle"][c]) for c in SCEN} for m in ("base", "candidate")}
    ratio = {m: {c: err[m][c] / err["pooled"][c] for c in SCEN} for m in ("base", "candidate", "oracle")}
    return {"seed": seed, "size": size, "pairs_true": [list(p) for p in NEW_PAIRS], "pairs_found": model.cfg["pairs"],
            "pair_prior": model.cfg["pair_prior"], "size_profiles_found": model.cfg["size_profiles"],
            "err": err, "gap": gap, "ratio": ratio, "base_vs_forecaster_maxdiff": maxdiff,
            "fit_seconds": t_fit, "total_seconds": time.time() - t0}


if __name__ == "__main__":
    jobs = [(s, z) for s in (1, 2, 3, 4, 5) for z in ("full", "half")]
    with ProcessPoolExecutor(10) as ex:
        res = list(ex.map(run, jobs))
    json.dump(res, open(HERE / "retrain_results.json", "w"), indent=1, default=float)
    for r in res:
        print(f"seed {r['seed']} {r['size']:4s} found={r['pairs_found']} true={r['pairs_true']} "
              f"prior={[round(x, 2) for x in r['pair_prior']]} check={r['base_vs_forecaster_maxdiff']:.1e} "
              f"fit={r['fit_seconds']:.0f}s total={r['total_seconds']:.0f}s")
    for size in ("full", "half"):
        for m in ("base", "candidate"):
            print(f"\n{size:4s} {m:9s} gap closed per condition, mean over seeds [min, max]:")
            for c in SCEN:
                v = [r["gap"][m][c] * 100 for r in res if r["size"] == size]
                rv = [r["ratio"][m][c] for r in res if r["size"] == size]
                print(f"   {c:14s} {np.mean(v):6.1f}% [{min(v):.1f}, {max(v):.1f}]   loss ratio vs pooled {np.mean(rv):.3f}")
            v = [np.mean([r["gap"][m][c] for c in SCEN]) * 100 for r in res if r["size"] == size]
            pts = [60 / (1 + np.mean([r["ratio"][m][c] for c in SCEN])) for r in res if r["size"] == size]
            print(f"   ALL            {np.mean(v):6.1f}% per seed {[round(x, 1) for x in v]}   implied points {np.mean(pts):.2f} {[round(x, 2) for x in pts]}")
