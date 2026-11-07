"""Compute model components (all 28 pairs) for:
  cv     TRAIN folds 0-2 (fold 3 LOCKED), all conditions, ticks cutoff-k for k = 0..5, models fitted on
         the other two unlocked folds (Step 2 fold configs, reused read-only)
  val    VALIDATION queries with the final model config (check only)
  synth  synthetic test queries, per seed, config fitted on synthetic train only
Short-history rows get a second pass with prior power 2.0 (for the eps + 2.1b combination)."""
import csv, json, pickle, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent; EXP = HERE.parent; ROOT = HERE.parents[2]
for p in (HERE, EXP / "step0_diagnosis", ROOT / "work/audit", ROOT / "work/synthetic_audit", ROOT):
    sys.path.insert(0, str(p))
KMAX = 6
PAY = (0, 1, 2, 5, 20)


def cv_job(args):
    f, chunk = args
    import components as C
    from common import load
    from folds import fold_map
    cfg = json.loads((EXP / f"step2_short_history/fold_cfgs/f{f}.json").read_text())
    model = C.he.Forecaster(cfg)
    tq, th, tcond, tlab = load("train")
    fm, _ = fold_map("train")
    te = [q for q in tq if fm[q["episode_id"]] == f][chunk::4]
    rows = []
    for q in te:
        s = th[(q["episode_id"], q["machine_id"])]
        for k in range(KMAX):
            c = q["cutoff"] - k
            y = tlab[q["query_id"]] if k == 0 else s.get(c)
            if y is None:
                continue
            hist = {t: v for t, v in s.items() if t < c}
            qq = dict(q, cutoff=c)
            comp = C.components(model, qq, hist, 1.0)
            comp2 = C.components(model, qq, hist, 2.0) if comp["n"] < C.SHORT_N else None
            rows.append({"fold": f, "episode": q["episode_id"], "cond": tcond[q["query_id"]], "k": k, "y": y,
                         "comp": comp, "comp_p2": comp2})
    return ("cv", f, rows)


def val_job(chunk):
    import components as C
    cfg = json.loads((ROOT / "entry-final/model.json").read_text())
    model = C.he.Forecaster(cfg)
    vq = C.he.read_queries(ROOT / "public/validation/queries.csv")[chunk::4]
    vh = C.he.read_history(ROOT / "public/validation/history.csv")
    rows = {}
    for q in vq:
        s = vh[(q["episode_id"], q["machine_id"])]
        comp = C.components(model, q, s, 1.0)
        rows[q["query_id"]] = {"comp": comp, "comp_p2": C.components(model, q, s, 2.0) if comp["n"] < C.SHORT_N else None}
    return ("val", chunk, rows)


def synth_cfg(seed):
    path = HERE / f"synth_cfgs/seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    from evaluate import he, build_cfg, size_profiles_from_targets
    d = ROOT / f"work/synthetic_audit/data/seed{seed}"
    tq = he.read_queries(d / "train/queries.csv"); th = he.read_history(d / "train/history.csv")
    cfg = build_cfg(tq, th, size_profiles_from_targets(d / "train"))
    path.parent.mkdir(exist_ok=True); path.write_text(json.dumps(cfg))
    return cfg


def synth_job(args):
    seed, chunk = args
    import components as C
    cfg = synth_cfg(seed)
    model = C.he.Forecaster(cfg)
    d = ROOT / f"work/synthetic_audit/data/seed{seed}"
    vq = C.he.read_queries(d / "test/queries.csv")[chunk::4]
    vh = C.he.read_history(d / "test/history.csv")
    rows = {}
    for q in vq:
        s = vh[(q["episode_id"], q["machine_id"])]
        comp = C.components(model, q, s, 1.0)
        rows[q["query_id"]] = {"comp": comp, "comp_p2": C.components(model, q, s, 2.0) if comp["n"] < C.SHORT_N else None}
    return ("synth", (seed, chunk), rows)


if __name__ == "__main__":
    t0 = time.time()
    with ProcessPoolExecutor(5) as ex:
        list(ex.map(synth_cfg, [1, 2, 3, 4, 5]))
    print(f"synthetic configs ready ({time.time()-t0:.0f}s)", flush=True)
    out = {"cv": [], "val": {}, "synth": {}}
    with ProcessPoolExecutor(11) as ex:
        futs = [ex.submit(cv_job, (f, c)) for f in (0, 1, 2) for c in range(4)]
        futs += [ex.submit(val_job, c) for c in range(4)]
        futs += [ex.submit(synth_job, (s, c)) for s in (1, 2, 3, 4, 5) for c in range(4)]
        for fu in futs:
            kind, key, rows = fu.result()
            if kind == "cv":
                out["cv"] += rows
            elif kind == "val":
                out["val"].update(rows)
            else:
                out["synth"].setdefault(key[0], {}).update(rows)
            print(f"  {kind} {key} done ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(out, open(HERE / "components.pkl", "wb"))
    print(f"done ({time.time()-t0:.0f}s)")
