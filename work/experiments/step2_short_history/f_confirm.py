"""Confirmation only (no selection). Settings = CV picks on folds 0-2 (prior power 2.0).
Validation: final model config (entry-final/model.json, read-only) + variant options; only n<100
machines change, others reuse submissions/final/validation_predictions.csv. Scored with the official scorer.
Synthetic: per-seed config fitted on synthetic train only; short-history machines re-predicted."""
import csv, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "work/audit"))
SA = ROOT / "work/synthetic_audit"
COLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
OPTS = {"2.0-fix_marg": {"marg": True}, "2.1a_all_pairs": {"all_pairs": True},
        "2.1b_prior_power_2.0": {"prior_power": 2.0}, "2.1c_prior_power_2.0+marg": {"prior_power": 2.0, "marg": True}}
SCEN = ["stable", "short_history", "drift", "shared_change", "unseen_pattern"]


def rd(p):
    return {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(p))}


def pair_counts(cfg, tq, th):
    from common import fit_pairs, he
    _, _, counts, _ = fit_pairs(tq, th, cfg, he.FlipGrid(cfg["grid_size"]))
    return {f"{i},{j}": c for (i, j), c in counts.items()}


def validation():
    import variant
    from slot_forecasting.contest_data import load_contest, score_predictions
    from common import load
    cfg = json.loads((ROOT / "entry-final/model.json").read_text())
    tq, th, _, _ = load("train")
    cfg["pair_counts"] = pair_counts(cfg, tq, th)
    model = variant.he.Forecaster(cfg)
    vq = variant.he.read_queries(ROOT / "public/validation/queries.csv")
    vh = variant.he.read_history(ROOT / "public/validation/history.csv")
    base = rd(ROOT / "submissions/final/validation_predictions.csv")
    bundle = load_contest(ROOT / "public")
    out = {}
    for name, o in OPTS.items():
        p = dict(base)
        for q in vq:
            s = vh[(q["episode_id"], q["machine_id"])]
            if len(s) < variant.SHORT_N:
                p[q["query_id"]] = variant.predict_one(model, q, s, o)
        sc = score_predictions({k: tuple(map(float, v)) for k, v in p.items()}, bundle)
        R = sc["mean_loss_ratio"]; pts = sc["technical_points"]
        out[name] = {"val_points": pts, "ratios": {c: sc["by_condition"][c]["loss_ratio"] for c in sc["by_condition"]},
                     "pts_lost": {c: 60 / (1 + R - sc["by_condition"][c]["loss_ratio"] / 4) - pts for c in sc["by_condition"]}}
    return out


def synth(seed):
    import variant
    sys.path.insert(0, str(SA))
    from evaluate import build_cfg, size_profiles_from_targets, read_probs
    he = variant.he
    d = SA / f"data/seed{seed}"
    tq = he.read_queries(d / "train/queries.csv"); th = he.read_history(d / "train/history.csv")
    cfg = build_cfg(tq, th, size_profiles_from_targets(d / "train"))
    cfg["pair_counts"] = pair_counts(cfg, tq, th)
    model = he.Forecaster(cfg)
    vq = he.read_queries(d / "test/queries.csv"); vh = he.read_history(d / "test/history.csv")
    base = {k: np.array(v) for k, v in json.load(open(HERE.parent / f"step1_calibration/synth_seed{seed}.json"))["pred"].items()}
    truth = read_probs(d / "test/targets.csv"); orc = read_probs(d / "test/param_oracle.csv")
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(d / "test/conditions.csv"))}
    ep = {q["query_id"]: q["episode_id"] for q in vq}
    pooled = json.load(open(SA / f"results/seed{seed}.json"))["episode_errors"]["pooled_frequency"]
    def gaps(p):
        g = {}
        for c in SCEN:
            def err(P):
                e = {}
                for k in P:
                    if cond[k] == c: e.setdefault(ep[k], []).append(float(((P[k] - truth[k]) ** 2).sum()))
                return np.mean([np.mean(v) for v in e.values()])
            pe = np.mean(list(pooled[c].values()))
            g[c] = (pe - err(p)) / (pe - err(orc))
        return g
    out = {"base": gaps(base)}
    for name, o in OPTS.items():
        p = dict(base)
        for q in vq:
            s = vh[(q["episode_id"], q["machine_id"])]
            if len(s) < variant.SHORT_N:
                p[q["query_id"]] = variant.predict_one(model, q, s, o)
        out[name] = gaps(p)
    return seed, out


if __name__ == "__main__":
    with ProcessPoolExecutor(6) as ex:
        fv = ex.submit(validation)
        syn = list(ex.map(synth, [1, 2, 3, 4, 5]))
        val = fv.result()
    res = {"validation": val, "synthetic": {s: o for s, o in syn}}
    json.dump(res, open(HERE / "f_confirm.json", "w"), indent=1, default=float)
    for name in ["base"] + list(OPTS):
        known = np.mean([np.mean([o[name][c] for c in SCEN[:4]]) for _, o in syn]) * 100
        unseen = np.mean([o[name]["unseen_pattern"] for _, o in syn]) * 100
        short = [o[name]["short_history"] * 100 for _, o in syn]
        v = val.get(name)
        vs = f"validation={v['val_points']:.3f} short_ratio={v['ratios']['short_history']:.4f} lost={ {k: round(x, 3) for k, x in v['pts_lost'].items()} }" if v else "validation=55.602 (base)"
        print(f"{name:28s} {vs}  synthetic known={known:.1f}% unseen={unseen:.1f}% short_history per seed={[round(x, 1) for x in short]}")
