"""Fit on synthetic TRAIN episodes only; score on unseen synthetic TEST episodes against true probabilities.

Score = squared error to the simulator's true distribution, sum_k (p_k - q_k)^2 (the competition
metric; a Brier score computed against true probabilities rather than sampled outcomes).
Averaged within episode, then over episodes, per scenario.
"""
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "audit"))
from common import he, build_cfg  # noqa: E402  (imports entry-final/houseedge.py read-only)
sys.path.insert(0, str(ROOT))
from slot_forecasting.io import load_inputs, read_labels  # noqa: E402
from slot_forecasting.baseline import FrequencyForecaster  # noqa: E402

COLS = ("p_0", "p_1", "p_2", "p_5", "p_20")
SCEN = ["stable", "short_history", "drift", "shared_change", "unseen_pattern"]


def read_probs(path):
    with open(path) as f:
        return {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(f)}


def size_profiles_from_targets(split_dir):
    """Same clustering as Forecaster._size_profiles, applied to the TRAIN split's targets."""
    qtype = {q["query_id"]: q["machine_type"] for q in he.read_queries(split_dir / "queries.csv")}
    groups = defaultdict(list)
    for qid, p in read_probs(split_dir / "targets.csv").items():
        s = p[1:]
        if s.sum() > 1e-6:
            groups[qtype[qid]].append(s / s.sum())
    out = {}
    for typ, vecs in groups.items():
        clusters = []
        for v in vecs:
            for c in clusters:
                if np.abs(c["sum"] / c["n"] - v).max() < 0.02:
                    c["sum"] += v; c["n"] += 1; break
            else:
                clusters.append({"sum": v.copy(), "n": 1})
        keep = [c for c in clusters if c["n"] >= 0.05 * len(vecs)] or [{"sum": np.sum(vecs, 0), "n": len(vecs)}]
        keep.sort(key=lambda c: -(c["sum"] / c["n"])[0])
        out[typ] = [list(map(float, c["sum"] / c["n"])) for c in keep]
    return out


def episode_scores(pred, truth, episode, cond):
    by = defaultdict(lambda: defaultdict(list))
    for qid, q in truth.items():
        by[cond[qid]][episode[qid]].append(float(((pred[qid] - q) ** 2).sum()))
    return {c: {e: float(np.mean(v)) for e, v in eps.items()} for c, eps in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = Path(a.data)
    t0 = time.time()

    # ---- fit on TRAIN only
    tq = he.read_queries(data / "train" / "queries.csv")
    th = he.read_history(data / "train" / "history.csv")
    cfg = build_cfg(tq, th, size_profiles_from_targets(data / "train"))
    model = he.Forecaster(cfg)
    sq, sh = load_inputs(data / "train")
    pooled = FrequencyForecaster("pooled", prior_strength=20.0).fit(sq, sh, read_labels(data / "train" / "labels.csv", sq))
    t_fit = time.time() - t0

    # ---- predict unseen TEST
    vq = he.read_queries(data / "test" / "queries.csv")
    vh = he.read_history(data / "test" / "history.csv")
    truth = read_probs(data / "test" / "targets.csv")
    with open(data / "test" / "conditions.csv") as f:
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(f)}
    episode = {q["query_id"]: q["episode_id"] for q in vq}
    tq_sf, th_sf = load_inputs(data / "test")
    preds = {
        "uniform": {k: np.full(5, 0.2) for k in truth},
        "pooled_frequency": {k: np.array(v) for k, v in pooled.predict(tq_sf, th_sf).items()},
        "house_edge_model": {k: np.asarray(v) for k, v in model.predict(vq, vh).items()},
        "oracle_params_no_hidden": read_probs(data / "test" / "param_oracle.csv"),
        "oracle_true": truth,
    }
    t_all = time.time() - t0
    result = {"data": str(data), "fit_seconds": t_fit, "total_seconds": t_all,
              "learned_pairs": cfg["pairs"], "learned_pair_prior": cfg["pair_prior"],
              "learned_size_profiles": cfg["size_profiles"], "episode_errors": {}}
    for name, p in preds.items():
        result["episode_errors"][name] = episode_scores(p, truth, episode, cond)
    Path(a.out).write_text(json.dumps(result, indent=1))
    print(f"{data.name}: pairs={cfg['pairs']} fit={t_fit:.0f}s total={t_all:.0f}s")


if __name__ == "__main__":
    main()
