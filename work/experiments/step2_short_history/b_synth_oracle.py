"""2.0b: oracle decomposition on synthetic data (fit on synthetic train only).
For each test machine, compare squared error to the true distribution for:
  base            the model as is
  true_pair       the model restricted to the machine's TRUE lag pair (rates still estimated)
  true_pair+rates the TRUE win probability (true pair and true rates; the hidden-state marginal),
                  with the model's own payout-size distribution
  true_size       the model's win probability with the TRUE size mixture (pi known)
  param_oracle    everything known except the hidden state at the query tick (best achievable)
Machine parameters are captured by regenerating the same seeds with sim.simulate_episode wrapped
(sim.py itself is not modified); the regenerated targets are checked byte-identical.
"""
import csv, hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SA = HERE.parents[1] / "synthetic_audit"
COLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
SCEN = ["stable", "short_history", "drift", "shared_change", "unseen_pattern"]


def capture_params(seed):
    sys.path.insert(0, str(SA))
    import sim
    captured = []
    orig = sim.simulate_episode
    def wrapped(*a, **k):
        ms = orig(*a, **k); captured.append([m["params"] for m in ms]); return ms
    sim.simulate_episode = wrapped
    out = HERE / f"synth_regen/seed{seed}"
    rng = np.random.default_rng(seed)
    prof = sim.size_profiles(rng)
    seen = ["stable", "short_history", "drift", "shared_change"]
    sim.write_split(out, "train", f"syn{seed}", seen, 32, rng, prof)
    n_train = len(captured)
    sim.write_split(out, "test", f"syn{seed}", seen + ["unseen_pattern"], 32, rng, prof)
    test_params = {}
    for e, ms in enumerate(captured[n_train:]):
        for m, p in enumerate(ms):
            test_params[f"syn{seed}-test-e{e:04d}-m{m:02d}-next"] = {k: (v.item() if hasattr(v, "item") else v) for k, v in p.items()}
    same = all(hashlib.sha256((out / "test" / f).read_bytes()).digest() ==
               hashlib.sha256((SA / f"data/seed{seed}/test" / f).read_bytes()).digest()
               for f in ("targets.csv", "history.csv", "queries.csv"))
    return test_params, same, prof


def run(seed):
    sys.path.insert(0, str(SA)); sys.path.insert(0, str(SA.parent / "audit"))
    params, same, prof = capture_params(seed)
    from evaluate import he, build_cfg, size_profiles_from_targets, read_probs
    d = SA / f"data/seed{seed}"
    tq = he.read_queries(d / "train/queries.csv"); th = he.read_history(d / "train/history.csv")
    cfg = build_cfg(tq, th, size_profiles_from_targets(d / "train"))
    model = he.Forecaster(cfg)
    vq = he.read_queries(d / "test/queries.csv"); vh = he.read_history(d / "test/history.csv")
    truth = read_probs(d / "test/targets.csv"); orc = read_probs(d / "test/param_oracle.csv")
    base = json.load(open(HERE.parent / f"step1_calibration/synth_seed{seed}.json"))["pred"]
    cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(d / "test/conditions.csv"))}
    ep = {q["query_id"]: q["episode_id"] for q in vq}
    sp = cfg["size_profiles"]
    preds = {k: {} for k in ("base", "true_pair", "true_pair+rates", "true_size", "param_oracle")}
    for q in vq:
        qid = q["query_id"]; p = params[qid]; b = np.array(base[qid])
        preds["base"][qid] = b
        preds["param_oracle"][qid] = orc[qid]
        if p["i"] and [p["i"], p["j"]] in cfg["pairs"] or True:
            c2 = dict(cfg, pairs=[[p["i"], p["j"]]], pair_prior=[1.0], null_prior=1e-12)
            m2 = he.Forecaster.__new__(he.Forecaster)
            m2.cfg = c2; m2.grid = model.grid; m2.grid4 = model.grid4; m2.prior2 = model.prior2; m2.prior4 = model.prior4
            preds["true_pair"][qid] = m2.predict_one(q, vh[(q["episode_id"], q["machine_id"])])
        size_model = b[1:] / b[1:].sum()
        pw_true = 1 - orc[qid][0]
        preds["true_pair+rates"][qid] = np.concatenate([[1 - pw_true], pw_true * size_model])
        # true size mixture given win: oracle's conditional size distribution
        size_true = orc[qid][1:] / orc[qid][1:].sum()
        pw_model = 1 - b[0]
        preds["true_size"][qid] = np.concatenate([[1 - pw_model], pw_model * size_true])
    res = {}
    for name, pr in preds.items():
        res[name] = {}
        for c in SCEN:
            e = {}
            for qid in pr:
                if cond[qid] == c:
                    e.setdefault(ep[qid], []).append(float(((pr[qid] - truth[qid]) ** 2).sum()))
            res[name][c] = float(np.mean([np.mean(v) for v in e.values()]))
    return seed, same, res


if __name__ == "__main__":
    with ProcessPoolExecutor(5) as ex:
        out = list(ex.map(run, [1, 2, 3, 4, 5]))
    json.dump(out, open(HERE / "b_synth_oracle.json", "w"), indent=1)
    print("regenerated data identical to synthetic_audit data:", [s for _, s, _ in out])
    print("\nShare of the gap (base -> param_oracle) closed, mean over 5 seeds [min, max]; error values are means")
    for c in SCEN:
        base = np.array([r["base"][c] for _, _, r in out]); orc = np.array([r["param_oracle"][c] for _, _, r in out])
        line = f"{c:15s} base={base.mean():.4f} oracle={orc.mean():.4f} |"
        for name in ("true_pair", "true_pair+rates", "true_size"):
            v = np.array([r[name][c] for _, _, r in out]); sh = (base - v) / (base - orc)
            line += f" {name}: err={v.mean():.4f} closes {100*sh.mean():5.1f}% [{100*sh.min():.0f},{100*sh.max():.0f}] |"
        print(line)
