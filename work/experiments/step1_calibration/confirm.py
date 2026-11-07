"""Confirmation only (no selection): apply the CV-chosen settings to (1) the submitted model's validation
predictions, scored with the official scorer, and (2) the base model's synthetic test predictions."""
import csv, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from calibrate import apply_combo, PAY
from slot_forecasting.contest_data import load_contest, score_predictions
res = json.load(open(HERE / "cv_results.json"))
COLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
val = {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(ROOT / "submissions/final/validation_predictions.csv"))}
vq = {r["query_id"]: r for r in csv.DictReader(open(ROOT / "public/validation/queries.csv"))}
hist_n = {}
for r in csv.DictReader(open(ROOT / "public/validation/history.csv")):
    hist_n[r["machine_id"]] = hist_n.get(r["machine_id"], 0) + 1
# base rates from FULL train history (prediction-time quantity)
tq = {r["machine_id"]: r for r in csv.DictReader(open(ROOT / "public/train/queries.csv"))}
glob = np.zeros(5); typ = {}
for r in csv.DictReader(open(ROOT / "public/train/history.csv")):
    i = PAY.index(int(r["payout"])); glob[i] += 1
    typ.setdefault(tq[r["machine_id"]]["machine_type"], np.zeros(5))[i] += 1
glob /= glob.sum(); typ = {k: v / v.sum() for k, v in typ.items()}
bundle = load_contest(ROOT / "public")
synth = {s: json.load(open(HERE / f"synth_seed{s}.json")) for s in range(1, 6) if (HERE / f"synth_seed{s}.json").exists()}
SA = ROOT / "work/synthetic_audit"

def synth_gap(params):
    known, unseen = [], []
    for s, d in synth.items():
        tgt = {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(SA / f"data/seed{s}/test/targets.csv"))}
        orc = {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(SA / f"data/seed{s}/test/param_oracle.csv"))}
        cond = {r["query_id"]: r["condition"] for r in csv.DictReader(open(SA / f"data/seed{s}/test/conditions.csv"))}
        ep = {r["query_id"]: r["episode_id"] for r in csv.DictReader(open(SA / f"data/seed{s}/test/queries.csv"))}
        # synthetic pooled base rates come from synthetic train (prediction-time info for that world)
        sq = {r["machine_id"]: r["machine_type"] for r in csv.DictReader(open(SA / f"data/seed{s}/train/queries.csv"))}
        g = np.zeros(5); ty = {}
        for r in csv.DictReader(open(SA / f"data/seed{s}/train/history.csv")):
            i = PAY.index(int(r["payout"])); g[i] += 1; ty.setdefault(sq[r["machine_id"]], np.zeros(5))[i] += 1
        g /= g.sum(); ty = {k: v / v.sum() for k, v in ty.items()}
        ids = list(d["pred"]); P = np.array([d["pred"][k] for k in ids]); n = np.array([d["n"][k] for k in ids], float)
        pbt = np.array([ty[d["type"][k]] for k in ids])
        X = P
        for kind, th in params:
            X = apply_combo([(kind, th)], X, n, pbt if kind == "shrink_type" else np.tile(g, (len(ids), 1)))
        # pooled ref error per condition from saved results
        rj = json.load(open(SA / f"results/seed{s}.json"))["episode_errors"]
        for c in ["stable", "short_history", "drift", "shared_change", "unseen_pattern"]:
            cid = [i for i, k in enumerate(ids) if cond[k] == c]
            def err(Q):
                e = {}
                for i in cid: e.setdefault(ep[ids[i]], []).append(((Q[i] - tgt[ids[i]]) ** 2).sum())
                return np.mean([np.mean(v) for v in e.values()])
            pooled = np.mean(list(rj["pooled_frequency"][c].values())); oracle = err(np.array([orc[k] for k in ids]))
            gap = (pooled - err(X)) / (pooled - oracle)
            (unseen if c == "unseen_pattern" else known).append(gap)
    return 100 * np.mean(known), 100 * np.mean(unseen)

out = {}
for name, r in res.items():
    params = [tuple(x) for x in r["chosen_full"]]
    ids = list(val); P = np.array([val[k] for k in ids]); n = np.array([hist_n[vq[k]["machine_id"]] for k in ids], float)
    pbt = np.array([typ[vq[k]["machine_type"]] for k in ids])
    X = P
    for kind, th in params:
        X = apply_combo([(kind, th)], X, n, pbt if kind == "shrink_type" else np.tile(glob, (len(ids), 1)))
    s = score_predictions({k: tuple(map(float, x)) for k, x in zip(ids, X)}, bundle)
    R = s["mean_loss_ratio"]; pts = s["technical_points"]
    lost = {c: 60 / (1 + R - s["by_condition"][c]["loss_ratio"] / 4) - pts for c in s["by_condition"]}
    sk, su = synth_gap(params) if synth else (float("nan"), float("nan"))
    out[name] = {"params": params, "val_points": pts, "pts_lost": lost, "synth_known": sk, "synth_unseen": su}
    print(f"{name:30s} params={params}  validation={pts:.3f}  lost={ {k: round(v, 3) for k, v in lost.items()} }  synthetic known={sk:.1f}% unseen={su:.1f}%")
json.dump(out, open(HERE / "confirm.json", "w"), indent=1, default=float)
