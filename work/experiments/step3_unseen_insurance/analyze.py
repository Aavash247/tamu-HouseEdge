"""Step 3 analysis: pair-prior epsilon grid (and + 2.1b) on train CV, validation (check), synthetic (target)."""
import csv, json, pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import components as C  # noqa: E402
from slot_forecasting.contest_data import load_contest, score_predictions  # noqa: E402

PAY = (0, 1, 2, 5, 20)
CONDS = ["stable", "short_history", "drift", "shared_change"]
SCEN = CONDS + ["unseen_pattern"]
EPS = [0.0, 0.005, 0.01, 0.02, 0.05]
COLS = ["p_0", "p_1", "p_2", "p_5", "p_20"]
D = pickle.load(open(HERE / "components.pkl", "rb"))
EXP = HERE.parent


def pick(r, p21b):
    return r["comp_p2"] if (p21b and r.get("comp_p2") is not None) else r["comp"]


# ---------- train CV
fold_cfg = {f: json.loads((EXP / f"step2_short_history/fold_cfgs/f{f}.json").read_text()) for f in (0, 1, 2)}


def cv_scores(eps, p21b):
    s = defaultdict(lambda: defaultdict(list))  # (fold, cond) -> episode -> briers
    for r in D["cv"]:
        p = C.combine(pick(r, p21b), fold_cfg[r["fold"]], eps)
        y = np.array([1.0 if r["y"] == v else 0.0 for v in PAY])
        s[(r["fold"], r["cond"])][r["episode"]].append(float(((p - y) ** 2).sum()))
    return {k: {e: np.mean(v) for e, v in d.items()} for k, d in s.items()}


base_cv = cv_scores(0.0, False)


def cv_delta(sc):
    """per fold: {cond: episode-weighted delta}, plus overall = mean over conditions. delta = base - variant."""
    out = {}
    for f in (0, 1, 2):
        out[f] = {c: float(np.mean([base_cv[(f, c)][e] - sc[(f, c)][e] for e in base_cv[(f, c)]])) for c in CONDS}
        out[f]["all"] = float(np.mean([out[f][c] for c in CONDS]))
    return out


# ---------- validation (check only)
final_cfg = json.loads((ROOT / "entry-final/model.json").read_text())
bundle = load_contest(ROOT / "public")


def val_points(eps, p21b):
    preds = {k: tuple(map(float, C.combine(pick(r, p21b), final_cfg, eps))) for k, r in D["val"].items()}
    sc = score_predictions(preds, bundle)
    return sc["technical_points"], {c: sc["by_condition"][c]["loss_ratio"] for c in CONDS}


# ---------- synthetic (main target)
SA = ROOT / "work/synthetic_audit"
syn_cfg = {s: json.loads((HERE / f"synth_cfgs/seed{s}.json").read_text()) for s in D["synth"]}
syn_meta = {}
for s in D["synth"]:
    d = SA / f"data/seed{s}/test"
    rd = lambda p: {r["query_id"]: np.array([float(r[c]) for c in COLS]) for r in csv.DictReader(open(p))}
    syn_meta[s] = {"truth": rd(d / "targets.csv"), "orc": rd(d / "param_oracle.csv"),
                   "cond": {r["query_id"]: r["condition"] for r in csv.DictReader(open(d / "conditions.csv"))},
                   "ep": {r["query_id"]: r["episode_id"] for r in csv.DictReader(open(d / "queries.csv"))},
                   "pooled": json.load(open(SA / f"results/seed{s}.json"))["episode_errors"]["pooled_frequency"]}


def synth_gaps(eps, p21b):
    per_seed = {}
    for s, rows in D["synth"].items():
        m = syn_meta[s]
        P = {k: C.combine(pick(r, p21b), syn_cfg[s], eps) for k, r in rows.items()}
        g = {}
        for c in SCEN:
            def err(Q):
                e = defaultdict(list)
                for k, v in Q.items():
                    if m["cond"][k] == c:
                        e[m["ep"][k]].append(float(((v - m["truth"][k]) ** 2).sum()))
                return np.mean([np.mean(v) for v in e.values()])
            pe = np.mean(list(m["pooled"][c].values()))
            g[c] = (pe - err(P)) / (pe - err(m["orc"]))
        per_seed[s] = g
    known = [100 * np.mean([g[c] for c in CONDS]) for g in per_seed.values()]
    unseen = [100 * g["unseen_pattern"] for g in per_seed.values()]
    return known, unseen


results = {}
base_val = val_points(0.0, False)[0]
print("delta = base - variant in Brier on train folds 0-2 (positive = better); fold 3 LOCKED. "
      "Validation and synthetic are checks/targets, not used for CV.\n")
for p21b in (False, True):
    for eps in EPS:
        name = f"eps={eps}" + (" +2.1b" if p21b else "")
        cvd = cv_delta(cv_scores(eps, p21b))
        vp, vr = val_points(eps, p21b)
        kn, un = synth_gaps(eps, p21b)
        allf = [cvd[f]["all"] for f in (0, 1, 2)]
        cond_stats = {c: (np.mean([cvd[f][c] for f in (0, 1, 2)]), np.std([cvd[f][c] for f in (0, 1, 2)])) for c in CONDS}
        within_noise = abs(np.mean(allf)) <= np.std(allf) + 1e-12 and all(m >= -sd - 1e-12 for m, sd in cond_stats.values())
        results[name] = {"eps": eps, "p21b": p21b, "cv": cvd, "val_points": vp, "val_ratios": vr,
                         "synth_known": kn, "synth_unseen": un, "within_noise": bool(within_noise),
                         "val_not_lower": bool(vp >= base_val - 1e-9)}
        print(f"{name:14s} CV all-cond delta per fold {[f'{x:+.6f}' for x in allf]} mean {np.mean(allf):+.6f} ± {np.std(allf):.6f}"
              f" | validation {vp:.4f} | synthetic known {np.mean(kn):.2f}% unseen {np.mean(un):.2f}% (seeds {[round(x, 1) for x in un]})")
        print("               per condition (mean ± sd over folds): " +
              "  ".join(f"{c}={m:+.6f}±{sd:.6f}" for c, (m, sd) in cond_stats.items()) +
              f"  | within_noise={within_noise} val_not_lower={vp >= base_val - 1e-9}")
json.dump(results, open(HERE / "analyze.json", "w"), indent=1, default=float)
for p21b in (False, True):
    ok = [r["eps"] for r in results.values() if r["p21b"] == p21b and r["eps"] > 0 and r["within_noise"] and r["val_not_lower"]]
    print(f"\nRULE PICK ({'with' if p21b else 'without'} 2.1b): largest eps passing = {max(ok) if ok else 'none'}")
