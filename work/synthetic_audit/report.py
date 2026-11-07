"""Aggregate per-seed results: squared error to truth per scenario, ratio to pooled, implied points."""
import json, sys
import numpy as np

SCEN = ["stable", "short_history", "drift", "shared_change", "unseen_pattern"]
MODELS = ["uniform", "pooled_frequency", "house_edge_model", "oracle_params_no_hidden", "oracle_true"]
runs = [json.load(open(p)) for p in sys.argv[1:]]


def cond_err(run, model, scen):
    eps = run["episode_errors"][model][scen]
    return float(np.mean(list(eps.values())))


print(f"seeds: {len(runs)} | learned pairs per seed: {[r['learned_pairs'] for r in runs]}\n")
print("Mean squared error to the TRUE distribution (episode-weighted), mean ± sd over seeds. Lower is better.")
print(f"{'scenario':16s}" + "".join(f"{m:>26s}" for m in MODELS))
for s in SCEN:
    cells = []
    for m in MODELS:
        v = np.array([cond_err(r, m, s) for r in runs])
        cells.append(f"{v.mean():.4f} ± {v.std():.4f}")
    print(f"{s:16s}" + "".join(f"{c:>26s}" for c in cells))

print("\nLoss ratio vs pooled_frequency (the competition normalisation), mean [min, max] over seeds:")
print(f"{'scenario':16s}" + "".join(f"{m:>26s}" for m in MODELS))
for s in SCEN:
    cells = []
    for m in MODELS:
        v = np.array([cond_err(r, m, s) / cond_err(r, "pooled_frequency", s) for r in runs])
        cells.append(f"{v.mean():.3f} [{v.min():.3f},{v.max():.3f}]")
    print(f"{s:16s}" + "".join(f"{c:>26s}" for c in cells))

print("\nImplied competition-style points 60/(1+R) using the four SEEN scenarios (R = mean ratio):")
for m in MODELS:
    pts = []
    for r in runs:
        R = np.mean([cond_err(r, m, s) / cond_err(r, "pooled_frequency", s) for s in SCEN[:4]])
        pts.append(60 / (1 + R))
    print(f"  {m:26s} {np.mean(pts):6.2f}  per seed: {[round(p, 2) for p in pts]}")

print("\nShare of the achievable gap closed  (pooled - model) / (pooled - oracle_params_no_hidden):")
for s in SCEN:
    v = []
    for r in runs:
        p, m, o = (cond_err(r, k, s) for k in ("pooled_frequency", "house_edge_model", "oracle_params_no_hidden"))
        v.append((p - m) / (p - o))
    print(f"  {s:16s} {np.mean(v):.3f}  per seed: {[round(x, 3) for x in v]}")
