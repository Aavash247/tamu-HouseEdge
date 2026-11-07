"""Effect of eps measured against 2.1b alone (eps=0 +2.1b), the correct reference for the combined variants."""
import json, numpy as np
R = json.load(open("analyze.json")); C = ["stable", "short_history", "drift", "shared_change"]
ref = R["eps=0.0 +2.1b"]
print("eps effect on top of 2.1b: delta = (2.1b) - (eps+2.1b), positive = better")
picks = []
for e in (0.005, 0.01, 0.02, 0.05):
    r = R[f"eps={e} +2.1b"]
    f_all = [r["cv"][str(f)]["all"] - ref["cv"][str(f)]["all"] for f in (0, 1, 2)]
    cs = {c: [r["cv"][str(f)][c] - ref["cv"][str(f)][c] for f in (0, 1, 2)] for c in C}
    noise = abs(np.mean(f_all)) <= np.std(f_all) and all(np.mean(v) >= -np.std(v) - 1e-12 for v in cs.values())
    vok = r["val_points"] >= ref["val_points"] - 1e-9
    if noise and vok: picks.append(e)
    print(f"eps={e:<6} folds {[f'{x:+.6f}' for x in f_all]} mean {np.mean(f_all):+.6f} ± {np.std(f_all):.6f} | short {np.mean(cs['short_history']):+.6f} ± {np.std(cs['short_history']):.6f}"
          f" | validation {r['val_points']:.4f} vs 2.1b {ref['val_points']:.4f} | within_noise={noise} val_ok={vok}")
print("RULE PICK on top of 2.1b: largest eps passing =", max(picks) if picks else "none")
