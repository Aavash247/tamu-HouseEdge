"""Sensitivity of Brier to each calibration parameter, pooled over folds 0-2 (diagnostic only;
the selection itself is nested CV in calibrate.py). Values are base - variant (positive = better), x1e4."""
import numpy as np
from calibrate import apply, brier, overall, ep_cond_mean, CONDS, HERE
d = np.load(HERE / "eval_arrays.npz", allow_pickle=True)
P, Y, n, E, C, PBT = d["P"], d["Y"], d["n"], d["E"], d["C"], d["PB_T"]
b0 = brier(P, Y)
for kind, grid in (("shrink_type", [0.5, 1, 2, 5, 10, 20]), ("temp", [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]), ("floor", [1e-4, 5e-4, 1e-3, 3e-3, 1e-2])):
    print(kind)
    for th in grid:
        b = brier(apply(kind, th, P, n, PBT), Y)
        print(f"  {th:<7} all={1e4*overall(b0-b, E, C):+7.2f}  " + "  ".join(f"{c[:6]}={1e4*ep_cond_mean(b0-b, E, C, c):+7.2f}" for c in CONDS))
short = C == "short_history"
print("short_history mean history length n:", n[short].mean(), "| others:", n[~short].mean())
