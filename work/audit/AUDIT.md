# Overfitting audit of the 55.60 submission

This audit does not change the model: `entry-final/` and `submissions/final/` are untouched.
Run the scripts from the repo root with numpy available (for example `work/.venv/bin/python`).

| File | What it does |
|---|---|
| `common.py` | Shared helpers; imports `entry-final/houseedge.py` read-only |
| `part1.py` → `part1.out` | Leakage and ID probes, lag-pair consistency between train and validation, size profiles fitted from train only |
| `probe_nb.py` | Corrected neighbor-scramble probe, run one query at a time (appended to `part1.out`) |
| `part2.py` → `part2.out`, `cv_rows.json`, `val_brier_rows.json`, `cv_fold_cfgs.json` | 4-fold episode-level cross-validation entirely inside train (about 10 min) |
| `part3.py` → `part3.out` | Summary of the cross-validation: Brier gain over the reference, 95% bootstrap intervals |

## Findings

**How validation was used.** The payout-size profiles were fitted from the validation targets. The settings were chosen by scoring on the full validation set across about 16 experiments: flip rate, prior mix, 2/4-cell blend weights, flip direction, size half-life, the no-lag model's weight, and grid levels. The two validation halves were therefore *not* independent of model development. The lag pairs and win-rate priors come from train only.

**ID memorization.** None. `model.json` holds no episode, machine or query IDs, and its priors are keyed only by machine type. Renaming IDs changes predictions by 0.0.

**Leakage.** None found. Adding fake ticks at or after the cutoff changes predictions by 0.0, and so does scrambling neighbor machines one query at a time. Flipping the lag-1 input changes them by 0.84, which shows the probe is sensitive. No history row in any split is at or after its query's cutoff.

**Lag pairs.** The best-evidence pair is one of (1,3), (2,5), (4,7) for 95.3%, 80.2%, 100% and 100% of train machines (stable, short history, drift, shared change). For validation the shares are 97.4%, 81.2%, 100% and 99.5%. Every cross-validation fold independently picked exactly these three pairs.

**Train-only cross-validation** (stratified 4-fold over whole episodes). The pairs, rate priors and size profiles were refitted per fold with no validation data; the settings were kept as chosen on validation. The table shows the Brier gain over the pooled reference, which estimates the reduction in squared error against the true probabilities:

| | Stable | Short | Drift | Shared | All |
|---|---|---|---|---|---|
| Train cross-validation | 0.194 | 0.172 | 0.191 | 0.203 | 0.190 |
| Validation | 0.180 | 0.160 | 0.166 | 0.182 | 0.172 |
| Original blend, train cross-validation | 0.007 | 0.002 | 0.014 | 0.014 | 0.009 |

Every fold and condition gains between 0.152 and 0.224. Generalization within train is at least as good as on validation.

**Size profiles fitted from validation targets.** This looks permitted: RULES.md allows "organizer-released data" and forbids only test answers and private files. It does make the validation score partly in-sample. Using train-only profiles gives 55.36 instead of 55.60 (−0.24). The profiles are exact simulator constants: two vectors per type, identical to 4 decimals across all 768 validation queries and in both halves. So using the exact values should help on test rather than overfit.
