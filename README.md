# House Edge: lag-pair flip HMM

TAMU Datathon 2026, slot-machine forecasting. Validation: **55.60 / 60** (the pooled reference scores 30).

| Folder | Contents |
|---|---|
| `entry/` | The submission: code, `model.json`, `reproduce.json`, setup README |
| `round2/run_round2.py` | One command for the adaptation round |
| `submissions/final/` | The main-round CSV, validation predictions and code ZIP as submitted |

The organizer kit (data, checker, `build_entry.py`) is not stored here. Download it and pass its path.

## Run

```sh
cd entry
python predict.py --data /path/to/kit/public --out submission.csv
python fit.py --data /path/to/kit/public          # optional refit of model.json
```

Round two, as soon as the new kit is out:

```sh
python round2/run_round2.py --kit /path/to/adaptation-kit --out round2-run
```

It refits on the new kit for four flip rates and three model families, keeps the main-round
model if it still scores best, then writes `round2-run/submission.csv` and `round2-run/entry.zip`
and checks that the ZIP reproduces the CSV.

## What the data showed

- Win chance is set by one lag pair per machine: (1,3), (2,5) or (4,7). Whether the two outcomes
  agree moves P(win) between about 0.03 and 0.97.
- Drift and shared change flip that relationship. In shared change all 8 machines flip within one tick.
- Payout size given a win follows two fixed profiles per type. Which one is active is hidden and is
  the main source of the remaining error.

## How we got there

| Model | Points / 60 | stable | short | drift | shared |
|---|---|---|---|---|---|
| Neighbor + pooled blend | 30.91 | 0.960 | 0.999 | 0.946 | 0.859 |
| Best lag pair, static | 32.20 | 0.084 | 0.178 | 1.788 | 1.405 |
| Pair model, recency weighted | 46.94 | 0.104 | 0.186 | 0.398 | 0.426 |
| + change-point detection | 53.64 | 0.088 | 0.172 | 0.112 | 0.103 |
| + two size profiles | 54.16 | 0.085 | 0.153 | 0.104 | 0.090 |
| Flip HMM instead of change points | 54.92 | 0.078 | 0.131 | 0.083 | 0.078 |
| + rate prior per type | 55.40 | 0.076 | 0.102 | 0.078 | 0.075 |
| **+ full 2x2 model, averaged 50/50** | **55.60** | **0.069** | **0.098** | **0.077** | **0.073** |

Loss ratios are against the pooled reference; lower is better.

## Overfitting checks

- No leakage: fake ticks at or after the cutoff, and scrambled neighbor machines, change predictions by 0.0.
- No ID memorization: `model.json` is keyed by machine type only.
- 4-fold cross-validation inside train, refitting everything per fold, picked the same three pairs
  every time and gained as much as validation does.
- The size profiles come from the validation targets, which the rules allow. Train-only profiles
  score 55.36, so the effect is small.
