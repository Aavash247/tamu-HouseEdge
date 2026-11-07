# Synthetic evaluation environment

This is an independent simulator used to test how well the House Edge model architecture generalizes. It does not modify `entry-final/` or `submissions/final/`; the model code is imported read-only.

```sh
sh work/synthetic_audit/run_all.sh 1 2 3 4 5      # any list of seeds; PY=... to choose the interpreter
```

| File | Purpose |
|---|---|
| `sim.py` | Seeded simulator; writes competition-format CSVs plus `targets.csv` (true probabilities) and `param_oracle.csv` |
| `evaluate.py` | Fits the model on synthetic **train** episodes only; scores unseen **test** episodes against true probabilities |
| `report.py` | Aggregates results across seeds into `results/report.txt` |
| `check_missing.py` | Diagnostic: error with vs without unobserved lags at the query tick |
| `data/seed*/` | Generated data (about 18 MB per seed; regenerate with `sim.py`) |
| `results/seed*.json`, `results/report.txt` | Per-episode errors and the summary |

**Design.** Each seed has 128 train episodes (32 per seen scenario) and 160 test episodes (the same 4 scenarios plus 32 `unseen_pattern`), with 8 machines each. The split is by episode. The score is the sum of squared errors to the simulator's true probabilities, averaged within each episode and then across episodes, as in the competition.

**Baselines.**
- uniform
- pooled frequency (the competition's reference, prior strength 20)
- `oracle_params_no_hidden`: knows every parameter but not the hidden state at the query tick, so it is the best achievable
- `oracle_true`: the true probabilities, which score 0 by definition

## Results (5 seeds)

Ratios are error relative to pooled frequency, given as mean [min–max] over seeds.

| Scenario | Model ratio | Achievable (parameter oracle) | Share of achievable gap closed |
|---|---|---|---|
| stable | 0.112 [0.090–0.136] | 0.038 | 92% |
| short_history | 0.062 [0.054–0.079] | 0.037 | 97% |
| drift | 0.111 [0.101–0.115] | 0.038 | 93% |
| shared_change | 0.106 [0.095–0.124] | 0.038 | 93% |
| **unseen_pattern** | **0.501 [0.471–0.542]** | 0.035 | **52%** |

Implied points on the four seen scenarios: model 54.66 (54.41–55.16 across seeds); pooled 30.00; uniform 22.93; parameter oracle 57.81.

**Weakness found: missing ticks at query time.** On seed 1 (stable, drift and shared change), queries where any of lags 1–8 is unobserved have a mean error of 0.082, against 0.015 when all are observed. This explains why stable scores worse than short_history here: short histories always keep their 40 most recent ticks. Competition long histories are complete per the kit docs, so this does not affect the main round as documented, but it is a risk if a future data release drops recent ticks.
