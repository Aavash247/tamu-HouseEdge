# House Edge entry: lag-pair flip HMM

- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13), `requirements.txt` (numpy==2.5.4)
- CPU only, one core, under 1 GB RAM. Test prediction takes about 65 s.
- Deterministic: no randomness and no seeds. Refitting gives the same `model.json`.

## Recreate the forecast

```sh
python predict.py --data /data --out /output/submission.csv
```

Loads `model.json` and predicts `/data/test` offline. `--split validation` predicts validation instead.

## Refit (optional)

```sh
python fit.py --data /path/to/public
```

Learns the candidate lag pairs and rate priors from train histories, and the payout-size
profiles from the validation targets. Forecasts only use a machine's own ticks before the
cutoff, and missing ticks stay unobserved.

## Model

- **Win chance.** Whether the outcomes at two lags agree, for one of the pairs (1,3), (2,5) or (4,7),
  switches P(win) between about 0.03 and 0.97. A two-state HMM tracks when that relationship
  flips (drift, shared change). A 2-cell model (agree/differ) and a full 2x2 model are averaged 50/50,
  each combining the pairs and a no-lag change-point model by Bayesian model averaging.
- **Payout size.** Each type has two fixed size profiles. Past winning payouts, with a 60-tick
  half-life, give the mixture weight.

| Code | Contents |
|---|---|
| `data.py` | CSV reading and writing, per-machine lag arrays |
| `win.py` | Rate grids, flip HMMs, no-lag model |
| `size.py` | Size profiles and their mixture |
| `forecaster.py` | Fitting, priors and prediction |

## Validation

55.60 / 60 (official checker). Loss ratios: stable 0.069, short history 0.098, drift 0.077, shared change 0.073.
