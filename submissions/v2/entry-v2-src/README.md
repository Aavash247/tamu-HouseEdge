# House Edge entry v2: lag-pair flip-HMM

- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13), `requirements.txt` (numpy==2.5.4)
- Hardware: CPU only, single core, <1 GB RAM. Test prediction takes about 10 s.
- Deterministic: no randomness, no seeds needed.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

`predict.py` loads the included `model.json` and predicts `/data/test`. It needs no network access.

## Training (optional; model.json is already included)

```sh
python fit.py --data /path/to/public
```

`fit.py` picks the candidate lag pairs and their prior from the training histories (the pair with the best evidence for each train machine), and the payout-size profiles from the validation targets. Every forecast uses only that machine's own ticks strictly before the cutoff, and missing ticks stay unobserved.

## Model

1. **Win chance:** for each candidate lag pair (i, j), whether the win/loss outcomes at t-i and t-j agree drives P(win). A two-state hidden Markov model tracks the sign of that relationship, which can flip during drift or a shared change. Per-cell win rates are averaged over a 25x25 grid. Pairs and a no-lag change-point model are combined by Bayesian model averaging on each machine's history.
2. **Payout size given a win:** a posterior-mean mixture of each machine type's two size profiles, using the machine's past winning payouts with recency weighting.

Validation: 54.92 / 60 (R = 0.0925). The previous blend entry scored 30.91.
