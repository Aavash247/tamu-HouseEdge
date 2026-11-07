# House Edge entry: lag-pair flip-HMM

- Round: main
- Season ID: 5fffdc4d0c700bbf61c3f911
- Environment: `python` image (Python 3.13), `requirements.txt` (numpy==2.5.4)
- Tested on: macOS (Apple silicon), Python 3.13.7 with numpy 2.5.4
- Hardware: CPU only, single core, under 1 GB RAM. Test prediction takes about 65 s.
- Deterministic: no randomness, so no seeds. Refitting gives a byte-identical `model.json`.

## Recreate the forecast

Run from this folder:

```sh
python predict.py --data /data --out /output/submission.csv
```

`predict.py` loads the included `model.json` and predicts `/data/test` without network access.
Add `--split validation` to predict the validation split instead.

## Training (optional; model.json is already included)

```sh
python fit.py --data /path/to/public
```

`fit.py` learns three things:
1. The candidate lag pairs and their prior, from training histories (the pair with the best evidence for each train machine).
2. Empirical-Bayes priors over win-rate grids for each machine type, from the train machines' posteriors.
3. The payout-size profiles for each machine type, from the validation targets.

Every forecast uses only that machine's own observed ticks strictly before the cutoff. Missing ticks stay unobserved and are never treated as losses.

## Model

- **Win chance.** For each candidate lag pair (i, j), the win/loss outcomes at t-i and t-j form a 2x2 table that drives P(win), mostly as an XOR-style "agree vs disagree" effect. A two-state hidden Markov model tracks the orientation of that relationship, which flips during drift or a shared change. Two families are averaged 50/50:
  - a 2-cell (agree/disagree) model on a 25x25 rate grid
  - a full 4-cell model on an 11^4 rate grid, where a flip maps cell (a, b) to (a, 1-b)

  Within each family, the pairs and a no-lag change-point model are combined by Bayesian model averaging on the machine's history.
- **Payout size given a win.** Each type has two size profiles. The machine's past winning payouts, weighted with a 60-tick half-life, give a posterior over its profile mixture weight.

## Validation (official checker)

| Model | Points / 60 | R | stable | short history | drift | shared change |
|---|---|---|---|---|---|---|
| Previous blend entry | 30.91 | 0.941 | 0.960 | 0.999 | 0.946 | 0.859 |
| This entry | 55.60 | 0.0791 | 0.069 | 0.098 | 0.077 | 0.073 |

## Local check

```sh
python check_submission.py submission.csv
```

Result: valid (1536 test queries). A fresh-folder reproduction from the ZIP on Python 3.13 matched the CSV with a max difference of 0.0.
