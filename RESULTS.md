# Experiment log (Aayush branch)

All scores come from the official validation scorer (`check_submission.py --split validation`, or `work/harness.py`, which matches it exactly). Loss ratios are relative to the frozen pooled reference; lower is better.

| # | Model | Points / 60 | R | stable | short | drift | shared | Kept |
|---|---|---|---|---|---|---|---|---|
| 0 | Original `winner-blend` (neighbor + pooled_recent) | 30.91 | 0.941 | 0.960 | 0.999 | 0.946 | 0.859 | baseline |
| 1 | Best lag pair, 2 static cells | 32.20 | 0.863 | 0.084 | 0.178 | 1.788 | 1.405 | no, breaks under drift |
| 2 | Pair model, recency half-life 25 | 46.94 | 0.278 | 0.104 | 0.186 | 0.398 | 0.426 | no |
| 3 | Pair model + Bayesian online change points (BOCPD) | 53.64 | 0.119 | 0.088 | 0.172 | 0.112 | 0.103 | no, superseded |
| 4 | + two-profile size mixture | 54.16 | 0.108 | 0.085 | 0.153 | 0.104 | 0.090 | yes |
| 5 | Orientation-flip HMM instead of BOCPD (`submissions/v2`) | 54.92 | 0.0925 | 0.078 | 0.131 | 0.083 | 0.078 | yes |
| 6 | + empirical-Bayes rate prior per type, rho = 0.001 | 55.40 | 0.0830 | 0.076 | 0.102 | 0.078 | 0.075 | yes |
| 7 | + 4-cell flip-HMM, 50/50 with 2-cell (`submissions/v3` = final) | 55.60 | 0.0791 | 0.069 | 0.098 | 0.077 | 0.073 | **final** |

Split-half check (odd vs even validation episodes): original 30.85 / 30.96, v2 55.41 / 54.41, final 55.85 / 55.34.

## What the data showed

- Win chance is almost fully set by one lag pair per machine, either (1,3), (2,5) or (4,7). Whether the two lagged win/loss outcomes agree switches P(win) between about 0.03 and 0.97. Learning that pair from each machine's own history cuts the win-probability error on stable machines by about 35x.
- Drift and shared change flip the sign of that relationship (cell (a,b) behaves like (a,1-b)). In shared-change episodes, all 8 machines flip at the same tick (within 1 tick). In drift, flip times are spread out.
- Payout size given a win follows exactly two profiles per machine type. The active profile is a hidden per-tick state that also raises win chance. No lag, neighbor, periodic or time-block feature predicts it beyond a machine-level tendency, so it is the main source of irreducible error.

## Tried and dropped

- Signal-specific priors: +0.02, not worth the extra parameters.
- Hidden Markov model for the size state, a learned size prior, and different size half-lives: at most +0.01.
- Size state driven by neighbor lags, payout-size lags, or periodic patterns: no signal above noise.
- Pooling flip timing across the episode: flips land 60 to 90 ticks before the cutoff, so per-machine models already catch them. Not implemented.
