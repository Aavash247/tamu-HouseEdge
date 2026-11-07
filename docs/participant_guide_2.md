# House Edge guide (part 2)


### How the points are calculated

1. Add the five squared probability errors for each query.
2. Average within each episode, then across episodes in each condition.
3. Divide each condition's error by the frozen pooled baseline's error for that condition, with a minimum denominator of `1e-9`.
4. Average the four ratios to get `R`.

```text
R = average across conditions of error / max(reference_error, 0.000000001)
round points = round maximum / (1 + R)
final points = main points + adaptation points
```

The four conditions are **stable, short history, drift, and shared change**. Each counts equally. Each round has its own frozen reference model.

| Example | Main | Round two | Total |
| --- | --- | --- | --- |
| Same error as the reference (`R = 1` in both rounds) | 30 | 20 | 50 |
| Half the normalized error (`R = 0.5` in both rounds) | 40 | 26.67 | 66.67 |
| Exact probabilities (`R = 0` in both rounds) | 60 | 40 | 100 |

These are scoring examples, not predictions of how teams will do. Points aren't accuracy percentages: 60 points doesn't mean you guessed 60% of spins correctly.

Use the checker's validation points, or `validation.contest.technical_points` in `run.json`, to compare models. The older generic `score` field is a different diagnostic and isn't used for ranking. Higher points and lower errors are better.

## 5. Experiments

Save your baseline results, change one thing, and compare on the same validation episodes. Look at each condition as well as the total. A shorter history window might help changing machines and hurt stable ones.

| Idea | First thing to try |
| --- | --- |
| Recent spins matter more | Compare history windows or give recent observations more weight. |
| Short histories are noisy | Blend a machine's estimate with an average from similar machines. |
| Order matters | Add earlier payouts or pairs of payouts at different tick lags. |
| Patterns change | Compare short and long memory, or try a change detector. |
| Machines share patterns | Use summaries of other machines' earlier results in the episode. |
| Models make different mistakes | Blend their probabilities and choose weights using validation data. |

### Run the included experiments

```sh
python run_experiments.py --out experiments-01
python check_submission.py experiments-01/submission.csv
```

This tries 12 settings. It chooses a model using half the validation episodes in each condition, then checks that model on the other half. It writes `submission.csv`, `validation_predictions.csv`, and `experiments.json`. Edit `starter_candidates()` in `slot_forecasting/experiments.py` to add your own candidates.

Once you start tuning against that second half, it stops being an independent check. Keep whole episodes together and look for future information accidentally getting into your features before trusting a large improvement.

Keep a short log for yourselves: what changed, the score, runtime, and whether you kept it. It isn't graded. Sequence models and other more involved approaches are welcome, but the score only depends on the forecasts. Leave time to package and test the model you choose. `EXPERIMENTS.md` has more ideas.

## 6. Round two

**New data: Saturday, November 7 at 9 PM Central.**

**Submission deadline: Sunday, November 8 at 11 AM Central.**

Main submissions close at 9 PM Saturday. Everyone then gets the second kit, with fresh training data, validation answers, and test histories. Some patterns have changed. You have **14 hours** to update your model and submit the new predictions.

You can reuse main-round data, code, and saved models. Refit the same model or try something different. Round two is scored directly out of 40; deliberately weakening your main entry won't earn an improvement bonus.

1. Keep your final main files and Form confirmation.
2. Extract the new kit into a separate folder. Check that `public/metadata.json` says `adaptation`.
3. Copy over your code and model files without overwriting the new `public/` data.
4. Run your existing approach on the new validation data before changing it.
5. Check the new test CSV, package the matching code, and submit through the **Adaptation Google Form**.

### A suggested work plan

| When | What to do |
| --- | --- |
| Saturday, soon after 9 PM | Download the kit and get a first valid run. Check that your code uses the new data. |
| Saturday night | Compare a few changes on the new validation set. Save a working entry before stopping. |
| Sunday morning | Review results, choose a model, and upload with time to fix problems. |
| Before 11 AM Sunday | Confirm the correct CSV and ZIP were submitted through the Adaptation Form. |

The deadline is the same whether you work overnight or in the morning. The Form must receive your response before 11 AM; validation and code checks happen afterward.

Your main CSV won't work for round two because the query IDs are different. A missing or invalid round-two entry earns zero for that round. You need a valid main entry for an overall ranking. Exact tied totals share a rank; organizers will announce how prize ties are handled before kickoff.

## 7. Submitting

**Submit through Google Forms, one Form per round.** Organizers will announce the Main and Adaptation links. Upload two files: your test CSV and the code ZIP that recreates it.

### Submit your files

1. Open the correct round's Form and sign in with your Google account. Google Forms file uploads require sign-in.
2. Choose your team from the list. Use the same team for every submission.
3. Attach your test CSV (up to **10 MB**) and code ZIP (up to **100 MB**).
4. Click **Submit** and wait for the confirmation page. Attaching files without clicking Submit does not turn them in.
5. Keep the exact CSV and ZIP you submitted, plus the confirmation. Save the response email too if the Form sends one.

You may submit again before the cutoff. **Your team's latest Form response received before the deadline counts, even if its files are invalid or its code later fails.** There is no automatic fallback to an older entry. Run the local checker before submitting a replacement.

The Form records when it receives each response. Organizers use that timestamp and the published Central-time cutoff. A confirmation shows receipt, not a passing file check, a score, or eligibility for awards. There is no live reproduction-status page. Hidden test scores stay private while submissions are open.

If an upload fails, preserve the files and exact error and contact organizers before cutoff. If you are unsure whether a response arrived, ask the submissions organizer to check your team and round. Don't assume an unfinished upload or a file sent in chat will count.

### Package a custom model

For the starter, use the `build_entry.py --starter` command in Section 1. For your own model, prepare a folder like this:

```text
my-entry/
  README.md
  reproduce.json
  predict.py
  model files and configuration
  required source modules and requirements.txt, if needed
```

The last two lines describe files to include; use their actual filenames. Keep the forecast CSV separate. Leave out unrelated data, secrets, and temporary outputs.

Put this in `reproduce.json` at the folder's root:

```json
{
  "image": "python",
  "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"]
}
```

Your `predict.py` must accept `--data` and `--out`. The command is an argument list, not a shell string. It runs from `/submission`, where your code is read-only. The paths are:

- `/data`: the current round's public data, read-only.
- `/main_data`: main-round public data, available during adaptation.
- `/tmp`: writable scratch space, limited to 128 MiB.
- `/output`: writable output space, limited to 32 MiB. Write the final CSV to `/output/submission.csv`.

The runner has no network access, hidden test answers, or organizer credentials. It doesn't receive your uploaded forecast CSV as an input.

Keep `"image": "python"` in `reproduce.json`. The base environment is Python 3.13 with the standard library. If you need extra packages, include a `requirements.txt` with exact versions at the ZIP root. Organizers prepare those dependencies in an isolated environment before the offline prediction run. Include your own source modules and saved model files. Ask early about unusual dependencies; a GPU is not guaranteed. Prediction must not download anything.

Test your command locally from the source folder, replacing the example paths with your kit's public data folder and a fresh output path:

```text
python predict.py --data "/absolute/path/to/kit/public" --out "/absolute/path/to/check/submission.csv"
```

Run the checker on that output from the kit folder. Make sure it matches the CSV you plan to upload. Saving a trained checkpoint is usually easier than retraining during the check, especially if training is slow or uses randomness.

Package the source folder from the kit folder:

```text
python build_entry.py --source my-entry --out custom-entry.zip
```

Keep the ZIP outside `my-entry/`. It must open directly to `README.md` and `reproduce.json`, without an extra enclosing folder. Limits are 5,000 entries, 100 MB compressed, and 512 MiB unpacked. Symlinks, encrypted files, unsafe paths, and duplicate paths are rejected.

Default runtime limits are **30 minutes, two CPU cores, and 16 GiB RAM**. The largest allowed difference per probability is `1e-8` after CSV normalization. The final CSV may be at most 10 MB. These are the event limits; the Form may enforce a stricter file-type check. Here MB means 1,000,000 bytes.

| Item | Include |
| --- | --- |
| Team and round | Team ID and `main` or `adaptation`. |
| Forecast CSV | Test predictions for this round. |
| Code ZIP | README, reproduce.json, source, dependencies, settings, and saved model files. |
| README | Dependency versions, seeds, data paths, training details, and the command to recreate the CSV. |

Use `SUBMISSION_README.md` as a starting point. A short setup README is enough. There is no separate report or video for this scoring process; follow any additional event-wide requirements announced by organizers.

### CSV format

```csv
query_id,p_0,p_1,p_2,p_5,p_20
example_query,0.60,0.20,0.12,0.06,0.02
```

Replace the example ID with the actual IDs from this round's `public/test/queries.csv`.

- Include every test query exactly once, with no extras. Row order doesn't matter.
- Use exactly these six columns, without a dataframe index or extra columns.
- Probabilities must be finite numbers between 0 and 1 and add up to 1 within `1e-6`.
- Save as UTF-8 CSV. Renaming an Excel workbook won't convert it.
- Keep enough decimal places to preserve your probabilities.

### Before you upload

- Check the exact CSV you're uploading against the right round's test data.
- Recreate it from a fresh folder using the code ZIP and README.
- Include all dependencies, model files, and settings. Don't rely on an old notebook session or paths that only exist on your computer.
- If you change the model or preprocessing, rebuild both the CSV and ZIP.
- Leave time for both uploads and clicking Submit before cutoff.

### After you upload

Keep your Form confirmation and exact files. Main responses must arrive before 9 PM Saturday; adaptation responses before 11 AM Sunday. You cannot replace main files after round two opens.

After each cutoff, organizers select each team's latest response and validate its files. They score valid forecasts on private test answers, combine the two rounds, and rerun prize contenders' code before awards. A received response can still fail validation or reproduction. Unresolved organizer-side failures must be reviewed before final rankings; use the announced correction process if something goes wrong.

## 8. Troubleshooting

### Setup and modeling

| Problem or question | What to try |
| --- | --- |
| `python` isn't found | Try `python3` or `py`, and check that it's version 3.11+. |
| Script or module not found | Fully unzip the kit and run from its folder. Keep `slot_forecasting/` beside the scripts. |
| Missing data or `Incomplete public dataset` | Check that `public/` fully extracted. Use `--data "path/to/public"` if it's elsewhere. |
| `query_id mismatch` | Check the round and split. Validation files need `--split validation`; uploads need test IDs. |

> Continued in [participant_guide_3.md](participant_guide_3.md).
