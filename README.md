# House Edge

**Participant guide · TAMU Datathon 2026**

**November 7–8, 2026 · All times Central**

Use the slot-machine logs to predict the odds of the next payout: **0×, 1×, 2×, 5×, or 20×**. You'll submit a CSV of probabilities and the code needed to recreate it. Your score depends on how close those probabilities are to the machine's actual odds.

**GitHub:** [https://github.com/tamu-datathon-org/slots-rl](https://github.com/tamu-datathon-org/slots-rl)

Start with the participant kit released by organizers. The repo is a project reference and may require access; you don't need repo access to compete. Practice data and the slot-machine demo are separate from the competition.

## Dates and links

**Round two opens Saturday at 9 PM and closes Sunday at 11 AM.** You'll have **14 hours** to work with the new data, including overnight and Sunday morning. Main submissions close when round two opens.

| What | When or where |
| --- | --- |
| Event hacking begins | Saturday, November 7 at 11 AM |
| Main submissions close | Saturday, November 7 at 9 PM |
| Round-two data releases and submissions open | Saturday, November 7 at 9 PM |
| Round-two submissions close | Sunday, November 8 at 11 AM |
| Event hacking ends | Sunday, November 8 at noon |
| Judging begins | Sunday, November 8 at 12:30 PM |
| Participant kit | Download link to be posted with the challenge announcement |
| Submission Forms | Separate Main and Adaptation links will be announced |
| Help | Challenge help channel to be announced |

Times above are **Central (America/Chicago)**. The [event schedule](https://tamudatathon.com/) lists the wider event activities; the two submission deadlines above are specific to House Edge.

Upload early! Attach both files and click Submit before the deadline. Wait for the Google Form's confirmation page and keep your exact files. Confirmation means the response was received; file validation and code checks happen afterward.

## What to predict

Each query asks for **one machine's next spin**, at its `cutoff` tick. Return five probabilities in this order: `0, 1, 2, 5, 20`. They should add up to 1.

For example, `[0.60, 0.20, 0.12, 0.06, 0.02]` gives a 60% chance of no payout and a 2% chance of a 20× payout. These probabilities also describe your uncertainty, so you don't need a separate confidence interval.

Every spin costs one chip. Payouts include the original wager: 0× loses the chip, 1× returns it, and 5× returns five chips, for a net gain of four. The competition scores your predictions, not the number of chips won.

Use whichever model you want. The starter runs on a CPU with Python; a neural network or GPU is optional. You don't need to build an agent or choose bets. The interactive demo is just a way to try the idea out and doesn't affect your score.

## 1. Getting started

Download and fully unzip the **main participant kit**, then open a terminal in the extracted folder. Keep the scripts, `slot_forecasting/`, and `public/` together. Run the commands from the folder containing `run_starter.py`.

You'll need **Python 3.11 or newer**. The starter and checker use the standard library, so there's nothing else to install. You don't need Docker for these steps.

```sh
python --version
```

If your system uses `python3` or `py`, use that instead of `python` below.

```sh
python run_starter.py --out outputs
python check_submission.py outputs/submission.csv
python check_submission.py outputs/validation_predictions.csv --split validation
python build_entry.py --starter outputs --out entry.zip
```

### What these commands do

The starter trains a baseline and compares six methods on validation data. **It saves predictions from `pooled` by default**, even if another method scores better. You can choose another method with `--method`.

The two checker commands check your test file and score your validation file. A local check doesn't upload anything or reveal your final test score.

The last command packages your saved starter model and code. Upload **`outputs/submission.csv` and `entry.zip`** as two separate files. The `--starter` option works with `run_starter.py` outputs. For your own model or the experiment runner, follow the packaging instructions in Section 7.

| File | What it contains |
| --- | --- |
| `submission.csv` | Test predictions to upload. |
| `validation_predictions.csv` | Predictions you can score locally. |
| `baseline.json` | The saved model. |
| `run.json` | Settings, comparisons, runtime, file hashes, and reproduction checks. |
| `validation_score.json` | Detailed validation scores. |
| `comparison/` | Validation predictions from each baseline. |

Keep this first run. Use a new output folder for each experiment so you can go back to an earlier model. If you prefer notebooks, `starter.ipynb` covers the same steps; you'll need your own Jupyter installation.

## 2. The data

The kit includes a README, `RULES.md`, `DATA.md`, `EXPERIMENTS.md`, a submission README template, starter scripts, and the `public/` data folder.

| Folder | What's inside | Use it for |
| --- | --- | --- |
| `public/train/` | Queries, histories, observed next-spin payouts, and condition labels. | Training your model. |
| `public/validation/` | Queries, histories, observed payouts, true probabilities, conditions, and reference forecasts. | Comparing models and choosing settings. |
| `public/test/` | Queries and histories, without answers or condition labels. | Making your final predictions. |

Each split includes a `sample_submission.csv`. Check `public/metadata.json` for the data version, season, round, points, and sizes. `SHA256.json` lists hashes of the original kit files.

### Terms used in the files

| Term | Meaning |
| --- | --- |
| Query | One request for five payout probabilities. |
| Episode | Related machines observed at the same cutoff. |
| Tick | A time step in the gameplay logs. |
| Label / outcome | One observed payout: 0, 1, 2, 5, or 20. |
| Target probabilities | The simulator's actual odds. Public for validation, hidden for test. |
| Condition | Stable, short history, drift, or shared change. Used to compare performance. |
| Adaptation | The name used in the files and Google Form for round two. |

### Input columns

`queries.csv` contains:

| Column | Meaning |
| --- | --- |
| `query_id` | Copy this ID into your prediction CSV. |
| `episode_id` | The group of related machines. |
| `machine_id` | The machine within that episode. Group histories by both IDs. |
| `machine_type` | `steady`, `balanced`, or `jackpot`. |
| `signal` | An observed feature: -1, 0, or 1. |
| `cutoff` | The tick to predict. Use inputs from earlier ticks only. |

`history.csv` contains `episode_id, machine_id, tick, payout`. Training and validation also have `labels.csv`, which maps each `query_id` to its observed next payout, named `outcome`.

Output columns always follow payout order **0, 1, 2, 5, 20**. Write probabilities between 0 and 1, not percentages.

### Using the history

- For cutoff 200, use ticks below 200. Another machine's result at tick 200 is unavailable too.
- If you make a training example for tick 150, use inputs from below 150. The payout at 150 can be its label, but not an input.
- A missing tick is unknown, not a zero payout. A two-tick lag isn't necessarily the second-most-recent row.
- Use other machines' earlier results only within the same episode.
- Condition labels are available for training and validation only. Your test predictor must work without them.

Train, validation, and test contain separate episodes. Keep whole episodes together if you make your own splits; related machines shouldn't appear on both sides of an evaluation split.

Cutoffs range from 192 to 256. Long histories include every earlier tick. Short histories include 40 recent consecutive ticks and 24 sampled older ticks, so always check the actual tick values.

Past payouts can interact at delays of up to eight ticks. Machines of the same type can respond differently, even in opposite directions. Win chances and payout sizes can change, sometimes across several machines together. These are patterns in the challenge's simulator, not a claim about real casino machines.

See `DATA.md` for details. Use validation answers to compare ideas, but remember that repeated tuning can overfit that data.

## 3. Your model

### Try another baseline

```sh
python run_starter.py --method recent --out recent-run
python check_submission.py recent-run/validation_predictions.csv --split validation
python check_submission.py recent-run/submission.csv
```

The included methods are `global`, `context`, `empirical`, `recent`, `pooled`, and `pooled_recent`. You're free to use other methods and libraries.

### Write your own script

Save this as `my_model.py` in the kit folder. It loads the data, fits a model, and writes validation and test predictions. Replace the model with your own when you're ready.

```python
from pathlib import Path
from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.io import load_inputs, read_labels, write_submission

DATA = Path("public")
OUT = Path("custom-run")
if OUT.exists():
    raise SystemExit("Choose a new output folder before running again.")

train_queries, train_history = load_inputs(DATA / "train")
labels = read_labels(DATA / "train/labels.csv", train_queries)
model = FrequencyForecaster(method="pooled_recent", half_life=24)
model.fit(train_queries, train_history, labels)

for split, filename in (
    ("validation", "validation_predictions.csv"),
    ("test", "submission.csv"),
):
    queries, history = load_inputs(DATA / split)
    predictions = model.predict(queries, history)
    write_submission(OUT / filename, predictions)
```

Then run:

```sh
python my_model.py
python check_submission.py custom-run/validation_predictions.csv --split validation
python check_submission.py custom-run/submission.csv
```

Here, `predict()` returns a dictionary from each `query_id` to five probabilities. The writer checks the numbers and keeps their precision; the checker makes sure every required query appears exactly once.

Save your dependency versions, seeds, settings, and trained model files. This example runs locally; to submit it, you'll also need a script that accepts the data and output paths in Section 7. Include any project modules it imports in your ZIP.

## 4. Scoring

**Main: 60 points. Round two: 40 points.** Add them for your final score. All points come from forecasts. There are no points for a writeup or presentation, but your code must reproduce your results and follow the rules for you to qualify for awards.

We compare your probabilities with the simulator's actual probabilities. That keeps a randomly drawn jackpot from deciding the score. Some machine state is hidden, so don't expect to recover the exact odds from the logs.

### A simple example

If you predict `[0.60, 0.20, 0.12, 0.06, 0.02]` and the true probabilities are `[0.50, 0.25, 0.15, 0.08, 0.02]`, the sum of squared differences is `0.0138`. It stays the same whichever payout happens on that spin.

Assign a reasonable chance to each payout. Picking only the most likely outcome throws away information and can hurt your score when several outcomes are plausible.

> Continued in [docs/participant_guide_2.md](docs/participant_guide_2.md).
