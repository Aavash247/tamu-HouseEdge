"""Grid-search pooled_recent half_life / prior_strength; save the best run."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.contest_data import compact_contest, export_run, load_contest, score_predictions
from slot_forecasting.io import write_json
from slot_forecasting.schema import ValidationError
from slot_forecasting.starter import public_directory

HALF_LIVES = (8.0, 12.0, 16.0, 24.0, 48.0, 96.0)
PRIOR_STRENGTHS = (5.0, 10.0, 20.0, 40.0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("sweep-pooled-recent-best"))
    args = parser.parse_args(argv)
    try:
        bundle = load_contest(public_directory(args.data))
        rows, best = [], None
        for half_life in HALF_LIVES:
            for prior_strength in PRIOR_STRENGTHS:
                started = perf_counter()
                model = FrequencyForecaster(
                    "pooled_recent", prior_strength=prior_strength, half_life=half_life,
                ).fit(bundle["train_q"], bundle["train_h"], bundle["labels"])
                contest = score_predictions(model.predict(bundle["val_q"], bundle["val_h"]), bundle)
                row = {
                    "half_life": half_life, "prior_strength": prior_strength,
                    "technical_points": contest["technical_points"],
                    "mean_loss_ratio": contest["mean_loss_ratio"],
                    "seconds": perf_counter() - started,
                }
                rows.append(row)
                print(
                    f"half={half_life:5g} prior={prior_strength:5g} "
                    f"pts={contest['technical_points']:.4f} R={contest['mean_loss_ratio']:.4f}"
                )
                key = (contest["mean_loss_ratio"], -contest["technical_points"])
                if best is None or key < best[0]:
                    best = (key, model, contest, row)
        export_run(
            args.out, best[1], bundle, best[2],
            {"method": "pooled_recent", "half_life": best[3]["half_life"],
             "prior_strength": best[3]["prior_strength"]},
            kind="FrequencyForecaster",
        )
        top = sorted(rows, key=lambda r: r["mean_loss_ratio"])[:5]
        write_json(args.out / "sweep.json", {
            "best": best[3], "contest": compact_contest(best[2]), "top5": top,
        })
        print(f"Best: half_life={best[3]['half_life']} prior={best[3]['prior_strength']} "
              f"pts={best[3]['technical_points']:.4f}")
        print(f"Saved: {args.out}")
        return 0
    except (ValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
