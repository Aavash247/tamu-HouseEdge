"""Sweep NeighborPooledRecent knobs; optionally blend with tuned pooled_recent."""

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.blend import BlendNeighborPooled, blend_predictions
from slot_forecasting.contest_data import compact_contest, export_run, load_contest, score_predictions
from slot_forecasting.io import write_json
from slot_forecasting.neighbor import NeighborPooledRecent
from slot_forecasting.schema import ValidationError
from slot_forecasting.starter import public_directory

HALF_LIVES = (12.0, 16.0, 24.0, 48.0)
PRIOR_STRENGTHS = (10.0, 20.0, 40.0)
NEIGHBOR_WEIGHTS = (0.15, 0.25, 0.35, 0.5, 0.75)
BLEND_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _pooled_settings(path):
    half, prior = 24.0, 20.0
    for candidate in (path, Path("sweep-pooled-recent-best/run.json")):
        if candidate and Path(candidate).is_file():
            settings = json.loads(Path(candidate).read_text(encoding="utf-8"))["settings"]
            return float(settings["half_life"]), float(settings["prior_strength"])
    return half, prior


def _search_neighbors(bundle):
    best = None
    top = []
    for half_life in HALF_LIVES:
        for prior_strength in PRIOR_STRENGTHS:
            for neighbor_weight in NEIGHBOR_WEIGHTS:
                for same_type_only in (True, False):
                    started = perf_counter()
                    model = NeighborPooledRecent(
                        prior_strength=prior_strength, half_life=half_life,
                        neighbor_weight=neighbor_weight, same_type_only=same_type_only,
                    ).fit(bundle["train_q"], bundle["train_h"], bundle["labels"])
                    pred = model.predict(bundle["val_q"], bundle["val_h"])
                    contest = score_predictions(pred, bundle)
                    row = {
                        "half_life": half_life, "prior_strength": prior_strength,
                        "neighbor_weight": neighbor_weight, "same_type_only": same_type_only,
                        "technical_points": contest["technical_points"],
                        "mean_loss_ratio": contest["mean_loss_ratio"],
                        "seconds": perf_counter() - started,
                    }
                    top.append(row)
                    print(
                        f"half={half_life:5g} prior={prior_strength:5g} nw={neighbor_weight:4g} "
                        f"same_type={same_type_only!s:5} pts={contest['technical_points']:.4f}"
                    )
                    key = (contest["mean_loss_ratio"], -contest["technical_points"])
                    if best is None or key < best[0]:
                        best = (key, model, contest, row, pred)
    top = sorted(top, key=lambda r: r["mean_loss_ratio"])[:5]
    return best, top


def _search_blend(bundle, best_neighbor, pooled, pooled_half, pooled_prior):
    pooled_val = pooled.predict(bundle["val_q"], bundle["val_h"])
    best = (
        best_neighbor[0], "neighbor", best_neighbor[1], best_neighbor[2],
        {**{k: best_neighbor[3][k] for k in (
            "half_life", "prior_strength", "neighbor_weight", "same_type_only")},
         "blend_weight": 1.0, "pooled_half_life": pooled_half,
         "pooled_prior_strength": pooled_prior},
    )
    blend_rows = []
    for weight in BLEND_WEIGHTS:
        if weight >= 1.0:
            pred = best_neighbor[4]
        elif weight <= 0.0:
            pred = pooled_val
        else:
            pred = blend_predictions(best_neighbor[4], pooled_val, weight)
        contest = score_predictions(pred, bundle)
        blend_rows.append({
            "blend_weight_on_neighbor": weight,
            "technical_points": contest["technical_points"],
            "mean_loss_ratio": contest["mean_loss_ratio"],
        })
        print(f"blend_w_neighbor={weight:4g} pts={contest['technical_points']:.4f}")
        key = (contest["mean_loss_ratio"], -contest["technical_points"])
        if key < best[0]:
            settings = {
                **{k: best_neighbor[3][k] for k in (
                    "half_life", "prior_strength", "neighbor_weight", "same_type_only")},
                "blend_weight": weight, "pooled_half_life": pooled_half,
                "pooled_prior_strength": pooled_prior,
            }
            if weight >= 1.0:
                model, kind = best_neighbor[1], "neighbor"
            elif weight <= 0.0:
                model, kind = pooled, "pooled_recent"
            else:
                model, kind = BlendNeighborPooled(best_neighbor[1], pooled, weight), "blend"
            best = (key, kind, model, contest, settings)
    return best, blend_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("neighbor-best"))
    parser.add_argument("--pooled-settings", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        bundle = load_contest(public_directory(args.data))
        pooled_half, pooled_prior = _pooled_settings(args.pooled_settings)
        pooled = FrequencyForecaster(
            "pooled_recent", prior_strength=pooled_prior, half_life=pooled_half,
        ).fit(bundle["train_q"], bundle["train_h"], bundle["labels"])
        best_neighbor, neighbor_top = _search_neighbors(bundle)
        best, blend_rows = _search_blend(
            bundle, best_neighbor, pooled, pooled_half, pooled_prior,
        )
        export_run(args.out, best[2], bundle, best[3], best[4], kind=best[1])
        write_json(args.out / "sweep.json", {
            "best_kind": best[1], "best_settings": best[4],
            "contest": compact_contest(best[3]),
            "neighbor_top5": neighbor_top, "blend_grid": blend_rows,
            "pooled_side": {"half_life": pooled_half, "prior_strength": pooled_prior},
        })
        print(f"Best kind={best[1]} pts={best[3]['technical_points']:.4f}")
        print(f"Saved: {args.out}")
        return 0
    except (ValidationError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
