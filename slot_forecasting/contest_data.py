"""Shared contest data loading / scoring / export for sweep scripts."""

from pathlib import Path

from .contest import contest_score
from .io import (
    load_inputs, read_labels, read_rows, read_submission, read_targets,
    write_json, write_submission,
)
from .rounds import VERSION, round_metadata
from .schema import ValidationError


def load_contest(public):
    metadata = round_metadata(public)
    train_q, train_h = load_inputs(public / "train")
    labels = read_labels(public / "train/labels.csv", train_q)
    val_q, val_h = load_inputs(public / "validation")
    test_q, test_h = load_inputs(public / "test")
    targets = read_targets(public / "validation/targets.csv")
    groups = {
        r["query_id"]: r["condition"]
        for r in read_rows(public / "validation/conditions.csv", ("query_id", "condition"))
    }
    reference = read_submission(public / "validation/reference.csv")
    return {
        "metadata": metadata,
        "train_q": train_q, "train_h": train_h, "labels": labels,
        "val_q": val_q, "val_h": val_h,
        "test_q": test_q, "test_h": test_h,
        "targets": targets, "groups": groups, "reference": reference,
        "max_points": metadata["max_points"],
    }


def score_predictions(predictions, bundle):
    return contest_score(
        predictions, bundle["targets"], bundle["groups"], bundle["reference"],
        max_points=bundle["max_points"], version=VERSION,
    )


def export_run(out, model, bundle, contest, settings, kind="model"):
    out = Path(out)
    if out.exists():
        raise ValidationError(f"Choose a fresh output folder: {out}")
    out.mkdir(parents=True)
    write_submission(out / "validation_predictions.csv", model.predict(bundle["val_q"], bundle["val_h"]))
    write_submission(out / "submission.csv", model.predict(bundle["test_q"], bundle["test_h"]))
    model.save(out / "baseline.json")
    write_json(out / "run.json", {
        "model": kind,
        "settings": settings,
        "contest": contest,
        "round": bundle["metadata"]["round"],
        "season_id": bundle["metadata"]["season_id"],
    })
    write_json(out / "validation_score.json", {"contest": contest})


def compact_contest(contest):
    return {
        "technical_points": contest["technical_points"],
        "mean_loss_ratio": contest["mean_loss_ratio"],
        "by_condition": {k: v["loss_ratio"] for k, v in contest["by_condition"].items()},
    }
