"""Train the selected blend winner, export forecasts, and build my-entry/."""

import json
import shutil
from pathlib import Path
import sys

from slot_forecasting.baseline import FrequencyForecaster
from slot_forecasting.blend import BlendNeighborPooled
from slot_forecasting.contest_data import load_contest, score_predictions
from slot_forecasting.io import read_submission, write_json, write_submission
from slot_forecasting.neighbor import NeighborPooledRecent
from slot_forecasting.starter import public_directory
from slot_forecasting.submission_contract import package_code

SETTINGS = {
    "half_life": 12.0, "prior_strength": 10.0, "neighbor_weight": 0.15,
    "same_type_only": True, "blend_weight": 0.25,
    "pooled_half_life": 16.0, "pooled_prior_strength": 10.0,
}
MODULES = ("__init__.py", "baseline.py", "schema.py", "io.py", "neighbor.py", "blend.py")


def _build_entry(out, contest, metadata):
    entry = Path("my-entry")
    if entry.exists():
        shutil.rmtree(entry)
    entry.mkdir()
    module = entry / "slot_forecasting"
    module.mkdir()
    for name in MODULES:
        shutil.copyfile(Path("slot_forecasting") / name, module / name)
    for name in ("baseline.json", "neighbor_model.json", "pooled_model.json"):
        shutil.copyfile(out / name, entry / name)
    (entry / "predict.py").write_text(
        "import argparse\nfrom pathlib import Path\n"
        "from slot_forecasting.io import load_inputs, write_submission\n"
        "from slot_forecasting.blend import BlendNeighborPooled\n"
        "p=argparse.ArgumentParser(); p.add_argument('--data', required=True)\n"
        "p.add_argument('--out', required=True); a=p.parse_args()\n"
        "m=BlendNeighborPooled.load(Path(__file__).parent/'baseline.json')\n"
        "q,h=load_inputs(Path(a.data)/'test'); write_submission(a.out, m.predict(q,h))\n",
        encoding="utf-8",
    )
    (entry / "reproduce.json").write_text(json.dumps({
        "image": "python",
        "command": ["python", "predict.py", "--data", "/data", "--out", "/output/submission.csv"],
    }, indent=2) + "\n", encoding="utf-8")
    (entry / "README.md").write_text(
        "# House Edge blend submission\n\nPython stdlib only.\n\n"
        f"Settings: {SETTINGS}\n\n"
        f"Validation points: {contest['technical_points']:.6f} / {metadata['max_points']}\n\n"
        "```\npython predict.py --data /data --out /output/submission.csv\n```\n",
        encoding="utf-8",
    )
    zip_path = Path("entry-winner-blend.zip")
    if zip_path.exists():
        zip_path.unlink()
    package_code(entry, zip_path)
    return entry, zip_path


def main():
    bundle = load_contest(public_directory(Path(".")))
    metadata = bundle["metadata"]
    neighbor = NeighborPooledRecent(
        prior_strength=SETTINGS["prior_strength"], half_life=SETTINGS["half_life"],
        neighbor_weight=SETTINGS["neighbor_weight"], same_type_only=SETTINGS["same_type_only"],
    ).fit(bundle["train_q"], bundle["train_h"], bundle["labels"])
    pooled = FrequencyForecaster(
        "pooled_recent", prior_strength=SETTINGS["pooled_prior_strength"],
        half_life=SETTINGS["pooled_half_life"],
    ).fit(bundle["train_q"], bundle["train_h"], bundle["labels"])
    model = BlendNeighborPooled(neighbor, pooled, SETTINGS["blend_weight"])
    val_pred = model.predict(bundle["val_q"], bundle["val_h"])
    contest = score_predictions(val_pred, bundle)

    out = Path("winner-blend")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    write_submission(out / "validation_predictions.csv", val_pred)
    write_submission(out / "submission.csv", model.predict(bundle["test_q"], bundle["test_h"]))
    model.save(out / "baseline.json")
    write_json(out / "run.json", {
        "model": "blend", "settings": SETTINGS, "contest": contest,
        "round": metadata["round"], "season_id": metadata["season_id"],
    })

    entry, zip_path = _build_entry(out, contest, metadata)
    reproduced = BlendNeighborPooled.load(entry / "baseline.json").predict(
        bundle["test_q"], bundle["test_h"])
    original = read_submission(out / "submission.csv", {q.query_id for q in bundle["test_q"]})
    max_diff = max(abs(a - b) for key in original for a, b in zip(original[key], reproduced[key]))
    print(f"Validation points: {contest['technical_points']:.6f} / {metadata['max_points']}")
    print(f"Reproduction max abs diff: {max_diff}")
    print(f"CSV: {out / 'submission.csv'}\nZIP: {zip_path}")
    if max_diff > 1e-8:
        print("ERROR: reproduction tolerance exceeded", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
