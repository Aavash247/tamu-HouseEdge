"""python predict.py --data /data --out /output/submission.csv"""
import argparse
from pathlib import Path

from data import read_history, read_queries, write_submission
from forecaster import Forecaster

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--split", default="test")
ap.add_argument("--model", default=str(Path(__file__).resolve().parent / "model.json"))
a = ap.parse_args()

split = Path(a.data) / a.split
preds = Forecaster.load(a.model).predict(read_queries(split / "queries.csv"), read_history(split / "history.csv"))
write_submission(a.out, preds)
