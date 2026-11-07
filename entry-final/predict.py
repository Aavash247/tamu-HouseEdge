"""Predict next-spin payout probabilities: python predict.py --data /data --out /output/submission.csv"""
import argparse
from pathlib import Path
from houseedge import Forecaster, read_history, read_queries, write_submission

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--out", required=True)
p.add_argument("--split", default="test")
p.add_argument("--model", default=str(Path(__file__).resolve().parent / "model.json"))
a = p.parse_args()
model = Forecaster.load(a.model)
split = Path(a.data) / a.split
write_submission(a.out, model.predict(read_queries(split / "queries.csv"), read_history(split / "history.csv")))
