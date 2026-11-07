"""Fit model.json from a round's public data (train histories + validation targets)."""
import argparse
from pathlib import Path
from houseedge import DEFAULT_CONFIG, Forecaster

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="public/ folder of the round")
p.add_argument("--out", default=str(Path(__file__).parent / "model.json"))
a = p.parse_args()
model = Forecaster.fit(a.data, DEFAULT_CONFIG)
model.save(a.out)
print("pairs:", model.cfg["pairs"], "prior:", [round(x, 3) for x in model.cfg["pair_prior"]])
print("size profiles:", {k: len(v) for k, v in model.cfg["size_profiles"].items()})
