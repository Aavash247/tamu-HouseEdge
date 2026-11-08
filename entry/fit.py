"""python fit.py --data /path/to/public  (rewrites model.json)"""
import argparse
from pathlib import Path

from forecaster import DEFAULT_CONFIG, Forecaster

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True)
ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "model.json"))
ap.add_argument("--rho", type=float, default=DEFAULT_CONFIG["rho"])
a = ap.parse_args()

model = Forecaster.fit(a.data, dict(DEFAULT_CONFIG, rho=a.rho))
model.save(a.out)
print("pairs:", model.cfg["pairs"], "prior:", [round(x, 3) for x in model.cfg["pair_prior"]])
print("size profiles:", {k: len(v) for k, v in model.cfg["size_profiles"].items()})
