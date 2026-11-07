"""Verify the candidate predictor reproduces Step 3's validation score (55.6085) and base reproduces 55.6016."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT))
import candidate
from slot_forecasting.contest_data import load_contest, score_predictions
he = candidate.C.he
m = he.Forecaster(json.loads((ROOT / "entry-final/model.json").read_text()))
vq = he.read_queries(ROOT / "public/validation/queries.csv"); vh = he.read_history(ROOT / "public/validation/history.csv")
b = load_contest(ROOT / "public")
for v in ("base", "candidate"):
    p = candidate.predict(m, vq, vh, v)
    print(v, round(score_predictions({k: tuple(map(float, x)) for k, x in p.items()}, b)["technical_points"], 4))
