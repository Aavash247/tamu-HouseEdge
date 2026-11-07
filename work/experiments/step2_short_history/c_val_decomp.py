"""Competition-side decomposition (scoring only, no fitting): loss ratio per condition if the win
probability were perfect vs if the payout-size distribution were perfect, using the submitted model's
validation predictions."""
import csv, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(ROOT))
from slot_forecasting.contest_data import load_contest, score_predictions
C = ["p_0", "p_1", "p_2", "p_5", "p_20"]
rd = lambda p: {r["query_id"]: np.array([float(r[c]) for c in C]) for r in csv.DictReader(open(p))}
pred = rd(ROOT / "submissions/final/validation_predictions.csv"); tgt = rd(ROOT / "public/validation/targets.csv")
b = load_contest(ROOT / "public")
def mk(win_from, size_from):
    out = {}
    for k in pred:
        pw = 1 - win_from[k][0]; s = size_from[k][1:] / size_from[k][1:].sum()
        out[k] = tuple(float(x) for x in np.concatenate([[1 - pw], pw * s]))
    return out
for name, p in (("model", {k: tuple(map(float, v)) for k, v in pred.items()}), ("perfect win", mk(tgt, pred)), ("perfect size", mk(pred, tgt))):
    s = score_predictions(p, b)
    print(f"{name:13s} points={s['technical_points']:.3f} " + " ".join(f"{c}={s['by_condition'][c]['loss_ratio']:.4f}" for c in s["by_condition"]))
