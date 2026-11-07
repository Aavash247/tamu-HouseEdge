"""Step-3 CANDIDATE predictor (not applied to any submission).
   base       : entry-final architecture, unchanged
   candidate  : 2.1b (type-level rate prior raised to power 2.0, short-history machines only: n < 100,
                i.e. n == 64 at test time) + eps = 0.005 pair-prior mass shared by the 25 non-learned pairs
                (all machines).
Works with ANY fitted config (entry-final model.json, a CV fold config, or a refit), so Step 4 retrain
tests can run both variants from the same fitted model."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "step3_unseen_insurance"))
import components as C  # noqa: E402

VARIANTS = {"base": {"power": 1.0, "eps": 0.0}, "candidate": {"power": 2.0, "eps": 0.005}}


def predict(model, queries, hist, variant="candidate"):
    v = VARIANTS[variant]
    if variant == "base":
        return model.predict(queries, hist)
    return {q["query_id"]: C.combine(C.components(model, q, hist.get((q["episode_id"], q["machine_id"]), {}), v["power"]),
                                     model.cfg, v["eps"]) for q in queries}
