"""Blend NeighborPooledRecent with a pooled_recent FrequencyForecaster."""

import json
from pathlib import Path

from .baseline import FrequencyForecaster
from .io import write_json
from .schema import PAYOUTS, ValidationError, finite_number, probabilities


def blend_predictions(a, b, weight_a):
    """Mix two probability maps with weight_a on the first model."""
    out = {}
    for key in a:
        mixed = tuple(weight_a * x + (1.0 - weight_a) * y for x, y in zip(a[key], b[key]))
        out[key] = probabilities(mixed)
    return out


class BlendNeighborPooled:
    """Blend NeighborPooledRecent with a pooled_recent FrequencyForecaster."""

    def __init__(self, neighbor, pooled, blend_weight):
        self.neighbor = neighbor
        self.pooled = pooled
        self.blend_weight = finite_number(blend_weight, "blend_weight")
        if not 0.0 <= self.blend_weight <= 1.0:
            raise ValidationError("blend_weight must be in [0, 1].")

    def predict(self, queries, history):
        if self.blend_weight >= 1.0:
            return self.neighbor.predict(queries, history)
        if self.blend_weight <= 0.0:
            return self.pooled.predict(queries, history)
        return blend_predictions(
            self.neighbor.predict(queries, history),
            self.pooled.predict(queries, history),
            self.blend_weight,
        )

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.neighbor.save(path.parent / "neighbor_model.json")
        self.pooled.save(path.parent / "pooled_model.json")
        write_json(
            path,
            {
                "model_version": 1,
                "model_type": "blend_neighbor_pooled",
                "blend_weight": self.blend_weight,
                "neighbor_path": "neighbor_model.json",
                "pooled_path": "pooled_model.json",
                "payouts": list(PAYOUTS),
            },
        )

    @classmethod
    def load(cls, path):
        from .neighbor import NeighborPooledRecent

        path = Path(path)
        value = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "model_version", "model_type", "blend_weight",
            "neighbor_path", "pooled_path", "payouts",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("model_version") != 1
            or value.get("model_type") != "blend_neighbor_pooled"
        ):
            raise ValidationError("Unsupported or malformed blend model.")
        neighbor = NeighborPooledRecent.load(path.parent / value["neighbor_path"])
        pooled = FrequencyForecaster.load(path.parent / value["pooled_path"])
        return cls(neighbor, pooled, value["blend_weight"])
