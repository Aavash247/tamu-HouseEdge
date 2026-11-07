"""Pooled-recent forecasts that also borrow same-episode neighbor histories."""

from collections import defaultdict
import json
import math
from pathlib import Path

from .baseline import FrequencyForecaster
from .io import write_json
from .schema import PAYOUTS, ValidationError, finite_number, index_history, probabilities, unique_queries

# Re-export blend helpers so older imports keep working.
from .blend import BlendNeighborPooled, blend_predictions  # noqa: F401


class NeighborPooledRecent:
    """Context prior + own-machine decay + reduced-weight same-episode neighbors."""

    def __init__(
        self,
        prior_strength=20.0,
        half_life=24.0,
        neighbor_weight=0.35,
        same_type_only=True,
    ):
        self.prior_strength = finite_number(prior_strength, "prior_strength")
        self.half_life = finite_number(half_life, "half_life")
        self.neighbor_weight = finite_number(neighbor_weight, "neighbor_weight")
        self.same_type_only = bool(same_type_only)
        if self.prior_strength <= 0 or self.half_life <= 0:
            raise ValidationError("prior_strength and half_life must be positive.")
        if self.neighbor_weight < 0:
            raise ValidationError("neighbor_weight must be non-negative.")
        self.base = FrequencyForecaster(
            "pooled_recent",
            prior_strength=self.prior_strength,
            half_life=self.half_life,
        )

    def fit(self, queries, history, labels=None):
        self.base.fit(queries, history, labels)
        return self

    def predict(self, queries, history):
        if self.base.global_prior is None:
            raise ValidationError("Fit or load the model before predicting.")
        queries = tuple(queries)
        query_map = unique_queries(queries)
        indexed = index_history(queries, history)
        by_episode = defaultdict(list)
        for query in query_map.values():
            by_episode[query.episode_id].append(query)

        predictions = {}
        for query in sorted(query_map.values(), key=lambda q: q.query_id):
            context = self.base.context_priors.get(self.base._key(query), self.base.global_prior)
            counts = [self.prior_strength * value for value in context]
            for play in indexed.get((query.episode_id, query.machine_id), []):
                age = query.cutoff - 1 - play.tick
                counts[PAYOUTS.index(play.payout)] += 2 ** (-age / self.half_life)
            for other in by_episode[query.episode_id]:
                if other.machine_id == query.machine_id:
                    continue
                if self.same_type_only and other.machine_type != query.machine_type:
                    continue
                for play in indexed.get((other.episode_id, other.machine_id), []):
                    if play.tick >= query.cutoff:
                        continue
                    age = query.cutoff - 1 - play.tick
                    weight = self.neighbor_weight * (2 ** (-age / self.half_life))
                    counts[PAYOUTS.index(play.payout)] += weight
            total = math.fsum(counts)
            predictions[query.query_id] = tuple(value / total for value in counts)
        return predictions

    def save(self, path):
        if self.base.global_prior is None:
            raise ValidationError("Cannot save an unfitted model.")
        write_json(path, {
            "model_version": 1,
            "model_type": "neighbor_pooled_recent",
            "prior_strength": self.prior_strength,
            "half_life": self.half_life,
            "neighbor_weight": self.neighbor_weight,
            "same_type_only": self.same_type_only,
            "global_prior": self.base.global_prior,
            "context_priors": self.base.context_priors,
        })

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        expected = {
            "model_version", "model_type", "prior_strength", "half_life",
            "neighbor_weight", "same_type_only", "global_prior", "context_priors",
        }
        if (
            not isinstance(value, dict)
            or set(value) != expected
            or value.get("model_version") != 1
            or value.get("model_type") != "neighbor_pooled_recent"
        ):
            raise ValidationError("Unsupported or malformed neighbor model.")
        model = cls(
            value["prior_strength"], value["half_life"],
            value["neighbor_weight"], value["same_type_only"],
        )
        model.base.global_prior = probabilities(value["global_prior"])
        if not isinstance(value["context_priors"], dict):
            raise ValidationError("context_priors must be an object.")
        model.base.context_priors = {
            key: probabilities(probs) for key, probs in value["context_priors"].items()
        }
        return model
