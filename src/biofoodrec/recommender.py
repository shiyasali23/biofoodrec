"""Lab values -> signed deviations -> per-food scores, with an exact per-biochemical explanation.

The score is linear in the deviations until the final min-max step:

    drive[b]        = -deviation[b] * importance[b]          # the -1: push each marker back toward its optimum
    food_effect[f]  = sum_b drive[b] * (bias_scale*food_bias[b,f] + weight_scale*food_weight[b,f])
    nutrient_drive  = sum_b drive[b] * (bias_scale*nutrient_bias[b,n] + weight_scale*nutrient_weight[b,n])
    nutrient_effect = normalized_nutrients[f,:] @ nutrient_drive
    score[f]        = food_mix*norm(food_effect) + nutrient_mix*norm(nutrient_effect)
                      + nutriscore_mix*normalized_nutriscore[f] + food_adjust[f]
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .reference import ReferenceData
from .tuning import Tuning


def _norm(values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return values
    return (values - values.min()) / (np.ptp(values) + 1e-9)  # same guard as the backend


class Recommendation:
    def __init__(self, table: pd.DataFrame, explain_parts: dict):
        self.table = table
        self._parts = explain_parts

    def top(self, n: int = 10) -> pd.DataFrame:
        return self.table.head(n)

    def to_dict(self) -> dict[str, float]:
        """{food: score} in reference order, the shape FoodsScore.foods_score stores today."""
        return self.table.sort_values("order")["score"].astype(float).to_dict()

    def explain(self, food: str) -> pd.DataFrame:
        """Each biochemical's pull on this food before normalization. Positive = pushes the food up."""
        parts = self._parts
        f = parts["foods"].get_loc(food)
        food_pull = parts["drive"] * parts["food_coef"][:, f]
        nutrient_pull = parts["drive"] * (parts["nutrient_coef"] @ parts["nutrients"][f])
        frame = pd.DataFrame({
            "deviation": parts["deviation"],
            "importance": parts["importance"],
            "food_pathway": food_pull,
            "nutrient_pathway": nutrient_pull,
        }, index=parts["biochemicals"])
        frame["total"] = frame.food_pathway + frame.nutrient_pathway
        frame = frame[frame.deviation != 0]
        return frame.reindex(frame.total.abs().sort_values(ascending=False).index)


class FoodRecommender:
    def __init__(self, reference: ReferenceData, tuning: Tuning | None = None):
        self.reference = reference
        self.tuning = tuning or Tuning()

    # ---------- step 1: lab values -> deviations ----------

    def deviations(self, values: Mapping, sex: str, tuning: Tuning | None = None) -> pd.DataFrame:
        """value, healthy range, signed deviation (0 = optimum, +-1 = range edge), health weight, status."""
        tuning = tuning or self.tuning
        ref = self.reference
        values = {ref.biochemical_name(k): float(v) for k, v in values.items()}
        ranges = ref.ranges(sex).loc[list(values)].copy()
        for name, (low, high) in tuning.ranges.items():
            if name in ranges.index:
                ranges.loc[name] = [low, high]
        if (ranges.high <= ranges.low).any():
            raise ValueError(f"empty range for {list(ranges.index[ranges.high <= ranges.low])}")

        low, high = ranges.low.to_numpy(float), ranges.high.to_numpy(float)
        if tuning.precision == "float16":  # the backend keeps ranges as float16
            low, high = low.astype(np.float16), high.astype(np.float16)
        value = np.array(list(values.values()), dtype=float)
        optimum = (low + high) / 2.0
        linear = 2 * (value - optimum) / (high - low)
        inside = (value >= low) & (value <= high)

        if tuning.scaling == "legacy":
            deviation = np.where(inside, np.round(linear, 2), np.round(np.where(value < low, value - low, value - high), 2))
            health = np.where(inside, np.round(1 - np.abs(linear), 2),
                              np.where(value < low, np.round(value - low, 2), np.round(high - value, 2)))
        else:
            deviation = linear
            health = 1 - np.abs(linear)

        deviation = np.where(np.abs(deviation) < tuning.deadband, 0.0, deviation)
        if tuning.cap is not None:
            deviation = np.clip(deviation, -tuning.cap, tuning.cap)

        return pd.DataFrame({
            "value": value, "low": np.asarray(low, float), "high": np.asarray(high, float),
            "deviation": deviation, "health_weight": health,
            "status": np.where(value > high, "high", np.where(value < low, "low", "normal")),
        }, index=pd.Index(list(values), name="biochemical"))

    # ---------- step 2: deviations -> food scores ----------

    def recommend(self, values: Mapping, sex: str, tuning: Tuning | None = None) -> Recommendation:
        """Raw lab values in, ranked foods out. This is what POST /backend/biometrics does."""
        tuning = tuning or self.tuning
        return self.score(self.deviations(values, sex, tuning)["deviation"].to_dict(), tuning)

    def score(self, deviations: Mapping, tuning: Tuning | None = None) -> Recommendation:
        """Pre-scaled deviations in ({name or id: value}). This is what POST /backend/foods_score/create does."""
        tuning = tuning or self.tuning
        ref = self.reference
        bios, foods, nutrients = ref.biochemicals.index, ref.foods.index, ref.nutrients.index
        cast = (lambda a: a.astype(np.float16).astype(float)) if tuning.precision == "float16" else (lambda a: a)

        deviation = np.zeros(len(bios))
        for key, value in deviations.items():
            deviation[bios.get_loc(ref.biochemical_name(key))] = value
        deviation = cast(deviation)
        importance = np.array([tuning.importance.get(name, 1.0) for name in bios])
        drive = -deviation * importance

        food_bias, food_weight = self._tables(ref.food_bias, ref.food_weight, tuning.food_cells, foods)
        nutrient_bias, nutrient_weight = self._tables(ref.nutrient_bias, ref.nutrient_weight, tuning.nutrient_cells, nutrients)
        food_coef = tuning.bias_scale * cast(food_bias) + tuning.weight_scale * cast(food_weight)
        nutrient_coef = tuning.bias_scale * cast(nutrient_bias) + tuning.weight_scale * cast(nutrient_weight)
        normalized_nutrients = cast(ref.normalized_nutrients)

        food_effect = drive @ food_coef
        nutrient_effect = normalized_nutrients @ (drive @ nutrient_coef)
        nutriscore = cast(ref.normalized_nutriscore)
        adjust = np.array([tuning.food_adjust.get(name, 0.0) for name in foods])

        table = pd.DataFrame({
            "score": tuning.nutrient_mix * _norm(nutrient_effect, tuning.normalize)
                     + tuning.food_mix * _norm(food_effect, tuning.normalize)
                     + tuning.nutriscore_mix * nutriscore + adjust,
            "food_effect": food_effect,
            "nutrient_effect": nutrient_effect,
            "nutriscore": nutriscore,
            "category": ref.foods["category"].to_numpy(),
            "sub_category": ref.foods["sub_category"].to_numpy(),
            "order": np.arange(len(foods)),
        }, index=pd.Index(foods, name="food"))

        excluded = set(tuning.exclude_foods)
        groups = set(tuning.exclude_groups)
        keep = ~(table.index.isin(excluded) | table.category.isin(groups) | table.sub_category.isin(groups))
        table = table[keep].sort_values("score", ascending=False)
        table.insert(0, "rank", np.arange(1, len(table) + 1))

        return Recommendation(table, {
            "biochemicals": bios, "foods": foods, "deviation": deviation, "importance": importance, "drive": drive,
            "food_coef": food_coef, "nutrient_coef": nutrient_coef, "nutrients": normalized_nutrients,
        })

    @staticmethod
    def _tables(bias: pd.DataFrame, weight: pd.DataFrame, cells: dict, columns) -> tuple[np.ndarray, np.ndarray]:
        """The reference tables with any per-cell overrides applied."""
        if not cells:
            return bias.to_numpy(), weight.to_numpy()
        bias, weight = bias.copy(), weight.copy()
        for bio, row in cells.items():
            for item, (b, w) in row.items():
                if bio not in bias.index or item not in columns:
                    raise KeyError(f"unknown cell {bio!r} x {item!r}")
                bias.loc[bio, item], weight.loc[bio, item] = b, w
        return bias.to_numpy(), weight.to_numpy()
