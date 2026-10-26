"""Every setting of the scoring, in one object that saves as JSON.

How settings stack: the admin keeps one global Tuning for everyone. Each person gets a small
patch on top with tuning.update({...}). Single values are replaced, dictionaries are merged
key by key (the person wins), and exclusion lists are added together. Store only the patch
per person and rebuild the full Tuning when you need it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

SCALINGS = ("legacy", "original", "linear")
NORMALIZATIONS = ("minmax", "none")
PRECISIONS = ("float64", "float16")


@dataclass(frozen=True)
class Tuning:
    # How a lab result becomes a deviation (0 = middle of the healthy range, 1 = the edge)
    scaling: str = "legacy"        # legacy: the backend today. original: the research notebook. linear: same unit for every test
    cap: float | None = None       # largest deviation allowed, so one extreme result cannot drown the rest
    deadband: float = 0.0          # ignore deviations smaller than this (0.3 acts only on the outer part of the range)
    ranges: dict[str, list[float]] = field(default_factory=dict)  # {"Vitamin D": [30, 60]} a person's own healthy range

    # How much each part counts in the final score
    food_mix: float = 1.0          # direct effect of the food on each biochemical
    nutrient_mix: float = 1.0      # effect through the nutrients the food contains
    nutriscore_mix: float = 1.0    # general food quality, the same for everyone

    # The bias and weight tables
    bias_scale: float = 1.0        # multiplies every bias cell (the direction: 1 raises, minus 1 lowers)
    weight_scale: float = 1.0      # multiplies every weight cell (the strength)
    importance: dict[str, float] = field(default_factory=dict)           # {"Blood Glucose": 2.0, "Estrogen": 0}
    food_cells: dict[str, dict[str, list[float]]] = field(default_factory=dict)      # {biochemical: {food: [bias, weight]}}
    nutrient_cells: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # {biochemical: {nutrient: [bias, weight]}}

    # The food list
    food_adjust: dict[str, float] = field(default_factory=dict)  # added to a food's final score, e.g. {"Salmon": 0.2}
    exclude_foods: list[str] = field(default_factory=list)       # allergies, dislikes
    exclude_groups: list[str] = field(default_factory=list)      # a category or sub category, e.g. "meat"

    # Numbers
    normalize: str = "minmax"      # scale each part to 0..1 for every person (the backend) or keep raw numbers
    precision: str = "float64"     # float16 reproduces the backend's stored scores exactly

    def __post_init__(self):
        for name, allowed in (("scaling", SCALINGS), ("normalize", NORMALIZATIONS), ("precision", PRECISIONS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.cap is not None and self.cap <= 0:
            raise ValueError("cap must be positive")
        if self.deadband < 0:
            raise ValueError("deadband cannot be negative")

    @classmethod
    def legacy(cls) -> "Tuning":
        """Exactly what the Django backend computes today."""
        return cls(precision="float16")

    @classmethod
    def recommended(cls) -> "Tuning":
        """Every test measured in the same unit (half ranges), capped at 4."""
        return cls(scaling="linear", cap=4.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Tuning":
        return cls().update(data or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, patch: dict[str, Any]) -> "Tuning":
        known = {f.name for f in fields(self)}
        if unknown := set(patch) - known:
            raise ValueError(f"unknown settings {sorted(unknown)}")
        changes = {}
        for key, value in patch.items():
            current = getattr(self, key)
            if key in ("food_cells", "nutrient_cells"):
                merged = {bio: dict(cells) for bio, cells in current.items()}
                for bio, cells in value.items():
                    merged.setdefault(bio, {}).update(cells)
                changes[key] = merged
            elif isinstance(current, dict):
                changes[key] = {**current, **value}
            elif isinstance(current, list):
                changes[key] = list(dict.fromkeys([*current, *value]))
            else:
                changes[key] = value
        return replace(self, **changes)
