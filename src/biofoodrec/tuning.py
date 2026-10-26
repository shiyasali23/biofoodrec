"""Every knob of the scoring, as one JSON-serializable object.

Layering: a global Tuning (what an admin sets for everyone) is refined per person with
`tuning.update({...partial dict...})`. Scalars are replaced, dict knobs are merged key by key
(the person wins), exclusion lists are unioned. Store the partial dict per user; rebuild on read.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

SCALINGS = ("legacy", "linear")
NORMALIZATIONS = ("minmax", "none")
PRECISIONS = ("float64", "float16")


@dataclass(frozen=True)
class Tuning:
    # --- turning a lab value into a signed deviation ---
    scaling: str = "legacy"        # "legacy": backend formula (raw units outside the range); "linear": half-ranges everywhere
    cap: float | None = None       # clip |deviation| to this; stops one extreme marker drowning the rest
    deadband: float = 0.0          # ignore deviations smaller than this (0.3 = only act on the outer 70% of the range)
    ranges: dict[str, list[float]] = field(default_factory=dict)  # {"Vitamin D": [30, 60]} person-specific range

    # --- how much each pathway counts in the final score ---
    food_mix: float = 1.0          # direct food -> biochemical tables
    nutrient_mix: float = 1.0      # food -> nutrient -> biochemical tables
    nutriscore_mix: float = 1.0    # static food quality prior, same for everyone

    # --- the bias and weight tables themselves ---
    bias_scale: float = 1.0        # multiplies every bias (direction) cell
    weight_scale: float = 1.0      # multiplies every weight (strength) cell
    importance: dict[str, float] = field(default_factory=dict)           # {"Blood Glucose": 2.0, "Estrogen": 0}
    food_cells: dict[str, dict[str, list[float]]] = field(default_factory=dict)      # {bio: {food: [bias, weight]}}
    nutrient_cells: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # {bio: {nutrient: [bias, weight]}}

    # --- the food list ---
    food_adjust: dict[str, float] = field(default_factory=dict)  # added to the final score, e.g. {"Salmon": 0.2}
    exclude_foods: list[str] = field(default_factory=list)       # allergies, dislikes
    exclude_groups: list[str] = field(default_factory=list)      # category or sub_category, e.g. "meat"

    # --- numerics ---
    normalize: str = "minmax"      # rescale each pathway to [0, 1] per request (backend) or keep raw
    precision: str = "float64"     # "float16" reproduces the backend's stored scores bit for bit

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
        """Unit-free deviations, capped, so a marker counts by how far off it is, not by its unit."""
        return cls(scaling="linear", cap=4.0)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Tuning":
        return cls().update(data or {})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(self, patch: dict[str, Any]) -> "Tuning":
        known = {f.name for f in fields(self)}
        if unknown := set(patch) - known:
            raise ValueError(f"unknown tuning keys {sorted(unknown)}")
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
