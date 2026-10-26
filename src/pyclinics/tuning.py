"""Every setting of the scoring, in one object that saves as JSON.

How settings stack: the admin keeps one global Tuning for everyone. Each person gets a small
patch on top with tuning.update({...}). Single values are replaced, dictionaries are merged
key by key (the person wins), and lists are added together. Store only the patch per person
and rebuild the full Tuning when you need it.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

SCALINGS = ("legacy", "original", "linear")
NORMALIZATIONS = ("minmax", "none")
PRECISIONS = ("float64", "float16")
LIST_FIELDS = ("limit_only", "exclude_foods", "exclude_groups")

# Nutrients that can be advised down but never up: free sugars, alcohol, the fats and salt most
# guidelines say to limit, and the anti nutrients. Change the list with Tuning(limit_only=[...]).
LIMIT_ONLY = (
    "Alcohol", "Trans Fatty Acids", "Saturated Fatty Acids", "Cholesterol", "Sodium",
    "Sucrose", "Glucose", "Fructose", "Caffeine", "Oxalates", "Phytic Acid", "Lectins", "Tannins",
)

# Tests where only one side of the range is a problem. Lower LDL, cholesterol, triglycerides,
# apolipoprotein B and CRP are better; higher HDL and apolipoprotein A1 are better. Without this,
# a low LDL result would ask for foods that raise LDL.
ONE_SIDED = {
    "LDL Cholesterol": "high", "Total Cholesterol": "high", "Triglycerides": "high",
    "Apolipoprotein B": "high", "C-Reactive Protein": "high",
    "HDL Cholesterol": "low", "Apolipoprotein A1": "low",
}


@dataclass(frozen=True)
class Tuning:
    # Which lab results change the advice
    act_in_range: bool = False     # False: only results outside the healthy range act. True: every result pushes to the middle (the backend)
    only_when: dict[str, str] = field(default_factory=lambda: dict(ONE_SIDED))  # {"LDL Cholesterol": "high"} act only on that side

    # How a lab result becomes a deviation (0 = middle of the healthy range, 1 = the edge)
    scaling: str = "linear"        # linear: same unit for every test. original: the research notebook. legacy: the backend today
    cap: float | None = 4.0        # largest deviation allowed, so one extreme result cannot drown the rest
    deadband: float = 0.0          # with act_in_range, ignore results inside the range closer to the middle than this
    ranges: dict[str, list[float]] = field(default_factory=dict)  # {"Vitamin D": [30, 60]} a person's own healthy range

    # How much each part counts in the final score
    food_mix: float = 1.0          # direct effect of the food on each biochemical
    nutrient_mix: float = 1.0      # effect through the nutrients the food contains
    nutriscore_mix: float = 1.0    # general food quality, the same for everyone

    # The bias and weight tables
    bias_scale: float = 1.0        # multiplies every bias cell (the direction from the research lists: 1, 0 or minus 1)
    weight_scale: float = 1.0      # multiplies every weight cell (a second, finer estimate between minus 1 and 1)
    importance: dict[str, float] = field(default_factory=dict)           # {"Blood Glucose": 2.0, "Estrogen": 0}
    food_cells: dict[str, dict[str, list[float]]] = field(default_factory=dict)      # {biochemical: {food: [bias, weight]}}
    nutrient_cells: dict[str, dict[str, list[float]]] = field(default_factory=dict)  # {biochemical: {nutrient: [bias, weight]}}

    # Safety guards
    limit_only: list[str] = field(default_factory=lambda: list(LIMIT_ONLY))  # nutrients never advised up
    food_guard: float | None = 0.3  # foods with nutriscore at or below this are never pushed up by lab results

    # The food list
    food_adjust: dict[str, float] = field(default_factory=dict)  # added to a food's final score, e.g. {"Salmon": 0.2}
    exclude_foods: list[str] = field(default_factory=list)       # allergies, dislikes
    exclude_groups: list[str] = field(default_factory=list)      # a category or sub category, e.g. "seafood"

    # Numbers
    normalize: str = "minmax"      # scale each part to 0..1 for every person (the backend) or keep raw numbers
    precision: str = "float64"     # float16 reproduces the backend's stored scores exactly

    def __post_init__(self):
        set_ = lambda name, value: object.__setattr__(self, name, value)
        # Copy everything into plain lists and dicts, so a Tuning never shares data with another one
        # and saves to JSON and back unchanged.
        for name in LIST_FIELDS:
            value = getattr(self, name)
            if isinstance(value, str):
                raise ValueError(f"{name} must be a list of names, not one text: {value!r}")
            set_(name, [str(v) for v in value])
        set_("importance", {k: float(v) for k, v in self.importance.items()})
        set_("food_adjust", {k: float(v) for k, v in self.food_adjust.items()})
        set_("only_when", {k: str(v) for k, v in self.only_when.items()})
        set_("ranges", {k: [float(x) for x in pair] for k, pair in self.ranges.items()})
        for name in ("food_cells", "nutrient_cells"):
            set_(name, {bio: {item: [float(x) for x in pair] for item, pair in row.items()}
                        for bio, row in getattr(self, name).items()})

        for name, allowed in (("scaling", SCALINGS), ("normalize", NORMALIZATIONS), ("precision", PRECISIONS)):
            if getattr(self, name) not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        if self.cap is not None and self.cap <= 0:
            raise ValueError("cap must be positive")
        if self.deadband < 0:
            raise ValueError("deadband cannot be negative")
        # A negative number here would flip advice (recommend what raises a high result), so it is refused.
        for name in ("food_mix", "nutrient_mix", "nutriscore_mix", "bias_scale", "weight_scale"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.food_mix + self.nutrient_mix + self.nutriscore_mix == 0:
            raise ValueError("at least one of food_mix, nutrient_mix, nutriscore_mix must be above 0")
        if negative := [name for name, value in self.importance.items() if value < 0]:
            raise ValueError(f"importance cannot be negative: {negative}")
        if bad := [name for name, pair in self.ranges.items() if len(pair) != 2 or not pair[0] < pair[1]]:
            raise ValueError(f"each range needs [low, high] with low below high: {bad}")
        if bad := [name for name, side in self.only_when.items() if side not in ("high", "low")]:
            raise ValueError(f"only_when takes 'high' or 'low': {bad}")
        if bad := [f"{bio} x {item}" for name in ("food_cells", "nutrient_cells")
                   for bio, row in getattr(self, name).items() for item, pair in row.items() if len(pair) != 2]:
            raise ValueError(f"each cell needs [bias, weight]: {bad}")

    def __hash__(self):
        return hash(json.dumps(asdict(self), sort_keys=True))

    @classmethod
    def legacy(cls) -> "Tuning":
        """Exactly what the Django backend computes today, rounding included. No safety guards."""
        return cls(act_in_range=True, only_when={}, scaling="legacy", cap=None,
                   limit_only=[], food_guard=None, precision="float16")

    @classmethod
    def recommended(cls) -> "Tuning":
        """The defaults: only results outside the range act, measured in half ranges, all guards on."""
        return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Tuning":
        """Full settings, as saved by to_dict(). Missing keys take the defaults. Use update() for a patch."""
        data = dict(data or {})
        if unknown := set(data) - {f.name for f in fields(cls)}:
            raise ValueError(f"unknown settings {sorted(unknown)}")
        return cls(**data)

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
            elif key in LIST_FIELDS:
                if isinstance(value, str):
                    raise ValueError(f"{key} must be a list of names, not one text: {value!r}")
                changes[key] = list(dict.fromkeys([*current, *value]))
            else:
                changes[key] = value
        return replace(self, **changes)
