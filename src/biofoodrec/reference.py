"""Reference tables: biochemicals, foods, nutrients and the bias/weight tables that link them.

Every table is aligned by *name*, never by row position, so a CSV with its columns in a
different order (the nutrient bias and weight files already differ) still lines up.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Default file names, as they sit in Biolabs/backend/datasets.
DEFAULT_FILES = {
    "biochemicals": "biochemical_units.csv",
    "foods": "food_nutrients.csv",
    "nutrients": "nutrients_categories.csv",
    "food_bias": "biochemical_foods_bias_df.csv",
    "food_weight": "biochemical_foods_weights.csv",
    "nutrient_bias": "biochemical_nutrients_bias.csv",
    "nutrient_weight": "biochemical_nutrients_weights.csv",
}

BIOCHEMICAL_COLUMNS = ["category", "unit", "validity", "female_min", "female_max", "male_min", "male_max"]
FOOD_COLUMNS = ["category", "sub_category", "nutriscore"]


def _minmax(values: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]; a constant column becomes all zeros (as normalize_nutriscore_nutrients does)."""
    low, high = np.nanmin(values, axis=0), np.nanmax(values, axis=0)
    span = np.where(high > low, high - low, 1.0)
    out = np.where(high > low, (values - low) / span, 0.0)
    return np.nan_to_num(out, nan=0.0)


def _missing(expected, present) -> list[str]:
    present = set(present)
    return [name for name in expected if name not in present]


@dataclass(frozen=True)
class ReferenceData:
    biochemicals: pd.DataFrame     # index: biochemical name. Row order defines backend ids (1-based).
    foods: pd.DataFrame            # index: food name. category, sub_category, nutriscore
    nutrients: pd.DataFrame        # index: nutrient name. category, unit
    food_nutrients: pd.DataFrame   # foods x nutrients, raw amounts
    food_bias: pd.DataFrame        # biochemicals x foods, direction (-1, 0, 1)
    food_weight: pd.DataFrame      # biochemicals x foods, strength
    nutrient_bias: pd.DataFrame    # biochemicals x nutrients
    nutrient_weight: pd.DataFrame  # biochemicals x nutrients

    # ---------- loading ----------

    @classmethod
    def from_csv_dir(cls, directory: str | Path, **paths) -> "ReferenceData":
        """Load the seven CSVs from one folder. Pass e.g. food_weight="other.csv" to swap one file."""
        directory = Path(directory)
        files = {key: directory / name for key, name in DEFAULT_FILES.items()}
        files.update({key: Path(value) for key, value in paths.items()})
        return cls.from_csv(**files)

    @classmethod
    def from_csv(cls, *, biochemicals, foods, nutrients, food_bias, food_weight, nutrient_bias, nutrient_weight):
        read = lambda source: pd.read_csv(source).set_index("name")
        return cls.from_frames(
            biochemicals=read(biochemicals), foods=read(foods), nutrients=read(nutrients),
            food_bias=read(food_bias), food_weight=read(food_weight),
            nutrient_bias=read(nutrient_bias), nutrient_weight=read(nutrient_weight),
        )

    @classmethod
    def from_frames(cls, *, biochemicals, foods, nutrients, food_bias, food_weight, nutrient_bias, nutrient_weight):
        """Build from DataFrames indexed by name, e.g. ones read from the Django ORM instead of CSV."""
        problems = []
        if missing := _missing(BIOCHEMICAL_COLUMNS, biochemicals.columns):
            problems.append(f"biochemicals table lacks columns {missing}")
        if missing := _missing(FOOD_COLUMNS, foods.columns):
            problems.append(f"foods table lacks columns {missing}")
        if missing := _missing(nutrients.index, foods.columns):
            problems.append(f"foods table lacks nutrient columns {missing}")

        def align(frame, label, columns):
            for axis, expected in (("rows", biochemicals.index), ("columns", columns)):
                found = frame.index if axis == "rows" else frame.columns
                if missing := _missing(expected, found):
                    problems.append(f"{label} is missing {axis} {missing[:5]}{' ...' if len(missing) > 5 else ''}")
            # A blank cell means "no link", which the backend also stores as 0.
            return frame.reindex(index=biochemicals.index, columns=columns).astype(float).fillna(0.0)

        tables = {
            "food_bias": align(food_bias, "food_bias", foods.index),
            "food_weight": align(food_weight, "food_weight", foods.index),
            "nutrient_bias": align(nutrient_bias, "nutrient_bias", nutrients.index),
            "nutrient_weight": align(nutrient_weight, "nutrient_weight", nutrients.index),
        }
        if problems:
            raise ValueError("Reference data does not line up:\n  " + "\n  ".join(problems))

        return cls(
            biochemicals=biochemicals[BIOCHEMICAL_COLUMNS].copy(),
            foods=foods[FOOD_COLUMNS].copy(),
            nutrients=nutrients.copy(),
            food_nutrients=foods[list(nutrients.index)].astype(float).fillna(0.0),
            **tables,
        )

    # ---------- lookups ----------

    def biochemical_name(self, key: str | int) -> str:
        """Accept a name or the backend's 1-based biochemical id."""
        if isinstance(key, (int, np.integer)) and not isinstance(key, bool):
            if not 1 <= key <= len(self.biochemicals):
                raise KeyError(f"biochemical id {key} is out of range 1..{len(self.biochemicals)}")
            return self.biochemicals.index[key - 1]
        if key not in self.biochemicals.index:
            raise KeyError(f"unknown biochemical {key!r}")
        return key

    def ranges(self, sex: str) -> pd.DataFrame:
        """Healthy low/high per biochemical. Like the backend, anything but "male" uses the female range."""
        prefix = "male" if sex == "male" else "female"
        return self.biochemicals[[f"{prefix}_min", f"{prefix}_max"]].set_axis(["low", "high"], axis=1)

    # ---------- derived matrices ----------

    @property
    def normalized_nutrients(self) -> np.ndarray:
        """(foods, nutrients), each nutrient min-max scaled across foods."""
        return _minmax(self.food_nutrients.to_numpy())

    @property
    def normalized_nutriscore(self) -> np.ndarray:
        return _minmax(self.foods["nutriscore"].to_numpy(dtype=float))
