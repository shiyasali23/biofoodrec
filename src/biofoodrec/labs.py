"""Read lab results from any JSON shape Biolabs uses.

Supported shapes for raw lab results (FoodRecommender.recommend):

1. Simple: {"Blood Glucose": 130, "Vitamin D": 14}
2. With objects: {"Blood Glucose": {"value": 130}}  (the research notebooks)
3. A list: [{"id": 1, "value": 130}] or [{"name": "Blood Glucose", "value": 130}]
4. The biometrics request: {"data": {"updatedBiochemicalsData": [...], "unExpiredBiometricsData": [...]}}
5. A full user payload: {"user": {"gender": ...}, "biometrics_entries": [...]}  (the frontend demo user)

Values that are already scaled (FoodRecommender.score) come as {name or id: value}, a list like 3,
or the foods_score request {"data": [{"id": 1, "value": 0.5}]}.

Each one can be a Python object, a JSON string, or a path to a .json file.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

DATA = Path(__file__).parent / "data"
SEXES = {"male": "male", "m": "male", "man": "male", "female": "female", "f": "female", "woman": "female"}
NOT_LABS = {"name", "first_name", "last_name", "age", "gender", "sex", "occupation", "email", "id", "bmi", "user"}


@dataclass
class Labs:
    values: dict[Any, float]                              # raw results, keyed by biochemical name or backend id
    sex: str | None = None
    scaled: dict[Any, float] = field(default_factory=dict)  # results that arrive already scaled (older, still valid ones)


def normalize_sex(sex: str | None) -> str | None:
    if sex is None:
        return None
    key = str(sex).strip().lower()
    if key not in SEXES:
        raise ValueError(f"sex must be male or female, got {sex!r}")
    return SEXES[key]


def sample_patient() -> dict:
    """The demo patient the Biolabs website loads: a 46 year old woman with 4 lab visits."""
    return json.loads((DATA / "sample_patient.json").read_text())


def _load(source):
    if isinstance(source, Path):
        return json.loads(source.read_text())
    if isinstance(source, str):
        text = source.strip()
        return json.loads(text) if text[:1] in "{[" else json.loads(Path(source).read_text())
    return source


def _number(key, value) -> float:
    if isinstance(value, dict):
        value = value.get("value")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"lab result for {key!r} is not a number: {value!r}") from None


def _collect(pairs, resolve: Callable | None) -> dict:
    """{key: value} from (key, value) pairs. With resolve, keys become names, so an id and a name
    for the same test are recognised as the same test. A test given twice in one visit is an error."""
    out = {}
    for key, value in pairs:
        name = resolve(key) if resolve else key
        if name in out:
            raise ValueError(f"{name} is given twice in the same set of results")
        out[name] = _number(key, value)
    return out


def _pairs(items):
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"each lab result must be an object with a name or id and a value: {item!r}")
        key = item.get("name") or item.get("id")
        if key is None:
            raise ValueError(f"each lab result needs a name or an id: {item!r}")
        yield key, item


def _when(entry) -> datetime:
    """A visit's date, so visits sort by real time even when time zones differ. No date sorts first."""
    text = entry.get("created_at")
    if not text:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        moment = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        raise ValueError(f"cannot read the visit date {text!r}") from None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def read_labs(source, resolve: Callable | None = None) -> Labs:
    """Raw lab results. resolve (name or id to name) lets visits that mix ids and names merge correctly."""
    data = _load(source)
    if isinstance(data, Labs):
        return data
    if isinstance(data, list):
        return Labs(_collect(_pairs(data), resolve))
    if not isinstance(data, dict):
        raise ValueError(f"cannot read lab results from {type(data).__name__}")

    if set(data) == {"data"}:  # the backend request envelope
        data = data["data"]
        if isinstance(data, list):
            raise ValueError("this is the foods_score request: its values are already scaled. "
                             "Pass it to FoodRecommender.score() instead of recommend()")
    sex = data.get("sex") or data.get("gender") or (data.get("user") or {}).get("gender")
    sex = normalize_sex(sex)

    if "biometrics_entries" in data:
        # Oldest visit first, so a newer result replaces an older one.
        values = {}
        for entry in sorted(data["biometrics_entries"], key=_when):
            values.update(_collect(_pairs(entry["biometrics"]), resolve))
        return Labs(values, sex)
    if "updatedBiochemicalsData" in data:
        # The frontend sends older, still valid results already scaled, as the backend expects.
        return Labs(_collect(_pairs(data["updatedBiochemicalsData"]), resolve), sex,
                    scaled=_collect(_pairs(data.get("unExpiredBiometricsData") or []), resolve))
    for key in ("biometrics", "labs"):
        if isinstance(data.get(key), dict):
            data = data[key]
        elif isinstance(data.get(key), list):
            return Labs(_collect(_pairs(data[key]), resolve), sex)
    return Labs(_collect(((k, v) for k, v in data.items() if k not in NOT_LABS), resolve), sex)


def read_scaled(source, resolve: Callable | None = None) -> dict:
    """Values that are already scaled: a mapping, a list of {id or name, value}, or {"data": [...]}."""
    data = _load(source)
    if isinstance(data, Mapping) and set(data) == {"data"}:
        data = data["data"]
    if isinstance(data, list):
        return _collect(_pairs(data), resolve)
    if isinstance(data, Mapping):
        return _collect(data.items(), resolve)
    raise ValueError(f"cannot read scaled values from {type(data).__name__}")
