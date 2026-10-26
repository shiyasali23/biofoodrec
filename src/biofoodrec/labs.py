"""Read lab results from any JSON shape Biolabs uses.

Supported shapes:

1. Simple: {"Blood Glucose": 130, "Vitamin D": 14}
2. With objects: {"Blood Glucose": {"value": 130}}  (the research notebooks)
3. A list: [{"id": 1, "value": 130}] or [{"name": "Blood Glucose", "value": 130}]  (the backend POST body)
4. The biometrics request: {"data": {"updatedBiochemicalsData": [...], "unExpiredBiometricsData": [...]}}
5. A full user payload: {"user": {"gender": ...}, "biometrics_entries": [...]}  (the frontend demo user)

Each one can be a Python object, a JSON string, or a path to a .json file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """The demo patient the Biolabs frontend loads: a 46 year old woman with 4 lab visits."""
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


def _from_list(items) -> dict:
    out = {}
    for item in items:
        key = item.get("name") or item.get("id")
        if key is None:
            raise ValueError(f"each lab result needs a name or an id: {item!r}")
        out[key] = _number(key, item)
    return out


def read_labs(source) -> Labs:
    data = _load(source)
    if isinstance(data, Labs):
        return data
    if isinstance(data, list):
        return Labs(_from_list(data))
    if not isinstance(data, dict):
        raise ValueError(f"cannot read lab results from {type(data).__name__}")

    if set(data) == {"data"}:  # the backend request envelope
        data = data["data"]
    sex = data.get("sex") or data.get("gender") or (data.get("user") or {}).get("gender")
    sex = normalize_sex(sex)

    if "biometrics_entries" in data:
        # Oldest visit first, so a newer result replaces an older one.
        values = {}
        for entry in sorted(data["biometrics_entries"], key=lambda e: e.get("created_at", "")):
            values.update(_from_list(entry["biometrics"]))
        return Labs(values, sex)
    if "updatedBiochemicalsData" in data:
        # The frontend sends older, still valid results already scaled, as the backend expects.
        return Labs(_from_list(data["updatedBiochemicalsData"]), sex,
                    scaled=_from_list(data.get("unExpiredBiometricsData") or []))
    for key in ("biometrics", "labs"):
        if isinstance(data.get(key), dict):
            data = data[key]
        elif isinstance(data.get(key), list):
            return Labs(_from_list(data[key]), sex)
    return Labs({k: _number(k, v) for k, v in data.items() if k not in NOT_LABS}, sex)
