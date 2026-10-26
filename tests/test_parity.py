"""Checks the package against what the Django backend actually produced.

The data comes from the Biolabs repo: the CSVs from backend/datasets and the expected output
from frontend/src/jsons/dummy-user-data.json (a real backend response for the demo user:
4 biometrics entries with value + scaled_value, and the food scores of the last one).
BIOLABS_ROOT points at that repo; by default it is the Biolabs folder next to this one.
Without it these tests are skipped.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from biofoodrec import FoodRecommender, ReferenceData, Tuning

ROOT = Path(os.environ.get("BIOLABS_ROOT", Path(__file__).resolve().parents[2] / "Biolabs"))
FIXTURE_PATH = ROOT / "frontend/src/jsons/dummy-user-data.json"
if not FIXTURE_PATH.exists():
    pytest.skip(f"Biolabs data not found at {ROOT}; set BIOLABS_ROOT", allow_module_level=True)

FIXTURE = json.loads(FIXTURE_PATH.read_text())
LAST_ENTRY = max(FIXTURE["biometrics_entries"], key=lambda e: e["created_at"])["biometrics"]
STORED_SCORES = json.loads(FIXTURE["foods_score"]["foods_score"])


@pytest.fixture(scope="module")
def recommender():
    return FoodRecommender(ReferenceData.from_csv_dir(ROOT / "backend/datasets"), Tuning.legacy())


def test_reference_shapes(recommender):
    ref = recommender.reference
    assert ref.food_bias.shape == (58, 150)
    assert ref.nutrient_weight.shape == (58, 75)
    assert ref.food_nutrients.shape == (150, 75)


def test_normalized_nutrients_match_seed_check(recommender):
    # The same file validate_normalization asserts the DB against.
    import pandas as pd
    expected = pd.read_csv(ROOT / "backend/datasets/normalized_nutrients.csv").set_index("name")
    got = recommender.reference.normalized_nutrients
    cols = list(recommender.reference.nutrients.index)
    assert np.allclose(got, expected.loc[recommender.reference.foods.index, cols].to_numpy(), atol=1e-4)


def test_deviations_match_backend(recommender):
    for entry in FIXTURE["biometrics_entries"]:
        values = {b["id"]: b["value"] for b in entry["biometrics"]}
        got = recommender.deviations(values, FIXTURE["user"]["gender"])["deviation"].to_numpy()
        expected = np.array([b["scaled_value"] for b in entry["biometrics"]])
        assert np.allclose(got, expected, atol=1e-9), np.abs(got - expected).max()


def test_food_scores_match_backend(recommender):
    result = recommender.score({b["id"]: b["scaled_value"] for b in LAST_ENTRY}).to_dict()
    assert list(result) == list(STORED_SCORES)  # same foods, same order
    diff = max(abs(result[k] - STORED_SCORES[k]) for k in STORED_SCORES)
    assert diff < 1e-9, diff


def test_recommend_end_to_end_matches_backend(recommender):
    values = {b["id"]: b["value"] for b in LAST_ENTRY}
    result = recommender.recommend(values, FIXTURE["user"]["gender"]).to_dict()
    assert max(abs(result[k] - STORED_SCORES[k]) for k in STORED_SCORES) < 1e-9


def test_explain_adds_up(recommender):
    rec = recommender.score({b["id"]: b["scaled_value"] for b in LAST_ENTRY})
    food = rec.top(1).index[0]
    parts = rec.explain(food)
    row = rec.table.loc[food]
    assert np.isclose(parts.food_pathway.sum(), row.food_effect)
    assert np.isclose(parts.nutrient_pathway.sum(), row.nutrient_effect)


def test_knobs_move_the_ranking(recommender):
    values = {b["id"]: b["value"] for b in LAST_ENTRY}
    sex = FIXTURE["user"]["gender"]
    base = recommender.recommend(values, sex)
    first = base.top(1).index[0]
    assert first not in recommender.recommend(values, sex, Tuning(exclude_foods=[first])).table.index
    boosted = recommender.recommend(values, sex, Tuning(food_adjust={"Tacos": 5.0}))
    assert boosted.top(1).index[0] == "Tacos"
    only_prior = recommender.recommend(values, sex, Tuning(food_mix=0, nutrient_mix=0))
    assert np.allclose(only_prior.table.score, only_prior.table.nutriscore)
