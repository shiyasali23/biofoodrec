"""The package must give the same numbers as the Django backend.

tests/data/backend_demo_output.json is a real backend response for the demo patient
(from Biolabs frontend/src/jsons/dummy-user-data.json): 4 visits with each value and the
scaled value the backend stored, plus the food scores it stored after the last visit.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pyclinics import FoodRecommender, ReferenceData, Tuning

EXPECTED = json.loads((Path(__file__).parent / "data/backend_demo_output.json").read_text())
LAST = EXPECTED["entries"][-1]["biometrics"]
STORED = EXPECTED["foods_score"]
SEX = EXPECTED["gender"]
BIOLABS = Path(os.environ.get("BIOLABS_ROOT", Path(__file__).resolve().parents[2] / "Biolabs"))


@pytest.fixture(scope="module")
def recommender():
    return FoodRecommender(tuning=Tuning.legacy())


def test_bundled_shapes(recommender):
    ref = recommender.reference
    assert ref.food_bias.shape == (58, 150)
    assert ref.nutrient_weight.shape == (58, 75)
    assert ref.food_nutrients.shape == (150, 75)


def test_deviations_match_backend(recommender):
    for entry in EXPECTED["entries"]:
        values = {b["id"]: b["value"] for b in entry["biometrics"]}
        got = recommender.deviations(values, SEX)["deviation"].to_numpy()
        expected = np.array([b["scaled_value"] for b in entry["biometrics"]])
        assert np.allclose(got, expected, atol=1e-9), np.abs(got - expected).max()


def test_food_scores_match_backend(recommender):
    result = recommender.score({b["id"]: b["scaled_value"] for b in LAST}).to_dict()
    assert list(result) == list(STORED)  # same foods, same order
    assert max(abs(result[k] - STORED[k]) for k in STORED) < 1e-9


def test_recommend_end_to_end_matches_backend(recommender):
    result = recommender.recommend({b["id"]: b["value"] for b in LAST}, sex=SEX).to_dict()
    assert max(abs(result[k] - STORED[k]) for k in STORED) < 1e-9


@pytest.mark.skipif(not (BIOLABS / "backend/datasets").exists(), reason="Biolabs repo not found; set BIOLABS_ROOT")
def test_bundled_data_is_the_backend_data():
    bundled, backend = ReferenceData.bundled(), ReferenceData.from_csv_dir(BIOLABS / "backend/datasets")
    for name in ("biochemicals", "foods", "food_nutrients", "food_bias", "food_weight", "nutrient_bias", "nutrient_weight"):
        pd.testing.assert_frame_equal(getattr(bundled, name), getattr(backend, name))
    expected = pd.read_csv(BIOLABS / "backend/datasets/normalized_nutrients.csv").set_index("name")
    cols = list(bundled.nutrients.index)
    assert np.allclose(bundled.normalized_nutrients, expected.loc[bundled.foods.index, cols].to_numpy(), atol=1e-4)


@pytest.mark.skipif(not (BIOLABS / "frontend/src/jsons/dummy-user-data.json").exists(), reason="Biolabs repo not found")
def test_reads_the_frontend_demo_file_directly(recommender):
    result = recommender.recommend(BIOLABS / "frontend/src/jsons/dummy-user-data.json").to_dict()
    assert max(abs(result[k] - STORED[k]) for k in STORED) < 1e-9
