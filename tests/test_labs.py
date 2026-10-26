import json

import pytest

from biofoodrec import FoodRecommender, read_labs, sample_patient


def test_simple_dict():
    labs = read_labs({"Blood Glucose": 130, "Vitamin D": "14"})
    assert labs.values == {"Blood Glucose": 130.0, "Vitamin D": 14.0} and labs.sex is None


def test_research_notebook_shape():
    labs = read_labs({"name": "x", "age": 40, "gender": "Female", "biometrics": {"Blood Glucose": {"value": 103}}})
    assert labs.values == {"Blood Glucose": 103.0} and labs.sex == "female"


def test_backend_list_and_json_text():
    labs = read_labs(json.dumps([{"id": 1, "value": 102.4}, {"name": "Iron", "value": 45}]))
    assert labs.values == {1: 102.4, "Iron": 45.0}


def test_biometrics_request_keeps_scaled_results_apart():
    body = {"data": {"updatedBiochemicalsData": [{"id": 1, "value": 130}],
                     "unExpiredBiometricsData": [{"id": 1, "value": 0.5}, {"id": 10, "value": -1.6}]}}
    labs = read_labs(body)
    assert labs.values == {1: 130.0} and labs.scaled == {1: 0.5, 10: -1.6}
    result = FoodRecommender().recommend(body, sex="male")
    deviation = result.matrices()["deviation"]
    assert deviation["Vitamin D"] == -1.6           # carried over
    assert deviation["Blood Glucose"] != 0.5        # the new result wins


def test_user_payload_takes_the_newest_visit(tmp_path):
    patient = sample_patient()
    path = tmp_path / "patient.json"
    path.write_text(json.dumps(patient))
    labs = read_labs(path)
    newest = max(patient["biometrics_entries"], key=lambda e: e["created_at"])["biometrics"]
    assert labs.sex == "female"
    assert labs.values == {b["name"]: b["value"] for b in newest}


def test_errors_are_clear():
    with pytest.raises(ValueError, match="not a number"):
        read_labs({"Blood Glucose": "high"})
    with pytest.raises(ValueError, match="sex is needed"):
        FoodRecommender().recommend({"Blood Glucose": 130})
    with pytest.raises(KeyError, match="unknown biochemical"):
        FoodRecommender().recommend({"Blood Sugar": 130}, sex="female")
