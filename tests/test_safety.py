"""Safety checks that must hold for every patient, plus input and settings validation."""
import json
import math

import pytest

from pyclinics import FoodRecommender, Tuning, sample_patient
from pyclinics.tuning import LIMIT_ONLY

REC = FoodRecommender()


def one_test_patients():
    """Every test far above and far below its healthy range, for women and men."""
    for sex in ("female", "male"):
        for name, (low, high) in REC.reference.ranges(sex).iterrows():
            middle, half = (low + high) / 2, (high - low) / 2
            yield sex, name, middle + 2.5 * half
            if low > 0:
                yield sex, name, max(middle - 2.5 * half, 0.0)


def test_never_advises_more_of_a_limit_nutrient():
    for sex, name, value in one_test_patients():
        advice = REC.recommend({name: value}, sex=sex).nutrients()
        more = set(advice.index[advice.Advice == "Eat more"])
        assert not more & set(LIMIT_ONLY), (sex, name, value, more & set(LIMIT_ONLY))


def test_never_puts_a_processed_food_in_the_top_20():
    foods = REC.reference.foods
    processed = set(foods.index[foods.nutriscore <= 0.3 + 1e-9])
    for sex, name, value in one_test_patients():
        top = set(REC.recommend({name: value}, sex=sex).table.index[:20])
        assert not top & processed, (sex, name, value, top & processed)


def test_guards_keep_explain_exact():
    result = REC.recommend({"Creatinine": 2.2, "Blood Urea Nitrogen": 40, "Potassium": 5.9}, sex="male")
    base = result.matrices()["base_points"]
    for food in ("Burgers", "Beer", "Lettuce"):
        score_100 = 100 * result.table.loc[food, "score"] / 3
        assert math.isclose(base[food] + result.explain(food).Points.sum(), score_100, abs_tol=0.05)


def test_healthy_results_do_not_change_the_advice():
    # Reviewer case: LDL 99 and LDL 60 used to ask for butter. Results inside the range now do nothing.
    general = REC.recommend({"Blood Glucose": 85}, sex="female")
    for labs in ({"LDL Cholesterol": 99}, {"LDL Cholesterol": 60}, {"C-Reactive Protein": 0.2},
                 {"HDL Cholesterol": 80}, {"Vitamin D": 49}):
        result = REC.recommend(labs, sex="female")
        assert result.acting.empty, labs
        assert list(result.table.index) == list(general.table.index), labs
        assert set(result.nutrients().Advice) == {"No change"}, labs
        assert "None of these results calls for a diet change" in result.summary()


def test_one_sided_tests_only_act_on_their_side():
    assert REC.recommend({"LDL Cholesterol": 30}, sex="female").acting.empty      # low LDL: fine
    assert REC.recommend({"HDL Cholesterol": 125}, sex="female").acting.empty     # high HDL: fine
    assert REC.recommend({"C-Reactive Protein": 0}, sex="female").acting.empty
    assert not REC.recommend({"LDL Cholesterol": 190}, sex="female").acting.empty
    assert not REC.recommend({"HDL Cholesterol": 30}, sex="female").acting.empty
    low_ldl = REC.recommend({"LDL Cholesterol": 30}, sex="female").summary()
    assert "No diet change for: LDL Cholesterol (low)" in low_ldl


def test_reasons_only_name_tests_that_help():
    result = REC.recommend(sample_patient())
    points = result.matrices()["points"]
    for food in result.table.index[:30]:
        why = result.foods(None).set_index("Food").loc[food, "Why"]
        for name in result.acting.index:
            if f" {name}" in f" {why.replace('helps lower', '').replace('helps raise', '')}" and points.loc[name, food] <= 0:
                raise AssertionError(f"{food}: '{why}' names {name}, which costs it points")


def test_legacy_has_no_guards():
    legacy = Tuning.legacy()
    assert legacy.limit_only == [] and legacy.food_guard is None and legacy.scaling == "legacy"
    assert Tuning.from_dict(json.loads(json.dumps(legacy.to_dict()))) == legacy


@pytest.mark.parametrize("labs, message", [
    ({}, "no lab results"),
    ({"Blood Glucose": float("nan")}, "numbers of 0 or more"),
    ({"Blood Glucose": float("inf")}, "numbers of 0 or more"),
    ({"Blood Glucose": -5}, "numbers of 0 or more"),
    ({"Blood Glucose": 100, 1: 120}, "given twice"),
])
def test_bad_lab_results_are_refused(labs, message):
    with pytest.raises(ValueError, match=message):
        REC.recommend(labs, sex="female")


@pytest.mark.parametrize("settings, message", [
    ({"importance": {"Blood Sugar": 2}}, "unknown names"),
    ({"exclude_foods": ["Peanuts"]}, "unknown names"),
    ({"exclude_groups": ["vegan"]}, "unknown names"),
    ({"limit_only": ["Sugar"]}, "unknown names"),
])
def test_typos_in_settings_are_refused(settings, message):
    with pytest.raises(ValueError, match=message):
        REC.recommend(sample_patient(), tuning=Tuning(**settings))


@pytest.mark.parametrize("settings", [
    {"importance": {"Blood Glucose": -1}},
    {"food_mix": -1},
    {"food_mix": 0, "nutrient_mix": 0, "nutriscore_mix": 0},
    {"ranges": {"Vitamin D": [60, 30]}},
])
def test_unsafe_settings_are_refused(settings):
    with pytest.raises(ValueError):
        Tuning(**settings)


def test_excluding_every_food_is_refused():
    groups = sorted(set(REC.reference.foods.category))
    with pytest.raises(ValueError, match="every food"):
        REC.recommend(sample_patient(), tuning=Tuning(exclude_groups=groups))


def test_explain_wording_agrees_with_the_verdict():
    result = REC.recommend(sample_patient())
    for food in result.table.index:
        rows = result.explain(food)
        rows = rows[rows.Points.abs() >= 0.01]
        good = ((rows["Your result"] == "High") & (rows["This food"] == "lowers it")) | \
               ((rows["Your result"] == "Low") & (rows["This food"] == "raises it"))
        assert (good == (rows.Verdict == "Good")).all(), (food, rows[good != (rows.Verdict == "Good")])
