import pytest

from pyclinics import Tuning


def test_tuning_layers_and_round_trips():
    base = Tuning(importance={"Blood Glucose": 2.0}, exclude_foods=["Peanuts"])
    person = base.update({"importance": {"Vitamin D": 3.0}, "exclude_foods": ["Shrimp"], "cap": 3.0})
    assert person.importance == {"Blood Glucose": 2.0, "Vitamin D": 3.0}
    assert person.exclude_foods == ["Peanuts", "Shrimp"]
    assert Tuning.from_dict(person.to_dict()) == person
    with pytest.raises(ValueError):
        Tuning.from_dict({"not_a_knob": 1})


def test_rejects_bad_values():
    with pytest.raises(ValueError):
        Tuning(scaling="cubic")
    with pytest.raises(ValueError):
        Tuning(cap=0)
    with pytest.raises(ValueError):
        Tuning(deadband=-0.1)


def test_scaling_choices_at_the_edge():
    from pyclinics import FoodRecommender
    rec = FoodRecommender()
    just_over = {"Hemoglobin A1c": 5.8}  # healthy 4 to 5.6
    dev = lambda scaling: rec.deviations(just_over, "female", Tuning(scaling=scaling)).deviation.iloc[0]
    assert dev("legacy") == 0.2     # the backend: looks normal
    assert dev("original") == 1.2   # the research notebook: past the edge
    assert round(dev("linear"), 2) == 1.25


def test_lists_are_safe():
    with pytest.raises(ValueError, match="list of names"):
        Tuning(exclude_foods="Shrimp")
    with pytest.raises(ValueError, match="list of names"):
        Tuning().update({"exclude_foods": "Shrimp"})
    allergies = Tuning(exclude_foods=("Peanuts",)).update({"exclude_foods": ["Shrimp"]})
    assert allergies.exclude_foods == ["Peanuts", "Shrimp"]


def test_settings_never_share_data():
    base = Tuning(importance={"Blood Glucose": 2.0})
    person = base.update({"cap": 3.0})
    person.importance["Blood Glucose"] = 9.0
    assert base.importance["Blood Glucose"] == 2.0


def test_settings_can_be_dictionary_keys():
    cache = {Tuning(): "default", Tuning.legacy(): "legacy"}
    assert cache[Tuning()] == "default"
    assert Tuning(ranges={"Vitamin D": (30, 60)}) == Tuning.from_dict({"ranges": {"Vitamin D": [30, 60]}})
