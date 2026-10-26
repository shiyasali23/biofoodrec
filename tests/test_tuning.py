import pytest

from biofoodrec import Tuning


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
