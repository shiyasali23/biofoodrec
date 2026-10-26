import json

import numpy as np
import pytest

from biofoodrec import FoodRecommender, Tuning, sample_patient


@pytest.fixture(scope="module")
def result():
    return FoodRecommender(tuning=Tuning.recommended()).recommend(sample_patient())


def test_readable_tables(result):
    foods = result.foods(10)
    assert list(foods.columns) == ["Food", "Group", "Score", "Why"]
    assert foods.Score.between(0, 100).all() and foods.Score.is_monotonic_decreasing
    assert foods.Why.str.startswith("helps").all()
    assert result.limit(5).Score.is_monotonic_increasing
    labs = result.labs()
    assert labs.Status.iloc[0] in ("High", "Low")


def test_nutrients(result):
    nutrients = result.nutrients()
    assert len(nutrients) == 75
    assert set(nutrients.Advice) <= {"Eat more", "Eat less", "No change"}
    assert nutrients.Need.max() == 100 or nutrients.Need.min() == -100


def test_explain_adds_up_to_the_score(result):
    m = result.matrices()
    for food in list(result.table.index[:5]) + list(result.table.index[-5:]):
        parts = result.explain(food)
        score_100 = 100 * result.table.loc[food, "score"] / 3
        assert np.isclose(m["base_points"][food] + parts.Points.sum(), score_100, atol=0.1)
        assert np.allclose(parts.Points, parts.Direct + parts["Through nutrients"], atol=0.02)


def test_matrices_rebuild_the_score(result):
    m = result.matrices()
    direct = m["drive"] @ m["food_coefficients"]
    need = m["drive"] @ m["nutrient_coefficients"]
    through = m["food_nutrients"] @ need
    through[m["guarded_foods"] & (through > 0)] = 0  # the food guard
    scale = lambda x: (x - x.min()) / (np.ptp(x) + 1e-9)
    assert np.allclose(scale(through) + scale(direct) + m["parts"].nutriscore, m["score"])
    assert np.allclose(m["pull"].sum(), direct + through)
    assert np.allclose(m["points"].sum() + m["base_points"], 100 * m["score"] / 3)


def test_update_changes_everything(result):
    worse = result.update({"Blood Glucose": 250})
    assert worse.labs().loc["Blood Glucose", "Status"] == "High"
    assert not worse.table.index[:10].equals(result.table.index[:10]) or not np.allclose(worse.table.score, result.table.score)
    assert "Was" in worse.compare(result).columns


def test_report_is_json(result):
    report = json.loads(result.to_json())
    assert {"labs", "foods", "limit", "nutrients", "summary", "settings"} <= set(report)
    assert set(report["foods"][0]) == {"rank", "food", "group", "score", "why"}


def test_plain_text_has_no_dashes(result):
    text = result.summary()
    assert "—" not in text and "–" not in text and " - " not in text
    assert "Eat more:" in text and "Best 10 foods" in text


def test_exclusions_and_boosts():
    rec = FoodRecommender()
    labs = sample_patient()
    base = rec.recommend(labs)
    first = base.table.index[0]
    assert first not in rec.recommend(labs, tuning=Tuning(exclude_foods=[first])).table.index
    assert rec.recommend(labs, tuning=Tuning(food_adjust={"Tacos": 5.0})).table.index[0] == "Tacos"
    only_prior = rec.recommend(labs, tuning=Tuning(food_mix=0, nutrient_mix=0))
    assert np.allclose(only_prior.table.score, only_prior.table.nutriscore)
