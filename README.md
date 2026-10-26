# biofoodrec

Personalized food recommendations from blood biochemistry. Load the reference CSVs, pass a person's
lab values, get every food ranked by how well it pulls their markers back toward healthy.

It is the Biolabs backend's food scorer (`utils/foods_score_handler.py` + `utils/biometrics_handler.py`)
taken out of Django, with every hard-coded number turned into a setting that can change per person.

## Install

```bash
pip install ./biofoodrec                  # from a local checkout
pip install git+https://github.com/<you>/biofoodrec   # once it is pushed
```

Needs Python 3.9+, numpy and pandas.

## Use

```python
from biofoodrec import FoodRecommender, ReferenceData, Tuning

ref = ReferenceData.from_csv_dir("Biolabs/backend/datasets")
rec = FoodRecommender(ref)

labs = {"Blood Glucose": 130, "Vitamin D": 14, "LDL Cholesterol": 175}
result = rec.recommend(labs, sex="female")

result.top(10)              # rank, score, food_effect, nutrient_effect, nutriscore, category
result.explain("Flax Seed") # each biochemical's pull on that food, split by pathway
result.to_dict()            # {food: score}, the shape the backend stores in FoodsScore
```

Lab values can be keyed by biochemical name or by the backend's 1-based biochemical id.
`rec.score({...})` takes values that are already scaled, like `POST /backend/foods_score/create`.

## How a score is made

1. Each lab value becomes a signed **deviation**: 0 at the middle of the healthy range, -1 at the
   low edge, +1 at the high edge.
2. Each food is scored two ways:
   - **food pathway**: the biochemical x food tables (bias + weight)
   - **nutrient pathway**: the biochemical x nutrient tables, times how much of each nutrient the food has
3. Both are multiplied by **-deviation**: if a marker is high and a food raises it, the food loses
   points. Foods that push markers back toward the middle win.
4. Each pathway is min-max scaled to 0-1 and the food's normalized nutriscore is added.

```
drive[b]        = -deviation[b] * importance[b]
food_effect[f]  = sum_b drive[b] * (bias_scale*food_bias[b,f] + weight_scale*food_weight[b,f])
nutrient_effect = normalized_nutrients @ (drive @ (bias_scale*nutrient_bias + weight_scale*nutrient_weight))
score[f]        = food_mix*norm(food_effect) + nutrient_mix*norm(nutrient_effect)
                  + nutriscore_mix*nutriscore[f] + food_adjust[f]
```

## Tuning

All settings live in one `Tuning` object that converts to and from JSON. An admin keeps a global
one; each person stores a small patch on top of it. Values in the patch replace single settings,
merge into dict settings key by key, and add to the exclusion lists.

```python
global_tuning = Tuning.recommended()
person = global_tuning.update({
    "importance": {"Blood Glucose": 2.0},
    "ranges": {"Vitamin D": [30, 60]},
    "exclude_groups": ["meat"],
})
rec.recommend(labs, sex="female", tuning=person)
```

| Setting | What it does |
|---|---|
| `scaling` | `"legacy"` (backend formula) or `"linear"` (deviation in half-ranges everywhere) |
| `cap`, `deadband` | clip large deviations; ignore small ones |
| `ranges` | person-specific healthy range per biochemical |
| `food_mix`, `nutrient_mix`, `nutriscore_mix` | weight of each part of the score (backend: 1, 1, 1) |
| `bias_scale`, `weight_scale` | scale the whole bias or weight tables |
| `importance` | multiplier per biochemical (0 ignores it) |
| `food_cells`, `nutrient_cells` | override single cells: `{biochemical: {food: [bias, weight]}}` |
| `food_adjust` | add to a food's final score |
| `exclude_foods`, `exclude_groups` | drop foods by name, category or sub-category |
| `normalize` | `"minmax"` per request (backend) or `"none"` |
| `precision` | `"float16"` reproduces the backend's stored scores exactly |

Presets: `Tuning.legacy()` is exactly what the backend computes today. `Tuning.recommended()` uses
linear scaling capped at 4.

### Why `recommended` exists

Outside the healthy range the backend measures distance in raw units (`value - max`). That makes
the deviation jump at the edge (1.0 at the max, 0.01 just above it), lets markers with large units
dominate (GGT 43 against a max of 36 counts as 7.0), and makes the backend's `is_hyper` flag miss
abnormal values whose excess is under one unit. Linear scaling measures every marker in the same
units: how many half-ranges it is from the middle.

## Reference data

`ReferenceData.from_csv_dir(folder)` reads these files (swap any one with a keyword argument, e.g.
`food_weight="my_weights.csv"`):

| Key | File | Shape |
|---|---|---|
| `biochemicals` | `biochemical_units.csv` | name, category, unit, validity, female_min/max, male_min/max |
| `foods` | `food_nutrients.csv` | name, category, sub_category, one column per nutrient, nutriscore |
| `nutrients` | `nutrients_categories.csv` | name, category, unit |
| `food_bias`, `food_weight` | `biochemical_foods_bias_df.csv`, `biochemical_foods_weights.csv` | biochemicals x foods |
| `nutrient_bias`, `nutrient_weight` | `biochemical_nutrients_bias.csv`, `biochemical_nutrients_weights.csv` | biochemicals x nutrients |

Tables are matched by name, not by position, and a missing row or column raises an error that names
it. `ReferenceData.from_frames(...)` takes DataFrames instead, for example ones built from the Django ORM.

## Tests

```bash
pip install "./biofoodrec[test]"
pytest
```

The parity tests read the CSVs and the demo fixture (`frontend/src/jsons/dummy-user-data.json`, a
real backend response) from the Biolabs repo, by default the `Biolabs` folder next to this one, or
wherever `BIOLABS_ROOT` points. They check that `Tuning.legacy()` reproduces the backend's stored
scaled values and food scores. Without the Biolabs data they are skipped.
