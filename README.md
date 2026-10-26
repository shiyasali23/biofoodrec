# biofoodrec

Food recommendations from blood test results.

Give it a person's blood test results. It tells you which results are high or low, which foods fit
that person best, which foods to limit, which nutrients to eat more or less of, and why.
Every bias, weight and setting can be changed for each person.

This is the food scoring from the Biolabs platform, taken out of the Django backend so a nutrition
or clinic platform can install it and use it directly. The Biolabs data is inside the package:
58 blood tests, 150 foods and 75 nutrients.

## Install

```bash
pip install ./biofoodrec
```

Or from GitHub once it is pushed: `pip install git+https://github.com/<you>/biofoodrec`.
It needs Python 3.9 or newer, numpy and pandas.

## Quick start

```python
from biofoodrec import FoodRecommender

rec = FoodRecommender()
result = rec.recommend({"Blood Glucose": 130, "Vitamin D": 14, "LDL Cholesterol": 175}, sex="female")

print(result.summary())     # a short report in plain words
result.labs()               # each result: high, low or normal
result.foods(20)            # the best foods, a score out of 100, and why
result.limit(10)            # foods to limit
result.nutrients()          # eat more or eat less, and which foods have each nutrient
result.explain("Salmon")    # why one food got its place
```

This is the summary for the demo patient from the Biolabs website (`sample_patient()`):

```text
58 lab results, 7 outside the healthy range.
High: C-Reactive Protein 4 mg/L (healthy 0 to 3); Gamma-Glutamyl Transferase 43 U/L (healthy 5 to 36); ...
Low: Vitamin D 18.4 ng/mL (healthy 20 to 50); Iron 47.8 µg/dL (healthy 50 to 170)

Eat more: Omega-3 Fatty Acids, Vitamin C, Antioxidants, Fiber, Magnesium, Polyphenols
Eat less: Trans Fatty Acids, Alcohol, Fats, Oxalates, Carbohydrates, Phytic Acid

Best 5 foods (score out of 100):
  1. Flax Seed  93.9  helps lower Blood Glucose and C-Reactive Protein
  2. Chia Seed  93  helps lower C-Reactive Protein and Blood Glucose
  3. Parsley  87.6  helps lower C-Reactive Protein and Blood Glucose
  4. Mint  86.6  helps lower C-Reactive Protein and Blood Glucose
  5. Salmon  84.5  helps raise Vitamin D and Iron

Foods to limit:
150. Brownies  2.8  raises Uric Acid and Blood Glucose
149. Donuts  10  raises Blood Glucose and Uric Acid
...
```

## Give it JSON

`recommend()` reads every shape Biolabs uses. Pass a Python object, a JSON string, or a path to a file.

| Shape | Where it comes from |
|---|---|
| `{"Blood Glucose": 130, "Vitamin D": 14}` | the simplest way |
| `{"Blood Glucose": {"value": 130}}` | the research notebooks |
| `[{"id": 1, "value": 130}]` | the backend request body (ids 1 to 58) |
| `{"data": {"updatedBiochemicalsData": [...], "unExpiredBiometricsData": [...]}}` | the frontend biometrics request |
| `{"user": {"gender": "female"}, "biometrics_entries": [...]}` | a full user, like the frontend demo |

When the JSON has no sex in it, add `sex="female"` or `sex="male"`. With several visits, the newest
result of each test is used. Older results the frontend sends already scaled are only used for tests
that were not measured again, the same as the backend does.

## Change a result

```python
after = result.update({"Blood Glucose": 180, "Vitamin D": 45})
after.compare(result)       # each food's new rank, old rank, and how far it moved
```

Everything updates: the lab table, the foods, the nutrients and the report.

## Personal settings

An admin keeps one `Tuning` for everyone. Each person gets a small patch on top of it. Only the patch
needs to be saved for that person.

```python
from biofoodrec import Tuning

clinic = Tuning.recommended()
person = clinic.update({
    "importance": {"Blood Glucose": 2.0},   # glucose counts double
    "ranges": {"Vitamin D": [30, 60]},      # this person's own target
    "exclude_groups": ["meat"],             # vegetarian
    "exclude_foods": ["Shrimp"],            # allergy
})
rec.recommend(labs, sex="female", tuning=person)
```

In a patch, a single value replaces the old one, a dictionary is merged key by key, and an exclusion
list is added to the old one. `Tuning.to_dict()` and `Tuning.from_dict()` turn it into JSON and back.

| Setting | What it changes |
|---|---|
| `importance` | how much each blood test counts (0 turns it off) |
| `ranges` | a person's own healthy range for a test |
| `bias_scale`, `weight_scale` | turn the whole bias table or the whole weight table up or down |
| `food_cells`, `nutrient_cells` | change single cells: `{"Blood Glucose": {"Banana": [bias, weight]}}` |
| `food_mix`, `nutrient_mix`, `nutriscore_mix` | how much each of the three parts counts (all 1 by default) |
| `food_adjust` | add points to a food, e.g. `{"Salmon": 0.2}` |
| `exclude_foods`, `exclude_groups` | remove foods by name, category or sub category |
| `scaling`, `cap`, `deadband` | how a result becomes a deviation (see below) |
| `normalize` | `"minmax"` scales each part to 0..1 (the backend), `"none"` keeps raw numbers |
| `precision` | `"float16"` copies the backend's rounding exactly |

## For your app

```python
result.to_json()    # everything as JSON: labs, foods, foods to limit, nutrients, summary, settings
result.report()     # the same, as a Python dictionary
result.to_dict()    # {food: score}, the shape the Biolabs backend stores today
```

## How the score works

A food gets points three ways:

1. **Direct**: what the food does to each blood test (the biochemical × food tables)
2. **Through nutrients**: what the nutrients in the food do to each test (the biochemical × nutrient tables)
3. **Nutriscore**: the food's general quality, the same for everyone

Each table cell is **bias + weight**. Bias is the direction: 1 means the food raises the test,
minus 1 means it lowers it. Weight is the strength. If a test is high and a food raises it, the food
loses points. Foods that bring the person's results back toward the middle of the healthy range win.

The whole thing is three matrix multiplications:

```text
drive    = deviation × importance × (minus 1)       58 numbers, one per test
direct   = drive @ food_coefficients                (58) @ (58 × 150)  =  150, one per food
need     = drive @ nutrient_coefficients            (58) @ (58 × 75)   =  75, one per nutrient
through  = food_nutrients @ need                    (150 × 75) @ (75)  =  150, one per food
score    = scale01(direct) + scale01(through) + nutriscore
```

`need` is also the nutrient advice: above zero means eat more, below zero means eat less.
`result.matrices()` returns every one of these as a labelled table, so you can check or study each step.

## Scaling: how a result becomes a deviation

Inside the healthy range, all three choices agree: 0 is the middle, 1 and minus 1 are the edges.
They differ outside the range.

| `scaling` | Outside the range | Note |
|---|---|---|
| `legacy` | distance in the test's own unit | The live backend today. HbA1c 5.8 (max 5.6) gets 0.2 and is called normal. Tests with big units count more. |
| `original` | 1 + distance in the test's own unit | The formula in the research notebook. The backend lost the 1. |
| `linear` | distance in half ranges | Every test is measured the same way. `Tuning.recommended()` uses this, capped at 4. |

`Tuning()` starts from `legacy` so it matches the live app. `Tuning.legacy()` matches it exactly,
down to the rounding.

## Data

The package ships these files from `Biolabs/backend/datasets`:

| File | What it holds |
|---|---|
| `biochemical_units.csv` | the 58 tests: unit, how many days a result stays valid, healthy ranges for women and men |
| `food_nutrients.csv` | the 150 foods: category, nutriscore, and the amount of each of the 75 nutrients |
| `nutrients_categories.csv` | the 75 nutrients and their units |
| `biochemical_foods_bias_df.csv`, `biochemical_foods_weights.csv` | test × food |
| `biochemical_nutrients_bias.csv`, `biochemical_nutrients_weights.csv` | test × nutrient |

To use your own tables:

```python
from biofoodrec import FoodRecommender, ReferenceData

ref = ReferenceData.from_csv_dir("my_folder")                                 # all 7 files
ref = ReferenceData.from_csv_dir("my_folder", food_weight="new_weights.csv")  # swap one file
rec = FoodRecommender(ref)
```

Tables are matched by name, so the column order does not matter. If a test or a food is missing,
the error message names it.

## Notebook

`examples/try_biofoodrec.ipynb` walks through everything with the demo patient, including the matrix
math step by step and sliders for the settings.

```bash
pip install "./biofoodrec[notebook]"
jupyter lab examples/try_biofoodrec.ipynb
```

## Tests

```bash
pip install "./biofoodrec[test]"
pytest
```

The tests check that `Tuning.legacy()` gives the same scaled values and food scores the live backend
stored for the demo patient (`tests/data/backend_demo_output.json`). If the Biolabs repo sits next to
this folder, or `BIOLABS_ROOT` points to it, two more tests check that the bundled data is the same as
the backend's and that the frontend demo file can be read directly.
