# biofoodrec

Food recommendations from blood test results.

Give it a person's blood test results. It tells you which results are high or low, which foods fit
that person best, which foods to limit, which nutrients to eat more or less of, and why.
Every bias, weight and setting can be changed for each person.

This is the food scoring from the Biolabs platform, taken out of the Django backend so a nutrition
or clinic platform can install it and use it directly. The Biolabs data is inside the package:
58 blood tests, 150 foods and 75 nutrients.

> **Not medical advice.** This is a decision support tool for nutritionists and clinicians, who stay
> responsible for what they recommend. Read [Limits to know](#limits-to-know) before using it with patients.

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
result.labs()               # each result: high, low or normal, and whether it changes the advice
result.foods(20)            # the best foods, a score out of 100, and why
result.limit(10)            # foods to limit
result.nutrients()          # eat more or eat less, and which foods have each nutrient
result.explain("Salmon")    # how many points each result gives one food
```

This is the start of the summary for the demo patient from the Biolabs website (`sample_patient()`):

```text
58 lab results, 7 outside the healthy range.
High: C-Reactive Protein 4 mg/L (healthy 0 to 3); Gamma-Glutamyl Transferase 43 U/L (healthy 5 to 36); ...
Low: Vitamin D 18.4 ng/mL (healthy 20 to 50); Iron 47.8 µg/dL (healthy 50 to 170)

Eat more: Vitamin C, Antioxidants, Fiber, Omega-3 Fatty Acids, Polyphenols, Magnesium
Eat less: Proteins, Caffeine, Alcohol, Fructose, Trans Fatty Acids, Oxalates

Best 5 foods (score out of 100):
  1. Parsley  88.1  helps lower C-Reactive Protein and Blood Glucose
  2. Flax Seed  85.9  helps lower C-Reactive Protein and Blood Glucose
  3. Cabbage  85.8  helps lower C-Reactive Protein and Blood Glucose
  ...
Foods to limit:
150. Brownies  9.4  raises Uric Acid and Blood Glucose
149. Pastries  15.7  raises Uric Acid and Gamma-Glutamyl Transferase
...
```

## Give it JSON

`recommend()` reads every shape Biolabs uses for lab results. Pass a Python object, a JSON string,
or a path to a file.

| Shape | Where it comes from |
|---|---|
| `{"Blood Glucose": 130, "Vitamin D": 14}` | the simplest way (ids work too: `{"1": 130}`) |
| `{"Blood Glucose": {"value": 130}}` | the research notebooks |
| `[{"id": 1, "value": 130}]` or `[{"name": "Blood Glucose", "value": 130}]` | a list of results |
| `{"data": {"updatedBiochemicalsData": [...], "unExpiredBiometricsData": [...]}}` | the frontend biometrics request |
| `{"user": {"gender": "female"}, "biometrics_entries": [...]}` | a full user, like the frontend demo |

When the JSON has no sex in it, add `sex="female"` or `sex="male"`. With several visits, the newest
result of each test is used, even when visits mix ids and names. Older results the frontend sends
already scaled are only used for tests that were not measured again, the same as the backend does.

The backend's foods score request, `{"data": [{"id": 1, "value": 2.4}]}`, holds values that are
already scaled, so it goes to `rec.score(...)` instead. `recommend()` refuses it with a message saying so.

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
from biofoodrec import FoodRecommender, Tuning, sample_patient

clinic = Tuning()
person = clinic.update({
    "importance": {"Blood Glucose": 2.0},   # glucose counts double
    "ranges": {"Vitamin D": [30, 60]},      # this person's own target
    "exclude_groups": ["seafood"],          # seafood allergy
    "exclude_foods": ["Whole Milk"],        # dislikes
})
result = FoodRecommender().recommend(sample_patient(), tuning=person)
```

In a patch, a single value replaces the old one, a dictionary is merged key by key, and a list is
added to the old one. `Tuning.to_dict()` saves the full settings as JSON and `Tuning.from_dict()`
reads them back exactly. Every name is checked: a typo like `"Blood Sugar"` stops with an error
instead of being ignored. Food groups are the categories and sub categories in the data, for example
`fruit`, `vegetable`, `seed`, `seafood`, `meat`, `dairy` and `nosh` (snacks and mixed dishes such as
burgers and pizza, which `meat` does not cover).

| Setting | What it changes |
|---|---|
| `act_in_range` | `False` (default): only results outside the healthy range change the advice. `True`: every result pushes toward the middle of its range, like the backend |
| `only_when` | tests where only one side matters, e.g. `{"LDL Cholesterol": "high"}` (see Safety) |
| `importance` | how much each blood test counts (0 turns it off) |
| `ranges` | a person's own healthy range for a test |
| `bias_scale`, `weight_scale` | turn the whole bias table or the whole weight table up or down |
| `food_cells`, `nutrient_cells` | change single cells: `{"Blood Glucose": {"Banana": [bias, weight]}}` |
| `food_mix`, `nutrient_mix`, `nutriscore_mix` | how much each of the three parts counts (all 1 by default) |
| `food_adjust` | add points to a food, e.g. `{"Salmon": 0.2}` |
| `exclude_foods`, `exclude_groups` | remove foods by name, category or sub category |
| `limit_only` | nutrients that can be advised down but never up (see Safety) |
| `food_guard` | foods with a nutriscore at or below this are never pushed up by lab results (see Safety) |
| `scaling`, `cap`, `deadband` | how a result becomes a deviation (see below) |
| `normalize` | `"minmax"` scales each part to 0..1 (the backend), `"none"` keeps raw numbers |
| `precision` | `"float16"` copies the backend's rounding exactly |

## For your app

```python
result.to_json()    # everything as JSON: labs, foods, foods to limit, nutrients, summary, settings
result.report()     # the same, as a Python dictionary
result.to_dict()    # {food: score}, the shape the Biolabs backend stores today
```

One patient takes about 3 ms, and the full JSON report about 16 ms.

## How the score works

A food gets points three ways:

1. **Direct**: what the food does to each blood test (the biochemical × food tables)
2. **Through nutrients**: what the nutrients in the food do to each test (the biochemical × nutrient tables)
3. **Nutriscore**: the food's general quality, the same for everyone

Each table cell is **bias + weight**. Bias comes from the research lists of foods and nutrients that
raise (1) or lower (minus 1) each test. Weight is a second, finer estimate between minus 1 and 1. The
score adds the two, so where they disagree they partly cancel. If a test is high and a food raises it,
the food loses points. Foods that bring the person's results back into the healthy range win.

The whole thing is three matrix multiplications:

```text
drive    = deviation × importance × (minus 1)       58 numbers, one per test
direct   = drive @ food_coefficients                (58) @ (58 × 150)  =  150, one per food
need     = drive @ nutrient_coefficients            (58) @ (58 × 75)   =  75, one per nutrient
through  = food_nutrients @ need                    (150 × 75) @ (75)  =  150, one per food
score    = scale01(direct) + scale01(through) + nutriscore
```

`need` is also the nutrient advice: above zero means eat more, below zero means eat less.
`result.matrices()` returns every step as a labelled table, so you can check or study each one.
`explain()` turns the same math into score points: a food's score out of 100 is its starting points
(general quality) plus the points from each test, exactly.

## Scaling: how a result becomes a deviation

Inside the healthy range, all three choices agree: 0 is the middle, 1 and minus 1 are the edges.
They differ outside the range.

| `scaling` | Outside the range | Note |
|---|---|---|
| `legacy` | distance in the test's own unit | The live backend today. HbA1c 5.8 (max 5.6) gets 0.2 and is called normal. Tests with big units count more. |
| `original` | 1 + distance in the test's own unit | The formula in the research notebook. The backend lost the 1. |
| `linear` | distance in half ranges | Every test is measured the same way. The default, capped at 4. |

`Tuning()` uses `linear` with all the safety rules below. `Tuning.legacy()` matches the live app
exactly, down to the rounding, with none of them.

## Safety

The tables say what each food and nutrient does to each test. Many of those facts are true but must
not be read as treatments. Four rules, all on by default, stop that:

1. **Only results outside the healthy range act** (`act_in_range=False`). A normal result never
   changes the advice, so a perfectly healthy person gets a list based on general food quality.
2. **One sided tests act on one side only** (`only_when`). A low LDL, total cholesterol, triglycerides,
   apolipoprotein B or CRP, and a high HDL or apolipoprotein A1, never ask for a diet change.
   Without this, a low LDL would recommend butter.
3. **`limit_only`**: alcohol, trans fats, saturated fats, cholesterol, sodium, free sugars, caffeine
   and the anti nutrients (oxalates, phytic acid, lectins, tannins) can be advised down, never up.
   "Alcohol lowers vitamin B1" is true, but it must never become "drink alcohol when B1 is high".
4. **`food_guard`**: foods with a nutriscore of 0.3 or less (bacon, sodas, pastries, burgers and the
   like) can move down the list for a person, never up.

The tests check every one of the 58 tests far above and far below its healthy range, for women and
men: no "eat more" advice for those nutrients, and no low quality food in the top 20. `explain()` stays
exact with all the rules on.

## Limits to know

1. **This is not medical advice.** It is a decision support tool for nutritionists and clinicians,
   who stay responsible for what they recommend.
2. **The tables were written by hand** in the Biolabs research notebooks and have not been clinically
   validated. `review/table_review.csv` lists 1,406 cells a nutritionist should check, for example
   cells where the bias and the weight point in opposite directions. Some advice follows from those
   cells and looks odd (a high vitamin D result puts tea and coffee near the top).
3. **One healthy range per sex.** Age, pregnancy and other conditions need a person's own range
   (`Tuning(ranges=...)`).
4. **The nutriscore is the team's own 0.1 to 0.9 rating**, not the official Nutri-Score label.
5. **Scores rank foods for one person.** They are not amounts or portion sizes. Because each part is
   scaled for each person, one result just outside its range can reorder the list as much as a
   severe one.

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

Tables are matched by name, so the column order does not matter. If a test or a food is missing or
listed twice, the error message names it.

## Notebook

`examples/try_biofoodrec.ipynb` walks through everything with the demo patient, including the matrix
math step by step, the safety rules, and sliders for the settings.

```bash
pip install "./biofoodrec[notebook]"
jupyter lab examples/try_biofoodrec.ipynb
```

## License

Apache License 2.0, see [LICENSE](LICENSE). This covers the code and the data tables that ship with it.

## Tests

```bash
pip install "./biofoodrec[test]"
pytest
```

The tests check that:

1. `Tuning.legacy()` gives the same scaled values and food scores the live backend stored for the
   demo patient (`tests/data/backend_demo_output.json`)
2. the safety rules hold for every test, high and low, for women and men
3. bad input (missing, not a number, negative, given twice) and typos in settings stop with a clear error
4. every JSON shape is read correctly, visits merge by date, and `explain()` and `matrices()` add up
   to the score

If the Biolabs repo sits next to this folder, or `BIOLABS_ROOT` points to it, two more tests check that
the bundled data is the same as the backend's and that the frontend demo file can be read directly.
GitHub runs the tests on Python 3.9 to 3.13 on every push (`.github/workflows/tests.yml`).
