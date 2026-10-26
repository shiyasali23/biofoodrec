# Table review for a nutritionist

`table_review.csv` lists 1,406 cells in the bias and weight tables that a qualified nutritionist or
dietitian should check before these tables are used with patients.

| Issue | Cells | What it means |
|---|---|---|
| bias and weight point in opposite directions | 1,189 | The bias says the food or nutrient lowers the test, the weight says it raises it (or the other way round). The score adds them, so one cancels part of the other. |
| a nutrient to limit is said to lower a test | 217 | Often true (alcohol lowers vitamin B1), but read as a treatment it would mean "drink alcohol if B1 is high". The safety guard (`Tuning.limit_only`) stops this from becoming advice, but the cells are listed so the tables can be cleaned. |

Columns: `table` (food or nutrient), `biochemical`, `food_or_nutrient`, `bias`, `weight`, `issue`.

To fix a cell, edit the CSV in `src/biofoodrec/data/` or override it per clinic with
`Tuning(food_cells=...)` or `Tuning(nutrient_cells=...)`.
