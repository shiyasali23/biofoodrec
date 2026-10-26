"""Lab results in, ranked foods and nutrient advice out.

The whole score is a few matrix multiplications (Recommendation.matrices() returns each step):

    deviation   how far each lab result is from the middle of its healthy range (one number per test)
    drive       = deviation x importance x (minus 1)   a high result pushes down, a low result pushes up
    direct      = drive @ food_coefficients            one number per food
    need        = drive @ nutrient_coefficients        one number per nutrient: eat more or eat less
    through     = food_nutrients @ need                one number per food
    score       = food_mix x scale01(direct) + nutrient_mix x scale01(through)
                  + nutriscore_mix x nutriscore + food_adjust

where every coefficient is bias_scale x bias + weight_scale x weight.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .labs import Labs, normalize_sex, read_labs
from .reference import ReferenceData
from .tuning import Tuning


def _scale01(values: np.ndarray, method: str) -> np.ndarray:
    if method == "none":
        return values
    return (values - values.min()) / (np.ptp(values) + 1e-9)  # same guard as the backend


def _num(value) -> str:
    return f"{round(float(value), 2):g}"


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


class FoodRecommender:
    def __init__(self, reference: ReferenceData | None = None, tuning: Tuning | None = None):
        self.reference = reference or ReferenceData.bundled()
        self.tuning = tuning or Tuning()

    # Step 1: lab results to deviations

    def deviations(self, values: Mapping, sex: str, tuning: Tuning | None = None) -> pd.DataFrame:
        """Value, healthy range, deviation (0 = middle, 1 = edge of the range), health weight, status."""
        tuning = tuning or self.tuning
        ref = self.reference
        values = {ref.biochemical_name(k): float(v) for k, v in values.items()}
        ranges = ref.ranges(normalize_sex(sex)).loc[list(values)].copy()
        for name, (low, high) in tuning.ranges.items():
            if name in ranges.index:
                ranges.loc[name] = [low, high]
        if (ranges.high <= ranges.low).any():
            raise ValueError(f"empty range for {list(ranges.index[ranges.high <= ranges.low])}")

        low, high = ranges.low.to_numpy(float), ranges.high.to_numpy(float)
        if tuning.precision == "float16":  # the backend keeps ranges as float16
            low, high = low.astype(np.float16), high.astype(np.float16)
        value = np.array(list(values.values()), dtype=float)
        optimum = (low + high) / 2.0
        linear = 2 * (value - optimum) / (high - low)
        inside = (value >= low) & (value <= high)
        below = value < low

        if tuning.scaling == "linear":
            deviation, health = linear, 1 - np.abs(linear)
        else:
            # Outside the range both older formulas count raw units. The research notebook also added
            # 1 so a result just past the edge stays past 1; the backend dropped that 1.
            edge = 1.0 if tuning.scaling == "original" else 0.0
            outside = np.where(below, value - low - edge, value - high + edge)
            deviation = np.where(inside, np.round(linear, 2), np.round(outside, 2))
            health = np.where(inside, np.round(1 - np.abs(linear), 2),
                              np.where(below, np.round(value - low, 2), np.round(high - value, 2)))

        deviation = np.where(np.abs(deviation) < tuning.deadband, 0.0, deviation)
        if tuning.cap is not None:
            deviation = np.clip(deviation, -tuning.cap, tuning.cap)

        return pd.DataFrame({
            "value": value, "unit": ref.biochemicals.loc[list(values), "unit"].to_numpy(),
            "low": np.asarray(low, float), "high": np.asarray(high, float),
            "deviation": deviation, "health_weight": health,
            "status": np.where(value > high, "high", np.where(below, "low", "normal")),
        }, index=pd.Index(list(values), name="biochemical"))

    # Step 2: deviations to food scores

    def recommend(self, labs, sex: str | None = None, tuning: Tuning | None = None) -> Recommendation:
        """Lab results in any supported shape (see labs.py), ranked foods out.
        This is what POST /backend/biometrics does."""
        tuning = tuning or self.tuning
        labs = read_labs(labs)
        sex = normalize_sex(sex) or labs.sex
        if sex is None:
            raise ValueError("sex is needed for the healthy ranges: pass sex='female' or sex='male'")
        table = self.deviations(labs.values, sex, tuning)
        deviations = table["deviation"].to_dict()
        for key, value in labs.scaled.items():  # older results only count if not measured again
            deviations.setdefault(self.reference.biochemical_name(key), value)
        result = self.score(deviations, tuning, labs_table=table)
        result._source = (self, Labs(dict(table["value"]), sex, labs.scaled), sex)
        return result

    def score(self, deviations: Mapping, tuning: Tuning | None = None, labs_table: pd.DataFrame | None = None) -> Recommendation:
        """Deviations already scaled ({name or id: value}). This is what POST /backend/foods_score/create does."""
        tuning = tuning or self.tuning
        ref = self.reference
        bios, foods, nutrients = ref.biochemicals.index, ref.foods.index, ref.nutrients.index
        cast = (lambda a: a.astype(np.float16).astype(float)) if tuning.precision == "float16" else (lambda a: a)

        deviation = np.zeros(len(bios))
        for key, value in deviations.items():
            deviation[bios.get_loc(ref.biochemical_name(key))] = value
        deviation = cast(deviation)
        importance = np.array([tuning.importance.get(name, 1.0) for name in bios])
        drive = -deviation * importance

        food_bias, food_weight = self._tables(ref.food_bias, ref.food_weight, tuning.food_cells, foods)
        nutrient_bias, nutrient_weight = self._tables(ref.nutrient_bias, ref.nutrient_weight, tuning.nutrient_cells, nutrients)
        food_coef = tuning.bias_scale * cast(food_bias) + tuning.weight_scale * cast(food_weight)
        nutrient_coef = tuning.bias_scale * cast(nutrient_bias) + tuning.weight_scale * cast(nutrient_weight)
        food_nutrients = cast(ref.normalized_nutrients)
        nutriscore = cast(ref.normalized_nutriscore)

        direct = drive @ food_coef
        need = drive @ nutrient_coef
        through = food_nutrients @ need
        adjust = np.array([tuning.food_adjust.get(name, 0.0) for name in foods])
        parts = pd.DataFrame({"direct": _scale01(direct, tuning.normalize),
                              "through_nutrients": _scale01(through, tuning.normalize),
                              "nutriscore": nutriscore}, index=foods)
        score = (tuning.nutrient_mix * parts.through_nutrients.to_numpy() + tuning.food_mix * parts.direct.to_numpy()
                 + tuning.nutriscore_mix * nutriscore + adjust)

        return Recommendation(
            reference=ref, tuning=tuning, labs_table=labs_table, deviation=deviation, importance=importance,
            drive=drive, food_coef=food_coef, nutrient_coef=nutrient_coef, food_nutrients=food_nutrients,
            direct=direct, need=need, through=through, parts=parts, score=score,
        )

    @staticmethod
    def _tables(bias: pd.DataFrame, weight: pd.DataFrame, cells: dict, columns) -> tuple[np.ndarray, np.ndarray]:
        """The reference tables with any single cell changes applied."""
        if not cells:
            return bias.to_numpy(), weight.to_numpy()
        bias, weight = bias.copy(), weight.copy()
        for bio, row in cells.items():
            for item, (b, w) in row.items():
                if bio not in bias.index or item not in columns:
                    raise KeyError(f"unknown cell {bio!r} x {item!r}")
                bias.loc[bio, item], weight.loc[bio, item] = b, w
        return bias.to_numpy(), weight.to_numpy()


class Recommendation:
    """The result for one person. Readable tables: labs(), foods(), limit(), nutrients(), explain(food),
    summary(). For code: table, to_dict(), report(), to_json(). For research: matrices()."""

    def __init__(self, *, reference, tuning, labs_table, deviation, importance, drive, food_coef, nutrient_coef,
                 food_nutrients, direct, need, through, parts, score):
        self.reference, self.tuning, self._labs = reference, tuning, labs_table
        self._source = None
        bios, foods = reference.biochemicals.index, reference.foods.index
        self._vectors = dict(deviation=deviation, importance=importance, drive=drive, direct=direct, need=need,
                             through=through, score=score)
        self._food_coef, self._nutrient_coef, self._food_nutrients, self._parts = food_coef, nutrient_coef, food_nutrients, parts
        # pull[b, f]: how much biochemical b moves food f. Positive helps, negative hurts.
        self._pull_direct = drive[:, None] * food_coef
        self._pull_through = drive[:, None] * (nutrient_coef @ food_nutrients.T)
        self._pull = pd.DataFrame(self._pull_direct + self._pull_through, index=bios, columns=foods)

        table = pd.DataFrame({
            "score": score, "food_effect": direct, "nutrient_effect": through, "nutriscore": parts.nutriscore,
            "category": reference.foods["category"].to_numpy(), "sub_category": reference.foods["sub_category"].to_numpy(),
            "order": np.arange(len(foods)),
        }, index=pd.Index(foods, name="food"))
        groups = set(tuning.exclude_groups)
        keep = ~(table.index.isin(set(tuning.exclude_foods)) | table.category.isin(groups) | table.sub_category.isin(groups))
        table = table[keep].sort_values("score", ascending=False)
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        self.table = table

        # How far the labs moved the list away from a general healthy list (ranked by nutriscore only),
        # the "index difference" from the research notebook, as an average.
        general = table.sort_values("order")["nutriscore"].rank(ascending=False, method="first")
        self.moved_places = float((general - table["rank"]).abs().mean())

    def __repr__(self):
        return f"<Recommendation: {len(self.table)} foods, best: {', '.join(self.table.index[:3])}>"

    # Readable tables

    def _status_words(self) -> pd.Series:
        bios = self.reference.biochemicals.index
        deviation = pd.Series(self._vectors["deviation"], index=bios)
        words = pd.Series(np.where(deviation > 0, "Above middle", np.where(deviation < 0, "Below middle", "Middle")), index=bios)
        if self._labs is not None:
            labs = self._labs
            side = np.where(labs.deviation > 0, "Normal, upper side", np.where(labs.deviation < 0, "Normal, lower side", "Normal"))
            words.loc[labs.index] = np.where(labs.status == "high", "High", np.where(labs.status == "low", "Low", side))
        return words

    def labs(self) -> pd.DataFrame:
        """Each lab result, the most out of range first."""
        if self._labs is None:
            deviation = pd.Series(self._vectors["deviation"], index=self.reference.biochemicals.index)
            deviation = deviation[deviation != 0]
            return pd.DataFrame({"Status": self._status_words().loc[deviation.index], "How far off": deviation.round(2)})
        labs = self._labs
        out = pd.DataFrame({
            "Result": [f"{_num(v)} {u}" for v, u in zip(labs.value, labs.unit)],
            "Healthy range": [f"{_num(lo)} to {_num(hi)}" for lo, hi in zip(labs.low, labs.high)],
            "Status": self._status_words().loc[labs.index].to_numpy(),
            "How far off": labs.deviation.round(2).to_numpy(),
        }, index=pd.Index(labs.index, name="Biochemical"))
        order = (labs.status == "normal").astype(int).to_numpy() * 1000 - labs.deviation.abs().to_numpy()
        return out.iloc[np.argsort(order, kind="stable")]

    def _score_100(self, scores: pd.Series) -> pd.Series:
        total = self.tuning.food_mix + self.tuning.nutrient_mix + self.tuning.nutriscore_mix
        if self.tuning.normalize == "minmax" and total > 0:
            return (100 * scores / total).clip(0, 100).round(1)
        return (100 * (scores - scores.min()) / (np.ptp(scores) + 1e-9)).round(1)

    def _reasons(self, food: str, helps: bool, count: int = 2) -> str:
        pull = self._pull[food]
        pull = pull[pull > 0] if helps else pull[pull < 0]
        if not len(pull):
            return "general food quality" if helps else "low general food quality"
        deviation = pd.Series(self._vectors["deviation"], index=self.reference.biochemicals.index)
        if self._labs is not None:
            off_range = set(self._labs.index[self._labs.status != "normal"])
        else:
            off_range = set(deviation.index[deviation.abs() >= 1])
        # Results outside the healthy range are named first, then the strongest pulls.
        top = sorted(pull.index, key=lambda b: (b not in off_range, -abs(pull[b])))[:count]
        if helps:
            verbs = ["lower" if deviation[b] > 0 else "raise" for b in top]
        else:
            verbs = ["raises" if deviation[b] > 0 else "lowers" for b in top]
        phrase = f"{verbs[0]} {_join(top)}" if len(set(verbs)) == 1 else _join([f"{v} {b}" for v, b in zip(verbs, top)])
        return "helps " + phrase if helps else phrase

    def _food_rows(self, rows: pd.DataFrame, helps: bool) -> pd.DataFrame:
        return pd.DataFrame({
            "Food": rows.index,
            "Group": rows.category.str.title().to_numpy(),
            "Score": self._score_100(self.table.score).loc[rows.index].to_numpy(),
            "Why": [self._reasons(food, helps) for food in rows.index],
        }, index=pd.Index(rows["rank"].to_numpy(), name="Rank"))

    def foods(self, n: int | None = 20) -> pd.DataFrame:
        """Best foods first. Score is out of 100: 100 means the best possible match on all three parts."""
        rows = self.table if n is None else self.table.head(n)
        return self._food_rows(rows, helps=True)

    def limit(self, n: int = 10) -> pd.DataFrame:
        """The foods that fit this person least, worst first."""
        return self._food_rows(self.table.tail(n).iloc[::-1], helps=False)

    def nutrients(self) -> pd.DataFrame:
        """Which nutrients to eat more of or less of. Need goes from 100 (eat much more) to minus 100 (eat much less)."""
        ref = self.reference
        need = pd.Series(self._vectors["need"], index=ref.nutrients.index)
        scale = need.abs().max() or 1.0
        need_100 = (100 * need / scale).round(0)
        advice = pd.Series(np.where(need_100 >= 1, "Eat more", np.where(need_100 <= -1, "Eat less", "No change")), index=need.index)
        kept = [ref.foods.index.get_loc(f) for f in self.table.index]
        amounts = pd.DataFrame(self._food_nutrients[kept], index=self.table.index, columns=ref.nutrients.index)
        # Sources to eat more of come from the better half of this person's list; sources to cut down are any food.
        suits = amounts.index[: max(10, len(amounts) // 2)]

        def sources(nutrient):
            column = amounts.loc[suits if advice[nutrient] == "Eat more" else amounts.index, nutrient]
            return ", ".join(column[column > 0].nlargest(3).index)

        out = pd.DataFrame({
            "Advice": advice,
            "Need": need_100.astype(int),
            "Group": ref.nutrients["category"].to_numpy(),
            "Found in": [sources(n) for n in need.index],
        }, index=pd.Index(need.index, name="Nutrient"))
        return out.sort_values("Need", ascending=False, kind="stable")

    def explain(self, food: str) -> pd.DataFrame:
        """Each lab result's pull on one food. Positive pushes the food up the list, negative pushes it down."""
        if food not in self._pull.columns:
            raise KeyError(f"unknown food {food!r}")
        bios = self.reference.biochemicals.index
        f = self.reference.foods.index.get_loc(food)
        effect = self._food_coef[:, f] + self._nutrient_coef @ self._food_nutrients[f]
        out = pd.DataFrame({
            "Your result": self._status_words().to_numpy(),
            "This food": np.where(effect > 0, "raises it", np.where(effect < 0, "lowers it", "no effect")),
            "Direct": self._pull_direct[:, f].round(3) + 0.0,  # + 0.0 turns -0.0 into 0.0
            "Through nutrients": self._pull_through[:, f].round(3) + 0.0,
            "Total": (self._pull_direct[:, f] + self._pull_through[:, f]).round(3) + 0.0,
        }, index=pd.Index(bios, name="Biochemical"))
        out["Verdict"] = np.where(out.Total > 0, "Good", np.where(out.Total < 0, "Bad", ""))
        out = out[self._vectors["deviation"] != 0]
        return out.reindex(out.Total.abs().sort_values(ascending=False, kind="stable").index)

    def summary(self, n: int = 10) -> str:
        lines = []
        if self._labs is not None:
            labs = self._labs
            off = labs[labs.status != "normal"]
            lines.append(f"{len(labs)} lab results, {len(off)} outside the healthy range.")
            for status, word in (("high", "High"), ("low", "Low")):
                rows = off[off.status == status]
                rows = rows.reindex(rows.deviation.abs().sort_values(ascending=False).index)
                if len(rows):
                    items = [f"{b} {_num(r.value)} {r.unit} (healthy {_num(r.low)} to {_num(r.high)})" for b, r in rows.iterrows()]
                    lines.append(f"{word}: " + "; ".join(items))
            lines.append("")
        nutrients = self.nutrients()
        more = nutrients[nutrients.Advice == "Eat more"].head(6).index
        less = nutrients[nutrients.Advice == "Eat less"].sort_values("Need").head(6).index
        lines.append("Eat more: " + (", ".join(more) or "nothing in particular"))
        lines.append("Eat less: " + (", ".join(less) or "nothing in particular"))
        lines += ["", f"Best {n} foods (score out of 100):"]
        lines += [f"{rank:>3}. {r.Food}  {r.Score:g}  {r.Why}" for rank, r in self.foods(n).iterrows()]
        lines += ["", "Foods to limit:"]
        lines += [f"{rank:>3}. {r.Food}  {r.Score:g}  {r.Why}" for rank, r in self.limit(min(5, n)).iterrows()]
        lines += ["", f"Compared with a general healthy eating list, these results moved each food "
                      f"{self.moved_places:.0f} places on average."]
        return "\n".join(lines)

    # Changing the inputs

    def update(self, changes: Mapping) -> Recommendation:
        """Change some lab results and get a new result. Everything else stays the same."""
        if self._source is None:
            raise ValueError("update needs a result made by recommend(), not score()")
        recommender, labs, sex = self._source
        values = dict(labs.values)
        values.update({recommender.reference.biochemical_name(k): float(v) for k, v in changes.items()})
        return recommender.recommend(Labs(values, sex, labs.scaled), tuning=self.tuning)

    def compare(self, before: Recommendation, n: int = 15) -> pd.DataFrame:
        """This result's best foods next to their rank in another result. Moved above 0 means it went up."""
        top = self.table.head(n)
        was = before.table["rank"].reindex(top.index)
        return pd.DataFrame({"Rank": top["rank"], "Was": was.astype("Int64"), "Moved": (was - top["rank"]).astype("Int64"),
                             "Score": self._score_100(self.table.score).loc[top.index]}).rename_axis("Food")

    # For code and research

    def to_dict(self) -> dict[str, float]:
        """{food: score} in reference order, the shape FoodsScore.foods_score stores today."""
        return self.table.sort_values("order")["score"].astype(float).to_dict()

    def report(self, n: int = 20) -> dict:
        """Everything as plain JSON ready data, for an API response."""
        def records(frame):
            frame = frame.reset_index()
            frame.columns = [c.lower().replace(" ", "_") for c in frame.columns]
            return json.loads(frame.to_json(orient="records"))
        return {
            "labs": records(self.labs()), "foods": records(self.foods(n)), "limit": records(self.limit()),
            "nutrients": records(self.nutrients()), "moved_places": round(self.moved_places, 1),
            "summary": self.summary(), "settings": self.tuning.to_dict(),
        }

    def to_json(self, n: int = 20, indent: int | None = 2) -> str:
        return json.dumps(self.report(n), indent=indent, ensure_ascii=False)

    def matrices(self) -> dict[str, pd.Series | pd.DataFrame]:
        """Every step of the math as labelled tables (all foods, before exclusions)."""
        ref = self.reference
        bios, foods, nutrients = ref.biochemicals.index, ref.foods.index, ref.nutrients.index
        v = self._vectors
        return {
            "deviation": pd.Series(v["deviation"], index=bios, name="deviation"),
            "drive": pd.Series(v["drive"], index=bios, name="drive"),
            "food_coefficients": pd.DataFrame(self._food_coef, index=bios, columns=foods),
            "nutrient_coefficients": pd.DataFrame(self._nutrient_coef, index=bios, columns=nutrients),
            "food_nutrients": pd.DataFrame(self._food_nutrients, index=foods, columns=nutrients),
            "direct": pd.Series(v["direct"], index=foods, name="direct"),
            "need": pd.Series(v["need"], index=nutrients, name="need"),
            "through_nutrients": pd.Series(v["through"], index=foods, name="through_nutrients"),
            "parts": self._parts.copy(),
            "score": pd.Series(v["score"], index=foods, name="score"),
            "pull": self._pull.copy(),
        }
