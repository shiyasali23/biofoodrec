"""Lab results in, ranked foods and nutrient advice out.

The whole score is a few matrix multiplications (Recommendation.matrices() returns each step):

    deviation   how far each lab result is from the middle of its healthy range (one number per test).
                By default only results outside the range act, and one sided tests act on one side only.
    drive       = deviation x importance x (minus 1)   a high result pushes down, a low result pushes up
    direct      = drive @ food_coefficients            one number per food
    need        = drive @ nutrient_coefficients        one number per nutrient: eat more or eat less
                  (a nutrient in Tuning.limit_only is never pushed above 0: the safety guard)
    through     = food_nutrients @ need                one number per food
                  (a food with nutriscore at or below Tuning.food_guard is never pushed above 0)
    score       = food_mix x scale01(direct) + nutrient_mix x scale01(through)
                  + nutriscore_mix x nutriscore + food_adjust

where every coefficient is bias_scale x bias + weight_scale x weight.
"""
from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np
import pandas as pd

from .labs import Labs, normalize_sex, read_labs, read_scaled
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


def _one_sided(names, deviation: np.ndarray, tuning: Tuning) -> np.ndarray:
    """Zero the side of a one sided test that is not a problem (a low LDL, a high HDL)."""
    side = np.array([tuning.only_when.get(name, "") for name in names])
    return np.where(((side == "high") & (deviation < 0)) | ((side == "low") & (deviation > 0)), 0.0, deviation)


class FoodRecommender:
    def __init__(self, reference: ReferenceData | None = None, tuning: Tuning | None = None):
        self.reference = reference or ReferenceData.bundled()
        self.tuning = tuning or Tuning()

    # Step 1: lab results to deviations

    def deviations(self, values: Mapping, sex: str, tuning: Tuning | None = None) -> pd.DataFrame:
        """Value, healthy range, position (0 = middle, 1 = edge, in half ranges), the deviation that
        acts on the advice, health weight and status."""
        tuning = tuning or self.tuning
        self._check(tuning)
        ref = self.reference
        named = {}
        for key, value in values.items():
            name = ref.biochemical_name(key)
            if name in named:
                raise ValueError(f"{name} is given twice (once by name and once by id?)")
            named[name] = float(value)
        values = named
        if not values:
            raise ValueError("no lab results given")
        if bad := [name for name, value in values.items() if not np.isfinite(value) or value < 0]:
            raise ValueError(f"lab results must be numbers of 0 or more: {bad}")
        if sex is None:
            raise ValueError("sex is needed for the healthy ranges: pass sex='female' or sex='male'")
        names = list(values)
        ranges = ref.ranges(normalize_sex(sex)).loc[names].copy()
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

        if not tuning.act_in_range:
            deviation = np.where(inside, 0.0, deviation)
        else:
            deviation = np.where(inside & (np.abs(deviation) < tuning.deadband), 0.0, deviation)
        deviation = _one_sided(names, deviation, tuning)
        if tuning.cap is not None:
            deviation = np.clip(deviation, -tuning.cap, tuning.cap)

        return pd.DataFrame({
            "value": value, "unit": ref.biochemicals.loc[names, "unit"].to_numpy(),
            "low": np.asarray(low, float), "high": np.asarray(high, float),
            "position": linear, "deviation": deviation + 0.0, "health_weight": health,
            "status": np.where(value > high, "high", np.where(below, "low", "normal")),
        }, index=pd.Index(names, name="biochemical"))

    # Step 2: deviations to food scores

    def recommend(self, labs, sex: str | None = None, tuning: Tuning | None = None) -> Recommendation:
        """Lab results in any supported shape (see labs.py), ranked foods out.
        This is what POST /backend/biometrics does."""
        tuning = tuning or self.tuning
        labs = read_labs(labs, resolve=self.reference.biochemical_name)
        sex = normalize_sex(sex) or labs.sex
        if sex is None:
            raise ValueError("sex is needed for the healthy ranges: pass sex='female' or sex='male'")
        table = self.deviations(labs.values, sex, tuning)
        deviations = table["deviation"].to_dict()
        for key, value in labs.scaled.items():  # older results only count if not measured again
            name = self.reference.biochemical_name(key)
            if name not in deviations:
                deviations[name] = self._acting(name, value, tuning)
        result = self.score(deviations, tuning, labs_table=table, _acting=False)
        result._source = (self, Labs(dict(table["value"]), sex, labs.scaled), sex)
        return result

    def score(self, deviations, tuning: Tuning | None = None, labs_table: pd.DataFrame | None = None,
              _acting: bool = True) -> Recommendation:
        """Deviations already scaled: {name or id: value}, a list of {id, value}, or the foods_score
        request {"data": [...]}. This is what POST /backend/foods_score/create does. A deviation between
        minus 1 and 1 counts as inside the healthy range."""
        tuning = tuning or self.tuning
        self._check(tuning)
        ref = self.reference
        bios, foods, nutrients = ref.biochemicals.index, ref.foods.index, ref.nutrients.index
        cast = (lambda a: a.astype(np.float16).astype(float)) if tuning.precision == "float16" else (lambda a: a)

        deviation = np.zeros(len(bios))
        for name, value in read_scaled(deviations, ref.biochemical_name).items():
            if not np.isfinite(value):
                raise ValueError(f"deviation for {name!r} is not a finite number: {value}")
            deviation[bios.get_loc(name)] = self._acting(name, value, tuning) if _acting else value
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
        # Safety guard: a nutrient in limit_only can be advised down, never up. Many true facts read
        # wrongly as treatments ("alcohol lowers vitamin B1" must not become "drink if B1 is high").
        # The nutrient's column is switched off for this person, so explain() still adds up exactly.
        need_before_guard = need
        blocked = nutrients.isin(tuning.limit_only) & (need > 0)
        if blocked.any():
            nutrient_coef = np.where(blocked, 0.0, nutrient_coef)
            need = drive @ nutrient_coef
        through = food_nutrients @ need
        # Food guard: a food with a poor general quality (nutriscore at or below food_guard) can be pushed
        # down by lab results but never up, so low sodium never puts bacon at the top.
        guarded = np.zeros(len(foods), dtype=bool)
        if tuning.food_guard is not None:
            guarded = ref.foods["nutriscore"].to_numpy(float) <= tuning.food_guard + 1e-9
        held_direct, held_through = guarded & (direct > 0), guarded & (through > 0)
        if held_direct.any():
            food_coef = np.where(held_direct, 0.0, food_coef)
            direct = drive @ food_coef
        through = np.where(held_through, 0.0, through)
        adjust = np.array([tuning.food_adjust.get(name, 0.0) for name in foods])
        parts = pd.DataFrame({"direct": _scale01(direct, tuning.normalize),
                              "through_nutrients": _scale01(through, tuning.normalize),
                              "nutriscore": nutriscore}, index=foods)
        score = (tuning.nutrient_mix * parts.through_nutrients.to_numpy() + tuning.food_mix * parts.direct.to_numpy()
                 + tuning.nutriscore_mix * nutriscore + adjust)

        return Recommendation(
            reference=ref, tuning=tuning, labs_table=labs_table, deviation=deviation, importance=importance,
            drive=drive, food_coef=food_coef, nutrient_coef=nutrient_coef, food_nutrients=food_nutrients,
            direct=direct, need=need, need_before_guard=need_before_guard, through=through, parts=parts, score=score,
            adjust=adjust, guarded=guarded, held_through=held_through,
        )

    @staticmethod
    def _acting(name: str, value: float, tuning: Tuning) -> float:
        """The same rules deviations() applies, for a value that arrives already scaled."""
        value = float(value)
        if abs(value) <= 1 and (not tuning.act_in_range or abs(value) < tuning.deadband):
            value = 0.0
        value = float(_one_sided([name], np.array([value]), tuning)[0])
        return value if tuning.cap is None else float(np.clip(value, -tuning.cap, tuning.cap))

    def _check(self, tuning: Tuning) -> None:
        """Every name in the settings must exist, so a typo never fails silently."""
        ref = self.reference
        groups = set(ref.foods["category"]) | set(ref.foods["sub_category"])
        checks = {
            "importance": (tuning.importance, ref.biochemicals.index),
            "ranges": (tuning.ranges, ref.biochemicals.index),
            "only_when": (tuning.only_when, ref.biochemicals.index),
            "limit_only": (tuning.limit_only, ref.nutrients.index),
            "food_adjust": (tuning.food_adjust, ref.foods.index),
            "exclude_foods": (tuning.exclude_foods, ref.foods.index),
            "exclude_groups": (tuning.exclude_groups, groups),
        }
        problems = [f"{key} {sorted(set(given) - set(valid))}" for key, (given, valid) in checks.items()
                    if set(given) - set(valid)]
        if problems:
            raise ValueError("unknown names in settings: " + "; ".join(problems))

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
                 food_nutrients, direct, need, need_before_guard, through, parts, score, adjust, guarded, held_through):
        self.reference, self.tuning, self._labs = reference, tuning, labs_table
        self._source = None
        self._nutrients = None
        bios, foods = reference.biochemicals.index, reference.foods.index
        self._vectors = dict(deviation=deviation, importance=importance, drive=drive, direct=direct, need=need,
                             need_before_guard=need_before_guard, through=through, score=score, guarded=guarded)
        self._food_coef, self._nutrient_coef, self._food_nutrients, self._parts = food_coef, nutrient_coef, food_nutrients, parts

        # pull[b, f]: how much test b moves food f, in the raw units of each part.
        self._pull_direct = drive[:, None] * food_coef
        self._pull_through = drive[:, None] * (nutrient_coef @ food_nutrients.T)
        self._pull_through[:, held_through] = 0.0  # the food guard held these foods at 0
        self._pull = pd.DataFrame(self._pull_direct + self._pull_through, index=bios, columns=foods)

        # points[b, f]: the same pull in score points out of 100, after each part is scaled and mixed.
        # A food's score out of 100 = base[f] + the sum of its points, exactly.
        total = tuning.food_mix + tuning.nutrient_mix + tuning.nutriscore_mix
        if tuning.normalize == "minmax":
            to_direct = tuning.food_mix / (np.ptp(direct) + 1e-9)
            to_through = tuning.nutrient_mix / (np.ptp(through) + 1e-9)
            shift = to_direct * direct.min() + to_through * through.min()
        else:
            to_direct, to_through, shift = tuning.food_mix, tuning.nutrient_mix, 0.0
        self._points_direct = 100 / total * to_direct * self._pull_direct
        self._points_through = 100 / total * to_through * self._pull_through
        self._points = pd.DataFrame(self._points_direct + self._points_through, index=bios, columns=foods)
        self._base = pd.Series(100 / total * (tuning.nutriscore_mix * parts.nutriscore.to_numpy() + adjust - shift),
                               index=foods)

        table = pd.DataFrame({
            "score": score, "food_effect": direct, "nutrient_effect": through, "nutriscore": parts.nutriscore,
            "category": reference.foods["category"].to_numpy(), "sub_category": reference.foods["sub_category"].to_numpy(),
            "order": np.arange(len(foods)),
        }, index=pd.Index(foods, name="food"))
        groups = set(tuning.exclude_groups)
        keep = ~(table.index.isin(set(tuning.exclude_foods)) | table.category.isin(groups) | table.sub_category.isin(groups))
        if not keep.any():
            raise ValueError("every food is excluded by the settings")
        table = table[keep].sort_values("score", ascending=False, kind="stable")
        table.insert(0, "rank", np.arange(1, len(table) + 1))
        self.table = table

        # How far the labs moved the list away from a general healthy list (ranked by nutriscore only),
        # the "index difference" from the research notebook, as an average.
        general = table.sort_values("order")["nutriscore"].rank(ascending=False, method="first")
        self.moved_places = float((general - table["rank"]).abs().mean())

    def __repr__(self):
        return f"<Recommendation: {len(self.table)} foods, best: {', '.join(self.table.index[:3])}>"

    @property
    def acting(self) -> pd.Series:
        """The tests that change the advice for this person, with their deviation."""
        deviation = pd.Series(self._vectors["deviation"], index=self.reference.biochemicals.index)
        return deviation[deviation != 0]

    # Readable tables

    def _status_words(self) -> pd.Series:
        bios = self.reference.biochemicals.index
        deviation = pd.Series(self._vectors["deviation"], index=bios)
        words = pd.Series(np.where(deviation > 0, "Above middle", np.where(deviation < 0, "Below middle", "Middle")), index=bios)
        if self._labs is not None:
            labs = self._labs
            side = np.where(labs.position > 0, "Normal, upper side", np.where(labs.position < 0, "Normal, lower side", "Normal"))
            words.loc[labs.index] = np.where(labs.status == "high", "High", np.where(labs.status == "low", "Low", side))
        return words

    def labs(self) -> pd.DataFrame:
        """Each lab result, the most out of range first. How far off is in half ranges: 1 is the edge."""
        if self._labs is None:
            acting = self.acting
            return pd.DataFrame({"Status": self._status_words().loc[acting.index], "How far off": acting.round(2)})
        labs = self._labs
        out = pd.DataFrame({
            "Result": [f"{_num(v)} {u}" for v, u in zip(labs.value, labs.unit)],
            "Healthy range": [f"{_num(lo)} to {_num(hi)}" for lo, hi in zip(labs.low, labs.high)],
            "Status": self._status_words().loc[labs.index].to_numpy(),
            "How far off": labs.position.round(2).to_numpy(),
            "Changes advice": np.where(labs.deviation != 0, "Yes", "No"),
        }, index=pd.Index(labs.index, name="Biochemical"))
        order = (labs.status == "normal").astype(int).to_numpy() * 1000 - labs.position.abs().to_numpy()
        return out.iloc[np.argsort(order, kind="stable")]

    def _score_100(self, scores: pd.Series) -> pd.Series:
        total = self.tuning.food_mix + self.tuning.nutrient_mix + self.tuning.nutriscore_mix
        if self.tuning.normalize == "minmax":
            return (100 * scores / total).clip(0, 100).round(1)
        return (100 * (scores - scores.min()) / (np.ptp(scores) + 1e-9)).round(1)

    def _reasons(self, food: str, helps: bool, count: int = 2) -> str:
        if helps and self._vectors["guarded"][self.reference.foods.index.get_loc(food)]:
            return "low general food quality, so lab results can only move it down"
        points = self._points[food]
        points = points[points > 0.05] if helps else points[points < -0.05]
        if not len(points):
            return "general food quality" if helps else "low general food quality"
        deviation = pd.Series(self._vectors["deviation"], index=self.reference.biochemicals.index)
        if self._labs is not None:
            off_range = set(self._labs.index[self._labs.status != "normal"])
        else:
            off_range = set(deviation.index[deviation.abs() > 1])
        # Results outside the healthy range are named first, then the biggest point changes.
        top = sorted(points.index, key=lambda b: (b not in off_range, -abs(points[b])))[:count]
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
        if self._nutrients is None:
            ref = self.reference
            need = self._vectors["need"]
            scale = np.abs(need).max() or 1.0
            need_100 = np.round(100 * need / scale) + 0.0
            advice = np.where(need_100 >= 1, "Eat more", np.where(need_100 <= -1, "Eat less", "No change"))
            # The richest foods per nutrient. Sources to eat more of come from the better half of this
            # person's list; sources to cut down can be any food.
            names = self.table.index.to_numpy()
            amounts = self._food_nutrients[[ref.foods.index.get_loc(f) for f in names]]
            half = max(10, len(names) // 2)
            best_all = np.argsort(-amounts, axis=0, kind="stable")[:3]
            best_half = np.argsort(-amounts[:half], axis=0, kind="stable")[:3]
            found = []
            for j in range(len(need)):
                rows = best_half[:, j] if advice[j] == "Eat more" else best_all[:, j]
                found.append(", ".join(names[r] for r in rows if amounts[r, j] > 0))
            out = pd.DataFrame({"Advice": advice, "Need": need_100.astype(int),
                                "Group": ref.nutrients["category"].to_numpy(), "Found in": found},
                               index=pd.Index(ref.nutrients.index, name="Nutrient"))
            self._nutrients = out.sort_values("Need", ascending=False, kind="stable")
        return self._nutrients.copy()

    def explain(self, food: str) -> pd.DataFrame:
        """Each test that acts, and how many score points it gives this food (out of 100).
        Positive pushes the food up the list, negative pushes it down."""
        if food not in self._points.columns:
            raise KeyError(f"unknown food {food!r}")
        bios = self.reference.biochemicals.index
        f = self.reference.foods.index.get_loc(food)
        effect = self._food_coef[:, f] + self._nutrient_coef @ self._food_nutrients[f]
        direct, through = self._points_direct[:, f], self._points_through[:, f]
        out = pd.DataFrame({
            "Your result": self._status_words().to_numpy(),
            "This food": np.where(effect > 0, "raises it", np.where(effect < 0, "lowers it", "no effect")),
            "Direct": direct.round(2) + 0.0,  # + 0.0 turns -0.0 into 0.0
            "Through nutrients": through.round(2) + 0.0,
            "Points": (direct + through).round(2) + 0.0,
        }, index=pd.Index(bios, name="Biochemical"))
        out["Verdict"] = np.where(out.Points > 0, "Good", np.where(out.Points < 0, "Bad", ""))
        out = out[self._vectors["deviation"] != 0]
        return out.reindex(out.Points.abs().sort_values(ascending=False, kind="stable").index)

    def summary(self, n: int = 10) -> str:
        lines = []
        if self._labs is not None:
            labs = self._labs
            off = labs[labs.status != "normal"]
            lines.append(f"{len(labs)} lab results, {len(off)} outside the healthy range.")
            for status, word in (("high", "High"), ("low", "Low")):
                rows = off[off.status == status]
                rows = rows.reindex(rows.position.abs().sort_values(ascending=False).index)
                if len(rows):
                    items = [f"{b} {_num(r.value)} {r.unit} (healthy {_num(r.low)} to {_num(r.high)})" for b, r in rows.iterrows()]
                    lines.append(f"{word}: " + "; ".join(items))
            quiet = off[off.deviation == 0]
            if len(quiet):
                lines.append("No diet change for: " + ", ".join(f"{b} ({r.status})" for b, r in quiet.iterrows())
                             + ", because only the other side of the range is a concern.")
            lines.append("")
        if not len(self.acting):
            lines += ["None of these results calls for a diet change, so the list follows general food quality.", ""]
        nutrients = self.nutrients()
        more = nutrients[nutrients.Advice == "Eat more"].head(6).index
        less = nutrients[nutrients.Advice == "Eat less"].sort_values("Need", kind="stable").head(6).index
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
            "need_before_guard": pd.Series(v["need_before_guard"], index=nutrients, name="need_before_guard"),
            "need": pd.Series(v["need"], index=nutrients, name="need"),
            "guarded_foods": pd.Series(v["guarded"], index=foods, name="guarded_foods"),
            "through_nutrients": pd.Series(v["through"], index=foods, name="through_nutrients"),
            "parts": self._parts.copy(),
            "score": pd.Series(v["score"], index=foods, name="score"),
            "pull": self._pull.copy(),
            "points": self._points.copy(),
            "base_points": self._base.copy(),
        }
