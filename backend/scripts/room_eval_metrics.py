"""The pure parts of scripts/room_eval.py: which products can be a test case, how
rooms are planned, how detections are tied to the answer, and the metrics.

No network and no models here, so all of it is covered by tests/test_room_eval.py.
"""

import random
import re
from collections import Counter, defaultdict
from statistics import median
from typing import Any, Callable, Iterable

# The first type of each kind is its anchor: every room of that kind has one.
ROOM_TEMPLATES: dict[str, tuple[str, ...]] = {
    "living room": (
        "sofa", "armchair", "coffee_table", "tv_bench", "side_table", "lamp", "bookshelf", "ottoman",
        "rug", "plant_pot", "artwork", "curtain", "mirror", "wall_shelf", "cabinet",
    ),
    "bedroom": (
        "bed", "nightstand", "wardrobe", "dresser", "lamp", "mirror", "bench", "rug", "curtain",
        "artwork", "ottoman", "plant_pot",
    ),
    "dining room": (
        "dining_table", "chair", "cabinet", "lamp", "stool", "rug", "artwork", "mirror", "plant_pot", "curtain",
    ),
    "home office": (
        "desk", "chair", "bookshelf", "lamp", "wall_shelf", "cabinet", "stool", "plant_pot", "artwork", "curtain", "rug",
    ),
}

ROOM_SIZE = 4
# A type with fewer usable products than this can't give a meaningful result.
MIN_PRODUCTS_PER_TYPE = 20

_ENGLISH_COUNTRIES = ("gb", "us")
# Things the catalog files under a type but that aren't one on their own: a rod
# isn't a curtain, a ledge isn't a picture, and a table sold with 2 chairs has
# no single right answer. Outdoor pieces don't belong in the rooms generated.
_EXCLUDED_TITLE = re.compile(
    r"\b(rod|rail|ring|rings|hook|hooks|ledge|frame for|insert|greenhouse|outdoor|children'?s?|cover|legs?)\b"
    r"|\band \d+ (chairs?|stools?)\b",
    re.IGNORECASE,
)


# What the title must say, in English, and what it must not, for a product to
# really be of its type. The catalog's type alone isn't enough to be an answer:
# it files a mattress and a crib under bed, a table's underframe under
# dining_table and a work bench under bench, and some "us" listings are in Spanish.
_TYPE_TITLES: dict[str, tuple[str, str | None]] = {
    "sofa": (r"\b(sofa|sectional|loveseat|couch)\b", None),
    "armchair": (r"\b(armchair|wing chair)\b", None),
    "chair": (r"\bchair\b", r"\b(armchair|high ?chair)\b"),
    "stool": (r"\bstool\b", r"\bstep\b"),
    "bench": (r"\bbench\b", r"\b(work|tv|coat rack)\b"),
    "ottoman": (r"\b(ottoman|pouffe?|footstool)\b", None),
    "coffee_table": (r"\bcoffee table\b", None),
    "dining_table": (r"\btable\b", r"\b(underframe|top|side|coffee|bedside|desk|wall|changing|conference)\b"),
    "side_table": (r"\bside table\b", None),
    "nightstand": (r"\b(nightstand|bedside table)\b", None),
    "desk": (r"\bdesk\b", r"\b(lamp|screen|pad|organi[sz]er|shelf)\b"),
    "bed": (r"\bbed\b", r"\b(mattress|under ?bed|drawers?|crib|canopy|tent|slatted|guard|tray|side table)\b"),
    "dresser": (r"\b(chest of|\d-drawer chest|dresser)\b", None),
    "wardrobe": (r"\bwardrobe\b", r"\b(door|interior|rail|organi[sz]er)\b"),
    "bookshelf": (r"\b(bookcase|shelving unit|shelf unit)\b", r"\bwall\b"),
    "cabinet": (r"\b(cabinet|sideboard)\b", r"\b(kitchen|bathroom|mirror|door|wash)\b"),
    "tv_bench": (r"\btv (unit|bench|stand|storage)\b", None),
    "lamp": (r"\blamp\b", r"\b(shade|cord|base)\b"),
    "rug": (r"\brug\b", None),
    "curtain": (r"\b(curtains?|blinds?)\b", None),
    "mirror": (r"\bmirror\b", r"\b(door|cabinet|wardrobe|lamp)\b"),
    "wall_shelf": (r"\bwall shelf\b", None),
    "plant_pot": (r"\b(plant pot|plant stand|planter|artificial potted plant)\b", None),
    "artwork": (r"\b(picture|poster|painting|canvas)\b", r"\bframe\b"),
}


def is_usable_product(row: dict[str, Any]) -> bool:
    """Whether a product's representative row can be a test case: an English
    listing with a picture, typed from its own title, a single product, and
    named in that title as what its type says it is."""
    title = row.get("main_title")
    if not row.get("main_image") or not title:
        return False
    if (row.get("country_code") or "").lower() not in _ENGLISH_COUNTRIES:
        return False
    if row.get("type_source") != "title" or _EXCLUDED_TITLE.search(title):
        return False
    rule = _TYPE_TITLES.get(row.get("furniture_type") or "")
    if rule is None:
        return False
    required, excluded = rule
    return bool(re.search(required, title, re.IGNORECASE)) and not (excluded and re.search(excluded, title, re.IGNORECASE))


def plan_rooms(
    count: int,
    available: Iterable[str],
    rng: random.Random,
    size: int = ROOM_SIZE,
    templates: dict[str, tuple[str, ...]] = ROOM_TEMPLATES,
    kinds: list[str] | None = None,
) -> list[dict[str, Any]]:
    """`count` rooms, each with `size` different types, spreading the types as
    evenly as the templates allow. Types are distinct within a room, so one
    type names one product. `kinds` fixes each room's kind (the app's room
    type of each photo); otherwise the kinds take turns."""
    available = set(available)
    fillable = [k for k, types in templates.items() if types[0] in available and len(set(types) & available) >= size]
    if not fillable:
        raise ValueError("No room template can be filled from the available types.")
    if kinds is not None:
        unfillable = sorted(set(kinds) - set(fillable))
        if unfillable:
            raise ValueError(f"These room kinds can't be filled from the available types: {unfillable}")
    used: Counter[str] = Counter()
    rooms = []
    for index in range(count):
        kind = kinds[index] if kinds is not None else fillable[index % len(fillable)]
        anchor = templates[kind][0]
        picked = [anchor]
        used[anchor] += 1
        others = [t for t in templates[kind][1:] if t in available]
        rng.shuffle(others)  # breaks ties between equally used types
        for type_name in sorted(others, key=lambda t: used[t])[: size - 1]:
            picked.append(type_name)
            used[type_name] += 1
        rooms.append({"id": f"room{index + 1:03d}", "kind": kind, "types": picked})
    return rooms


def detections_of(group: str, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The detections the reviewer tied to a product. The detector's label
    isn't used for this: it often names a neighbouring type (a dining table
    called a desk), and that piece is still the product in the picture."""
    return [d for d in detections if d.get("group") == group]


# How deep in the results equivalent listings are looked for, and how alike two
# catalog pictures must be (cosine of their index vectors) to be one product.
# Only the same picture: in the pilot, pairs from 0.985 to 0.999 were still
# different products (2 vs 3 drawers, other legs, another bookcase size).
EQUIVALENT_DEPTH = 20
EQUIVALENT_SIMILARITY = 0.999


def canonical_sku(sku: str | None) -> str:
    """"S994.433.05" and "99443305" are the same article number."""
    return re.sub(r"\D", "", sku or "")


def equivalent_rank(results: list[dict[str, Any]], group: str, equivalence: list[dict[str, Any]] | None) -> int | None:
    """Like rank_of, but a result counts when it is the answer under another
    listing: the same SKU, or a picture the index can't tell from the answer's."""
    for position, result in enumerate(results, start=1):
        if result["group"] == group:
            return position
        if equivalence and position <= len(equivalence):
            alike = equivalence[position - 1]
            if alike.get("same_sku") or (alike.get("similarity") or 0) >= EQUIVALENT_SIMILARITY:
                return position
    return None


def rank_of(results: list[dict[str, Any]], group: str) -> int | None:
    """1-based position of the product among the search results, or None."""
    for position, result in enumerate(results, start=1):
        if result["group"] == group:
            return position
    return None


def first_word(title: str | None) -> str:
    """The product line a title names: IKEA titles start with it ("MALM bed frame")."""
    words = (title or "").split()
    return words[0].casefold() if words else ""


def same_line_rank(results: list[dict[str, Any]], title: str | None) -> int | None:
    """Like rank_of, but any colour or size of the same product line counts."""
    line = first_word(title)
    if not line:
        return None
    for position, result in enumerate(results, start=1):
        if first_word(result.get("title")) == line:
            return position
    return None


def detection_outcome(
    detection: dict[str, Any], product: dict[str, Any], floor: float, results_key: str = "results"
) -> dict[str, Any]:
    """What one search found for one detection of a product. `results_key`
    picks the search: with the detector's label, or with the right type."""
    results = detection.get(results_key) or []
    rank = rank_of(results, product["group"])
    answer_score = results[rank - 1]["score"] if rank else None
    equivalent = equivalent_rank(results, product["group"], detection.get(results_key.replace("results", "equivalence")))
    equivalent_score = results[equivalent - 1]["score"] if equivalent else None
    return {
        "verdict": detection.get("verdict"),
        "label": detection.get("label"),
        "label_ok": detection.get("type") == product["type"],
        "rank": rank,
        "rank_with_floor": rank if answer_score is not None and answer_score >= floor else None,
        "equivalent_rank": equivalent,
        "equivalent_rank_with_floor": equivalent if equivalent_score is not None and equivalent_score >= floor else None,
        "line_rank": same_line_rank(results, product.get("title")),
        "top_score": results[0]["score"] if results else None,
        "answer_score": answer_score,
        "filter_level": detection.get(results_key.replace("results", "filter_level")),
    }


def item_outcomes(rooms: list[dict[str, Any]], floor: float, results_key: str = "results") -> list[dict[str, Any]]:
    """One record per product placed in a room. A product detected twice counts
    at its best rank, because the user sees a card for each detection."""
    items = []
    for room in rooms:
        for product in room["products"]:
            outcomes = [
                detection_outcome(d, product, floor, results_key)
                for d in detections_of(product["group"], room.get("detections", []))
            ]
            items.append(
                {
                    "room": room["id"],
                    "kind": room["kind"],
                    "type": product["type"],
                    "group": product["group"],
                    "title": product.get("title"),
                    "ref_kind": product.get("ref_kind"),
                    "detections": outcomes,
                }
            )
    return items


def _best(values: Iterable[int | None]) -> int | None:
    present = [v for v in values if v is not None]
    return min(present) if present else None


def faithful_only(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Items reduced to the detections reviewed as the same product. Items
    without one drop out: nothing in them can be blamed on the search."""
    kept = []
    for item in items:
        detections = [d for d in item["detections"] if d["verdict"] == "same"]
        if detections:
            kept.append({**item, "detections": detections})
    return kept


def detected_only(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [item for item in items if item["detections"]]


def summarize(items: list[dict[str, Any]], rank_key: str = "rank") -> dict[str, Any]:
    """Hit@1, Hit@5 and MRR of the best rank per item; an undetected item is a miss."""
    total = len(items)
    if not total:
        return {"n": 0, "hit1": None, "hit5": None, "mrr": None, "not_in_pool": None, "line_hit5": None}
    ranks = [_best(d[rank_key] for d in item["detections"]) for item in items]
    line_ranks = [_best(d["line_rank"] for d in item["detections"]) for item in items]
    return {
        "n": total,
        "hit1": sum(1 for r in ranks if r == 1) / total,
        "hit5": sum(1 for r in ranks if r is not None and r <= 5) / total,
        "mrr": sum(1 / r for r in ranks if r is not None) / total,
        "not_in_pool": sum(1 for r in ranks if r is None) / total,
        "line_hit5": sum(1 for r in line_ranks if r is not None and r <= 5) / total,
    }


def bootstrap_ci(
    items: list[dict[str, Any]],
    statistic: Callable[[list[dict[str, Any]]], float | None],
    rng: random.Random,
    draws: int = 2000,
    level: float = 0.95,
) -> tuple[float, float] | None:
    """Percentile interval from resampling whole rooms: pieces of one room share
    a picture, a lighting and a detector run, so they aren't independent."""
    by_room: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_room[item["room"]].append(item)
    rooms = list(by_room.values())
    if not rooms:
        return None
    values = []
    for _ in range(draws):
        sample = [item for room in rng.choices(rooms, k=len(rooms)) for item in room]
        value = statistic(sample)
        if value is not None:
            values.append(value)
    if not values:
        return None
    values.sort()
    low = int((1 - level) / 2 * len(values))
    return values[low], values[min(len(values) - 1, len(values) - 1 - low)]


# ── Confidence thresholds ───────────────────────────────────────────────────

def calibration_points(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """(top score, was the top result right) for each search that has a right
    answer: a detection reviewed as the product, with results. Right means the
    answer or one of its equivalent listings, which is what the user would buy."""
    points = []
    for item in items:
        for detection in item["detections"]:
            if detection["verdict"] == "same" and detection["top_score"] is not None:
                points.append(
                    {
                        "room": item["room"],
                        "type": item["type"],
                        "score": detection["top_score"],
                        "correct": detection["equivalent_rank"] == 1,
                    }
                )
    return points


def split_rooms(room_ids: Iterable[str], rng: random.Random, fit_share: float = 0.7) -> tuple[set[str], set[str]]:
    """Rooms to choose thresholds on and rooms to check them on. Split by room
    so pieces of one picture never sit on both sides."""
    rooms = sorted(set(room_ids))
    rng.shuffle(rooms)
    cut = round(len(rooms) * fit_share)
    return set(rooms[:cut]), set(rooms[cut:])


def threshold_grid(low: float = 0.50, high: float = 0.95, step: float = 0.01) -> list[float]:
    return [round(low + i * step, 2) for i in range(round((high - low) / step) + 1)]


def precision_curve(points: list[dict[str, Any]], grid: Iterable[float]) -> list[dict[str, Any]]:
    """At each threshold: how many searches reach it and how often their top result is right."""
    curve = []
    for threshold in grid:
        kept = [p for p in points if p["score"] >= threshold]
        curve.append(
            {
                "threshold": threshold,
                "n": len(kept),
                "precision": sum(1 for p in kept if p["correct"]) / len(kept) if kept else None,
            }
        )
    return curve


def choose_threshold(curve: list[dict[str, Any]], target: float, min_n: int = 10) -> float | None:
    """The lowest threshold whose precision reaches `target` on at least `min_n`
    searches. Lowest, so the label covers as many matches as it honestly can."""
    for row in curve:
        if row["n"] >= min_n and row["precision"] is not None and row["precision"] >= target:
            return row["threshold"]
    return None


def label_precision(points: list[dict[str, Any]], high: float | None, medium: float | None) -> dict[str, dict[str, Any]]:
    """How often the top result is right within each confidence label."""

    def label(score: float) -> str:
        if high is not None and score >= high:
            return "high"
        if medium is not None and score >= medium:
            return "medium"
        return "low"

    table: dict[str, dict[str, Any]] = {}
    for name in ("high", "medium", "low"):
        kept = [p for p in points if label(p["score"]) == name]
        table[name] = {
            "n": len(kept),
            "precision": sum(1 for p in kept if p["correct"]) / len(kept) if kept else None,
        }
    return table


def scores_by_type(points: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Median top score of right and wrong searches per type, to see whether one
    threshold can serve every type."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for point in points:
        grouped[point["type"]].append(point)
    table = {}
    for type_name, group in sorted(grouped.items()):
        right = [p["score"] for p in group if p["correct"]]
        wrong = [p["score"] for p in group if not p["correct"]]
        table[type_name] = {
            "n": len(group),
            "precision": len(right) / len(group),
            "median_right": median(right) if right else None,
            "median_wrong": median(wrong) if wrong else None,
        }
    return table


# ── Grading by eye ──────────────────────────────────────────────────────────

# How well the result a person picked as the best one matches the piece.
JUDGE_GRADES = ("exact", "similar")


def human_judgments(rooms: list[dict[str, Any]], judgments: dict[str, Any], grid: Iterable[float], top: int = 5) -> dict[str, Any]:
    """What a person picked, blind, as the best of the app's top results for
    each piece, beside the app's own order and the answer."""
    searches = []
    for room in rooms:
        for detection in room.get("detections", []):
            answer = judgments.get(detection.get("id"))
            results = (detection.get("results") or [])[:top]
            if not answer or not results:
                continue
            groups = [r["group"] for r in results]
            best = answer.get("best")
            searches.append(
                {
                    "score": results[0]["score"],
                    "best_rank": groups.index(best) + 1 if best in groups else None,
                    "grade": answer.get("grade"),
                    "has_answer": detection.get("group") is not None,
                    "best_is_answer": best is not None and best == detection.get("group"),
                }
            )
    n = len(searches)
    if not n:
        return {"n": 0}

    def rates(group):
        if not group:
            return {"n": 0, "useful": None, "exact": None}
        return {
            "n": len(group),
            "useful": sum(1 for x in group if x["best_rank"] == 1) / len(group),
            "exact": sum(1 for x in group if x["best_rank"] == 1 and x["grade"] == "exact") / len(group),
        }

    def label(score):
        return "high" if score >= 0.75 else "medium" if score >= 0.65 else "low"

    with_answer = [x for x in searches if x["has_answer"]]
    return {
        "n": n,
        "hit_useful": sum(1 for x in searches if x["best_rank"] is not None) / n,
        "hit_exact": sum(1 for x in searches if x["grade"] == "exact") / n,
        "best_is_top1": sum(1 for x in searches if x["best_rank"] == 1) / n,
        "best_is_answer": sum(1 for x in with_answer if x["best_is_answer"]) / len(with_answer) if with_answer else None,
        "best_rank": {str(k): v for k, v in sorted(Counter(x["best_rank"] for x in searches if x["best_rank"]).items())},
        "none": sum(1 for x in searches if x["best_rank"] is None),
        "labels": {name: rates([x for x in searches if label(x["score"]) == name]) for name in ("high", "medium", "low")},
        "curve": [{"threshold": t, **rates([x for x in searches if x["score"] >= t])} for t in grid],
    }


# ── Products that suit a room's style ───────────────────────────────────────

def rank_by_similarity(vectors: dict[str, list[float]], target: list[float]) -> list[str]:
    """Ids by cosine similarity to `target`, most alike first."""

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
        return dot / norm if norm else 0.0

    return sorted(vectors, key=lambda key: cosine(vectors[key], target), reverse=True)


def one_per_kind(products: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The products in order, keeping the first of each type: a room's answers
    are one product per kind."""
    seen: set[str] = set()
    kept = []
    for product in products:
        if product["type"] not in seen:
            seen.add(product["type"])
            kept.append(product)
    return kept
