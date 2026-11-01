"""One shared vocabulary of furniture types for product matching.

Both sides of a match must agree on it:
- catalog side: every product in the Pinecone index gets a `furniture_type`
  metadata field, derived from its title (scripts/add_furniture_type.py and
  scripts/ingest_products.py), so a query can be restricted to one type;
- query side: the label segmentation gives a detected item ("armchair",
  "floor lamp", or Gemini's free text like "floating planter") is mapped to the
  same type before querying (label_to_type).

The first eight + five types and their patterns are the ones measured in
pinecone-evaluation (sample_catalog_expanded.py / sample_catalog_newtypes.py),
kept unchanged, including their exclusion words, so those results still apply.
The rest were added to cover the rest of segmentation.py's vocabulary.
"""

import re
from collections import Counter, defaultdict
from typing import Any, Iterable

# Accessories and spare parts whose titles name a piece of furniture
# ("sofa cover", "PAX wardrobe door") but aren't one.
_EVALUATED_EXCLUDE = (
    "cover", "housse", "fodera", "funda", "cushion", "coussin", "cuscino",
    "mirror", "shade", "door", "hinge", "organiser", "organizer",
    "accessories", "accessory", "handle", "leg for", "spare part",
    "foot-rest", "footrest", "wall org", "cushion pad",
)
# Looser list for the types added later: "mirror", "door", "cushion"... are
# themselves products there (a mirror, a cabinet with doors).
_ACCESSORY_EXCLUDE = (
    "cover", "housse", "fodera", "funda", "hinge", "handle", "leg for",
    "spare part", "organiser", "organizer", "accessories", "accessory",
)

# METOD kitchen tall cabinets are titled "armoire ..." in French.
_KITCHEN_WORDS = (
    "réfrigérateur", "refrigerateur", "congél", "fridge", "freezer", "built-in", "sink",
    "évier", "micro", "oven", "four ", "appliance", "kühl", "lave", "cleaning",
)
_DRAWER_WORDS = (
    "drawer", "tiroir", "schublade", "cajón", "cajon", "gaveta", "cassetto", "tray",
    "plateau", "bandeja", "tabuleiro", "ausziehboden", "pull-out",
)
# Words that make a "mirror" title a door/part of a wardrobe or a cabinet.
_MIRROR_PARTS = (
    "cabinet", "wardrobe", "armoire", "armadio", "guardaroba", "armario", "meuble",
    "door", "porte", "ptes", "ante", "puerta", "tür", "combination", "combinazione",
    "combinaison", "kombination", "cabina",
)

# (type, title patterns, exclusion words). Order matters: the first type whose
# pattern matches wins, so specific names come before the words they contain
# ("tv bench" before "bench", "bedside table" before "table").
_TYPES: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    # Measured in pinecone-evaluation (13 types).
    ("tv_bench", (r"\btv (bench|unit|stand)\b", r"meuble tv", r"porta ?tv"), _EVALUATED_EXCLUDE),
    ("coffee_table", (r"\bcoffee table\b", r"table basse", r"\btavolino\b"), _EVALUATED_EXCLUDE),
    ("nightstand", (r"\bnightstand\b", r"\bbedside table\b", r"table de chevet"), _EVALUATED_EXCLUDE),
    ("armchair", (r"\barmchair\b", r"\bfauteuil\b", r"\bpoltrona\b"), _EVALUATED_EXCLUDE),
    ("stool", (r"\bstool\b", r"\btabouret\b", r"\bsgabello\b"), _EVALUATED_EXCLUDE),
    ("sofa", (r"\bsofa\b", r"\bcouch\b", r"\bcanap[eé]\b", r"\bdivano\b"), _EVALUATED_EXCLUDE),
    ("bed", (r"\bbed frame\b", r"\blit\b", r"\bletto\b"), _EVALUATED_EXCLUDE),
    ("dining_table", (r"\bdining table\b", r"table [aà] manger", r"tavolo da pranzo"), _EVALUATED_EXCLUDE),
    ("chair", (r"\bchair\b", r"\bchaise\b", r"\bsedia\b", r"\bsilla\b"), _EVALUATED_EXCLUDE),
    (
        "bookshelf",
        (r"\bbookcase\b", r"\bshelving unit\b", r"biblioth[eè]que", r"\blibreria\b", r"\bestanter[ií]a\b"),
        _EVALUATED_EXCLUDE,
    ),
    ("lamp", (r"\blamp\b", r"\blampe\b", r"\blampada\b"), _EVALUATED_EXCLUDE),
    # "guardaroba" is Italian for wardrobe (347 PAX rows); without it they fell
    # through to "mirror" via "specchio" (found by a real query, 2026-10-13).
    ("wardrobe", (r"\bwardrobe\b", r"\barmoire\b", r"\barmadio\b", r"\bguardaroba\b"), _EVALUATED_EXCLUDE + _KITCHEN_WORDS),
    ("desk", (r"\bdesk\b", r"\bbureau\b", r"\bscrivania\b", r"\bescritorio\b"), _EVALUATED_EXCLUDE),
    # Added for the rest of the segmentation vocabulary (not yet evaluated).
    (
        "side_table",
        (r"\bside table\b", r"\bend table\b", r"\bconsole table\b", r"table d'appoint", r"\bconsole\b"),
        _ACCESSORY_EXCLUDE,
    ),
    ("dresser", (r"\bchest of( \d+)? drawers\b", r"\bdresser\b", r"\bcommode\b"), _ACCESSORY_EXCLUDE),
    ("cabinet", (r"\bsideboard\b", r"\bbuffet\b", r"\bcabinet\b"), _ACCESSORY_EXCLUDE + ("mirror",)),
    ("ottoman", (r"\bfootstool\b", r"\bottoman\b", r"\bpouffe?\b"), _ACCESSORY_EXCLUDE),
    # "banc TV" (French) and "TV-Bank" (German) are TV benches.
    ("bench", (r"\bbench\b", r"\bbanc\b", r"\bpanca\b"), _ACCESSORY_EXCLUDE + ("banc tv", "tv-bank")),
    # The catalog has no rugs at all: every "tapis" in it is a drawer mat (the
    # KOMPLEMENT pull-out trays), hence the exclusions. Kept in the vocabulary so
    # a "rug" label is understood and answered with "no match".
    ("rug", (r"\brug\b", r"\bcarpet\b", r"\btapis\b", r"\btappeto\b"), _ACCESSORY_EXCLUDE + _DRAWER_WORDS),
    ("curtain", (r"\bcurtains?\b", r"\brideaux?\b", r"\btende\b", r"\bblinds?\b"), _ACCESSORY_EXCLUDE),
    # Only when the mirror is the product: "mirror door", "meuble avec miroir" and
    # PAX wardrobes with a mirror finish name it as a part or a finish.
    ("mirror", (r"\bmirror\b", r"\bmiroir\b", r"\bspecchio\b"), _ACCESSORY_EXCLUDE + _MIRROR_PARTS),
    ("plant_pot", (r"\bplant pot\b", r"\bplanter\b", r"\bplant stand\b", r"\bartificial (potted )?plant\b"), _ACCESSORY_EXCLUDE),
    ("vase", (r"\bvase\b",), _ACCESSORY_EXCLUDE),
    ("artwork", (r"\bpicture\b", r"\bposter\b", r"\bpainting\b", r"\bcanvas\b"), _ACCESSORY_EXCLUDE),
    # A shelf fixed to the wall, as a detected "floating shelf" would be.
    ("wall_shelf", (r"\bwall shelf\b", r"\bfloating shelf\b", r"\bwall-mounted shelf\b"), _ACCESSORY_EXCLUDE),
)

FURNITURE_TYPES: tuple[str, ...] = tuple(name for name, _, _ in _TYPES)

_COMPILED = tuple(
    (name, tuple(re.compile(p) for p in patterns), excludes) for name, patterns, excludes in _TYPES
)


def classify_title(text: str | None) -> str | None:
    """The furniture type a catalog title (or any free-text label) names, or
    None when it names none of them or is an accessory/spare part."""
    text_l = (text or "").lower()
    if not text_l:
        return None
    for name, patterns, excludes in _COMPILED:
        if any(p.search(text_l) for p in patterns):
            if any(ex in text_l for ex in excludes):
                # An accessory of this type ("sofa cover") isn't any other
                # type either.
                return None
            return name
    return None


# segmentation.py vocabulary -> furniture type. Labels outside it (Gemini's
# fallback returns free text) go through classify_title instead.
_LABEL_TO_TYPE: dict[str, str] = {
    **dict.fromkeys(["sofa", "couch"], "sofa"),
    "armchair": "armchair",
    **dict.fromkeys(["chair", "dining chair", "office chair"], "chair"),
    "stool": "stool",
    "bench": "bench",
    "ottoman": "ottoman",
    "coffee table": "coffee_table",
    "dining table": "dining_table",
    **dict.fromkeys(["side table", "end table", "console table"], "side_table"),
    "desk": "desk",
    **dict.fromkeys(["bed", "headboard"], "bed"),
    **dict.fromkeys(["nightstand", "bedside table"], "nightstand"),
    "dresser": "dresser",
    "wardrobe": "wardrobe",
    **dict.fromkeys(["bookshelf", "bookcase"], "bookshelf"),
    **dict.fromkeys(["cabinet", "sideboard"], "cabinet"),
    **dict.fromkeys(["tv stand", "media console"], "tv_bench"),
    **dict.fromkeys(["floor lamp", "table lamp", "pendant lamp", "chandelier", "lamp"], "lamp"),
    **dict.fromkeys(["rug", "carpet", "area rug"], "rug"),
    **dict.fromkeys(["curtain", "curtains", "blinds"], "curtain"),
    **dict.fromkeys(["plant pot", "potted plant", "plant", "planter"], "plant_pot"),
    **dict.fromkeys(["artwork", "painting", "picture", "framed picture"], "artwork"),
    "mirror": "mirror",
    "vase": "vase",
    **dict.fromkeys(["floating shelf", "wall shelf", "shelf"], "wall_shelf"),
}


def label_to_type(label: str | None) -> str | None:
    """Furniture type for a segmentation label, or None when it doesn't name
    one we can filter the catalog by (e.g. "decoration")."""
    key = " ".join((label or "").lower().replace("_", " ").split())
    if key in _LABEL_TO_TYPE:
        return _LABEL_TO_TYPE[key]
    return classify_title(key)


# ── Subtypes: a floor lamp is not a pendant ─────────────────────────────────
# `lamp` alone merged floor, table, pendant, ceiling and wall lamps, so a floor
# lamp in a room could be matched with a chandelier. The subtype narrows a
# query first; the type is still the fallback (product_matching.build_filter_ladder).
# Order matters: "pendant" before "ceiling" before "wall" before "floor" before "table".
_LAMP_SUBTYPES: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        ("pendant", r"\bpendant\b|chandelier|hängeleuchte|pendelleuchte|suspension|suspendu|sospensione|lámpara colgante|colgante|candeeiro suspenso|hanglamp|kroonluchter|\blustre\b|taklampa|pendente"),
        ("ceiling", r"\bceiling\b|plafon|deckenleuchte|deckenlampe|soffitto|\btecho\b|plafondlamp|plafoniera"),
        ("wall", r"\bwall (lamp|light)\b|wandleuchte|wandlampe|applique|lámpara de pared|lampada da parete|\bwall\b|parete"),
        ("floor", r"\bfloor\b|stehleuchte|stehlampe|lampadaire|lámpara de pie|\bde pie\b|lampada da terra|candeeiro de pé|vloerlamp|golvlampa|reading lamp"),
        ("table", r"\btable lamp\b|\bdesk lamp\b|\bwork lamp\b|tischleuchte|tischlampe|lampe de table|lampe de bureau|lámpara de mesa|lampada da tavolo|lampada da scrivania|tafellamp|bordslampa|candeeiro de mesa|\btable\b"),
    )
)
_LABEL_TO_SUBTYPE: dict[str, str] = {
    "floor lamp": "floor",
    "table lamp": "table",
    "pendant lamp": "pendant",
    "chandelier": "pendant",
}


def lamp_subtype(title: str | None) -> str | None:
    """floor / table / pendant / ceiling / wall for a lamp title, or None."""
    text = (title or "").lower()
    for name, pattern in _LAMP_SUBTYPES:
        if pattern.search(text):
            return name
    return None


def label_to_subtype(label: str | None) -> str | None:
    """The subtype a segmentation label names ("floor lamp" -> "floor"); None
    for labels that name only a type ("lamp", "sofa")."""
    key = " ".join((label or "").lower().replace("_", " ").split())
    return _LABEL_TO_SUBTYPE.get(key)


# Types close enough that, when a label's own type has nothing, they are the
# next thing to search. Used only when the type returned nothing: never to pad
# a result that exists (a lamp matching drawer fronts is the case this guards).
RELATED_TYPES: dict[str, tuple[str, ...]] = {
    "tv_bench": ("cabinet",),            # a BESTÅ sideboard is often the TV unit
    "cabinet": ("dresser", "bookshelf"),
    "dresser": ("cabinet",),
    "side_table": ("coffee_table", "nightstand"),
    "coffee_table": ("side_table",),
    "nightstand": ("side_table", "dresser"),
}


# ── Grouping catalog rows that show the same product ────────────────────────

# IKEA image URLs embed an image id ("malm-bed-frame-high-white__0749130_pe745499_s5.jpg"
# -> "0749130_pe745499") that stays the same across countries, languages and
# SKUs of one product. pinecone-evaluation/build_groups.py, kept identical.
_IMAGE_ID_RE = re.compile(r"__(\d+_[a-z]+\d+)")


def image_group_key(main_image: str | None, sku: str | None) -> str:
    match = _IMAGE_ID_RE.search(main_image or "")
    if match:
        return f"img:{match.group(1)}"
    return f"sku:{re.sub(r'[^0-9]', '', sku or '')}"


def types_by_unique_id(rows: Iterable[dict[str, Any]]) -> dict[str, str]:
    """`furniture_type` for every catalog row whose image group has one.

    The type is the majority vote over the titles of every row in the row's
    image group, so a French or Spanish title the patterns miss still gets the
    type of the same product's English rows. Rows whose group gets no vote
    are left out (they get no `furniture_type` field at all)."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("unique_id") is None:
            continue
        groups[image_group_key(row.get("main_image"), row.get("sku"))].append(row)

    from app.services.catalog_types import OVERRIDING_TYPES, classify_group

    result: dict[str, str] = {}
    for members in groups.values():
        votes = Counter(classify_title(row.get("main_title")) for row in members)
        votes.pop(None, None)
        group_type = votes.most_common(1)[0][0] if votes else None
        # The multilingual reader fills groups the regexes could not name, and
        # moves kitchen cabinets, parts, bathroom units and appliances out of
        # types they were mistaken for (cabinet, wardrobe...).
        extra = classify_group([row.get("main_title") for row in members])
        if extra is not None and (group_type is None or extra in OVERRIDING_TYPES):
            group_type = extra
        if group_type is None:
            group_type = table_type_from_size(members)
        if group_type is None:
            continue
        for row in members:
            result[str(row["unique_id"])] = group_type
    return result


def subtypes_by_unique_id(rows: Iterable[dict[str, Any]], types: dict[str, str]) -> dict[str, str]:
    """`furniture_subtype` for the rows whose group is a lamp: the majority
    vote of lamp_subtype over the titles of the image group."""
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if types.get(str(row.get("unique_id"))) == "lamp":
            groups[image_group_key(row.get("main_image"), row.get("sku"))].append(row)
    result: dict[str, str] = {}
    for members in groups.values():
        votes = Counter(lamp_subtype(row.get("main_title")) for row in members)
        votes.pop(None, None)
        if not votes:
            continue
        subtype = votes.most_common(1)[0][0]
        for row in members:
            result[str(row["unique_id"])] = subtype
    return result


_BARE_TABLE = re.compile(r"^(?:[A-ZÅÄÖÆØÜÉ0-9/ ]{3,}\s+)?(?:table|tables)\b(?!\s+(?:lamp|top|runner|cloth|mat|pad))", re.IGNORECASE)


def table_type_from_size(members: list[dict[str, Any]]) -> str | None:
    """A title that says only "table" ("TONSTAD table, oak veneer") names no
    kind of table. Its size does: low and large is a coffee table, low and
    small a side table, tall and wide a dining table. Returns None when the
    title isn't a bare "table" or the size is unknown."""
    if not any(_BARE_TABLE.match((row.get("main_title") or "").strip()) for row in members):
        return None
    for row in members:
        try:
            height = float(row.get("height_cm"))
            reach = max(float(row.get("width_cm") or 0), float(row.get("depth_cm") or 0))
        except (TypeError, ValueError):
            continue
        if height <= 0 or reach <= 0:
            continue
        if height <= 60:
            return "coffee_table" if reach >= 70 else "side_table"
        return "dining_table" if reach >= 100 else "side_table" if reach < 60 else "desk"
    return None
