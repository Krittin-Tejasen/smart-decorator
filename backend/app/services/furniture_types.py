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
            continue
        for row in members:
            result[str(row["unique_id"])] = group_type
    return result
