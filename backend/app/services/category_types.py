"""A furniture type from IKEA's own category, for products whose title says none.

A title is written in the shop's language, and the regexes in furniture_types.py
and catalog_types.py read a few of them: an Arabic "مكتبة" (bookcase) got no type.
The category is better: `category_id` is the same number on every country's
site, so the English path of an id ("Bookcases & shelving units > Bookcases")
types the products of that id in every language.

`type_from_category` reads that English path. Rules are tried in order and the
first match wins, so specific names come before the general words they contain
("Sofa & armchairs covers" is a textile, not a sofa). Every product gets a type
and there is no "other": a category no rule reads is a "household" item
(FALLBACK_TYPE). Types that are not furniture a room is furnished
with (cookware, tableware, toys...) are named too, so each product has a label
and the catalog can be scaled to other uses later.

Titles stay first: this only fills products whose title gave no type
(see `fill_missing_types`).
"""

import re
from collections import Counter
from typing import Iterable

# (type, regex over the lower-cased path "a > b > c", root "Products" removed).
_RULES: tuple[tuple[str, str], ...] = (
    # ── Parts and covers, before the furniture words they contain ─────────
    ("tv_bench", r"tv benches|tv & media|media combinations|bestå tv"),
    ("textile", r"\bcovers\b|cushions & headrests|shower curtains|bath textiles|"
                r"kitchen textiles|bedding|bedspreads|comforters|duvet|pillows|towels|throws|blankets|"
                r"fabrics & sewing|napkins"),
    ("part", r"\blegs\b|bed slats|underframes|seat shells|knobs & handles|hinges|shelf brackets|"
             r"shelf boards|table tops|desk dividers|room & desk dividers|inner drawers|"
             r"doors & (drawers|drawer fronts)|drawer fronts|plinths|cover panels|deco strips|"
             r"fittings|inside organisers|interior|\bparts\b|top panels|frames for |frames & legs|"
             r"\baccessories for |worktop accessories|inserts?\b|sink accessories|"
             r"\b(sliding|hinged|glass) doors\b|doors with hinges|sliding door (frames|combinations)|"
             r"maximera|knivshult|kitchen drawers|"
             r"curtain (rods|poles|wires)|finials|tools & fittings|dining table tops"),
    # ── Lighting ───────────────────────────────────────────────────────────
    # Small movable or hanging pieces that no room type fits: named, not left as "other".
    ("cart", r"trolley|trolleys|\bcarts?\b|utility cart|storage carts|kitchen islands & trolleys"),
    ("rack", r"racks & stands|hat racks|shoe, coat|coat racks|drying racks|wine racks|dish racks|"
             r"clothes racks?|shoe racks?|hanging clothes organisers|bottle wine"),
    ("room_divider", r"room dividers|privacy screens|garden sails & privacy"),
    ("kitchen_cabinet", r"\bmetod\b|\bsektion\b|\benhet kitchen|knoxhult|sunnersta|kitchenettes?|"
                        r"kitchen systems|kitchen unit systems|kitchen islands|modular kitchens|"
                        r"outdoor kitchens?|kabinet & peralatan dapur|cocina|kitchen cabinet|"
                        r"kitchen wall storage|kitchen storage|kitchen shelves|pantry|cuisine"),
    ("appliance", r"ceiling fans|air purifiers|extractor|fridges|freezers|hobs|cooktop|ovens?\b|"
                  r"ranges\b|dishwashers|washing machines|microwave|ironing|appliances"),
    ("lamp", r"ceiling lamps|pendant lights|pendant lamps|desk lamps|floor lamps|table lamps|"
             r"wall lamps|chandeliers|rechargeable lamps|smart lamps|led lamps|lamps & light fixtures|"
             r"bathroom (ceiling|wall) lights|outdoor (floor lamps|pendant|table lights|wall lights)|"
             r"wall spotlights|led lanterns|children's lighting|^lamps$"),
    ("lighting_accessory", r"light bulbs|led strip|integrated lighting|lighting for|lamp shades|bases & cords|"
                           r"smart led|smart lighting|remote light|track lighting|cabinet lighting|"
                           r"string lights|pathway lights|table light decorations|decorative lighting|"
                           r"bathroom cabinet lighting|bathroom lighting|outdoor lighting|candles|led candles"),
    # ── Kitchen and bathroom ──────────────────────────────────────────────
    ("kitchen_fixture", r"kitchen taps|kitchen sinks|sinks & taps|kitchen worktops|kitchen countertops|"
                        r"splashbacks|backsplashes|bathroom sinks|bathroom taps|showers\b|"
                        r"kitchen mixer|worktops|countertops"),
    ("bathroom", r"mirror cabinets|storage & mirrors"),
    ("mirror", r"mirrors?\b"),
    ("bathroom", r"bathroom|laundry combination|vanity|vanities|toilet|shower caddies|"
                 r"muebles para baño|laundry baskets|bath\b"),
    # ── Seating and sleeping ───────────────────────────────────────────────
    ("sofa", r"sofa beds|\bsofas\b|chaise longues|outdoor sofas|chair beds|modular outdoor|"
             r"modular sofas|corner sofas|outdoor chaise|lounge sofa"),
    ("armchair", r"armchairs?|recliner|lounge chairs|kerusi berlengan|poäng|outdoor patio furniture|"
                 r"garden lounge"),
    ("ottoman", r"ottomans|footstools|pouffes"),
    ("stool", r"^stools|\bstools\b|bar stools|step stools|bathroom stools"),
    ("bench", r"benches|bancas"),
    ("chair", r"\bchairs?\b|dining furniture > kitchen chairs|desk chairs|gaming chairs|conference chairs"),
    ("mattress", r"mattress|cot mattress|patjat|runkopatjat"),
    ("bed", r"\bbeds\b|headboards|bed and mattress sets|cots|bassinets|bunk|loft beds|"
            r"mattress bases|nursery furniture sets|bedroom furniture sets|guest beds|day beds"),
    # ── Tables, desks, storage ─────────────────────────────────────────────
    ("coffee_table", r"coffee tables|nesting tables|garden coffee"),
    ("side_table", r"side tables|console tables|accent tables|garden coffee & side|bedside|"
                   r"dressing tables"),
    ("nightstand", r"bedside tables"),
    ("desk", r"\bdesks?\b|computer desks|gaming desks|gaming furniture|meeting & conference|"
             r"conference tables|office desks|laptop stands|idåsen|mittzon|trotten|relatera|"
             r"table bar combinations|desk organisers"),
    ("dining_table", r"dining (tables?|sets)|dining table|\btables\b|bar tables|bar furniture|"
                     r"seater dining|person dining|garden dining|garden bistro|outdoor dining|"
                     r"children's tables|picnic|wall-mounted dining|extendable"),
    ("dresser", r"chest of drawers|dressers|drawer units|kids dressers"),
    ("wardrobe", r"wardrobes?|children's wardrobes|clothes storage"),
    ("bookshelf", r"bookcases|shelving units|shelving furniture|cube storage|storage shelves|"
                  r"complete wall shelves|trofast|småstad|display shelves|picture ledges"),
    ("wall_shelf", r"wall shelves|hooks & wall organisation|wall organisers|wall storage|"
                   r"bathroom shelves|towel rails"),
    ("cabinet", r"cabinets|cupboards|sideboards|buffets|display cabinets|storage units|"
                r"storage combinations|storage solution systems|storage units & cabinets|"
                r"wasmachinekast|wasmachinerek|muebles para lavadero|"
                r"pax|bestå|boaxel|aurdal|ivar|platsa|eket|skytta|bror|shoe|garden storage|"
                r"children's furniture|room dividers|cat furniture|small storage|trolleys"),
    ("storage_box", r"under bed storage|boxes|baskets|storage boxes|food storage|jars|paper & media|file|magazine|"
                    r"drying racks|moving supplies|laundry|bags\b"),
    # ── Decor ──────────────────────────────────────────────────────────────
    ("rug", r"\brugs\b|outdoor rugs|outdoor flooring|doormats|runners"),
    ("curtain", r"\bcurtains\b|\bblinds\b|smart & electric blinds|window films|roller blinds"),
    ("vase", r"\bvases\b|vases & bowls"),
    ("plant_pot", r"plants|flower pots|planters|plant pots|outdoor pots|maceteros|pasu & tumbuhan|"
                  r"winter pots|flowers|garlands|artificial trees|christmas trees"),
    ("artwork", r"picture|photo frames|posters|wall decor|pictures, prints|wall art"),
    ("decoration", r"decorations|christmas|candle holders|lanterns|ornaments|decor|clocks|"
                   r"dried flowers|droogbloemen|kunstkr|girlanden|wreath"),
    # ── Not furniture, but named ───────────────────────────────────────────
    ("kitchenware", r"cookware|bakeware|cooking|baking|knives|chopping|cutting boards|colanders|"
                    r"coffee & tea|dishwashing|kitchen cabinet & drawer|kitchen utensils|serveware|"
                    r"kitchen textiles|pot holders|bbq|grills|outdoor kitchen"),
    ("tableware", r"flatware|cutlery|drinkware|tableware|dinnerware|plates|glasses|mugs|"
                  r"coffee mugs|water bottle|vacuum flasks|napkin|serving"),
    ("baby", r"baby|highchairs|changing table|safety products"),
    ("toy", r"toys|soft toys|play|children's kitchenware|kids|pretend|wooden toys"),
    ("electronics", r"cables & chargers|chargers|charging|mobile, tablet|batteries|gaming accessories|speakers"),
    ("outdoor_equipment", r"garden sun shades|parasols|gazebos|pergolas|sails|garden sails|recreation|"
                          r"outdoor accessories|garden"),
    ("household", r"cleaning|waste|bins|recycling|hooks|bags|laundry|bathroom accessories|soap|"
                  r"safety|dogs|pet|desk organisers|noticeboards|tools|treatment|paint|clocks"),
)
_COMPILED = tuple((name, re.compile(pattern)) for name, pattern in _RULES)

# Names the English site uses for the first level, so the path below it is what
# is read. "Products" is the site root and says nothing.
_ROOT = "products"

# What a product is when no rule reads its category: a household item. There is
# no "other" label: every product says what it is. "household" also names the
# things the rules put there on purpose (tools, hooks, cleaning, safety...).
FALLBACK_TYPE = "household"
# The word the title reader (catalog_types.classify) uses for "a small item I
# recognise but can't type". It is replaced by the category's type, never kept.
_TITLE_OTHER = "other"


def match_category(path: Iterable[str | None]) -> str | None:
    """The type a category path names ("Bookcases & shelving units > Bookcases"
    -> "bookshelf"), or None when no rule reads the path (an unknown language)."""
    names = [str(n).strip() for n in path if n and str(n).strip()]
    if names and names[0].lower() == _ROOT:
        names = names[1:]
    if not names:
        return None
    text = " > ".join(names).lower()
    # The leaf is the most specific part: try it alone first, then the whole path.
    leaf = names[-1].lower()
    for scope in (leaf, text):
        for name, pattern in _COMPILED:
            if pattern.search(scope):
                return name
    return None


def type_from_category(path: Iterable[str | None]) -> str:
    """Like match_category, but never None: a path no rule reads is FALLBACK_TYPE."""
    return match_category(path) or FALLBACK_TYPE


def types_by_category_id(
    paths_by_country: dict[str, dict[str, list[str]]],
    english_sites: Iterable[str] = (),
) -> dict[str, str]:
    """One type per category_id from {category_id: {country: path}}.

    IKEA names the same category differently on each site ("Pendant lighting" on
    one, "Pendants" on another, Malay on a third), so no single country's path is
    reliable. Every country's path is read and the type most of them agree on
    wins; paths no rule reads do not vote. A tie goes to the type the
    `english_sites` give first. An id none of the paths can read is FALLBACK_TYPE."""
    english = tuple(english_sites)
    result: dict[str, str] = {}
    for category_id, by_country in paths_by_country.items():
        votes: Counter[str] = Counter()
        first_english: dict[str, int] = {}
        for country, path in by_country.items():
            voted = match_category(path)
            if voted is None:
                continue
            votes[voted] += 1
            rank = english.index(country) if country in english else len(english)
            first_english[voted] = min(first_english.get(voted, rank), rank)
        if votes:
            result[category_id] = max(votes, key=lambda t: (votes[t], -first_english[t]))
        else:
            result[category_id] = FALLBACK_TYPE
    return result


def one_type_per_product(types: dict[str, str], rows: Iterable[dict]) -> dict[str, str]:
    """Give every row of a product (same `group_key`: the product on every country's
    site) the same type, the one most of its rows have. The category fills rows one
    at a time and countries name a category differently, so without this the same
    product could be a cabinet in one country and a bookshelf in another. A tie
    goes to the type of the product's first row."""
    members: dict[str, list[str]] = {}
    for row in rows:
        members.setdefault(str(row.get("group_key")), []).append(str(row.get("unique_id")))
    result = dict(types)
    for uids in members.values():
        counts = Counter(types[uid] for uid in uids)
        top = max(counts.values())
        winner = next(types[uid] for uid in uids if counts[types[uid]] == top)
        for uid in uids:
            result[uid] = winner
    return result


def fill_missing_types(
    types: dict[str, str],
    rows: Iterable[dict],
    type_by_category_id: dict[str, str],
) -> dict[str, str]:
    """`types` (title-derived, unique_id -> type) completed from the category:
    every row without a type gets the type of its `category_id` (see
    types_by_category_id), or FALLBACK_TYPE when the id is unknown. Rows that
    already have a type are kept, except the title reader's "other": that only
    says the title matched a small-item word, so the category may still name it.
    The result never holds "other"."""
    filled = dict(types)
    for row in rows:
        uid = str(row.get("unique_id"))
        if filled.get(uid, _TITLE_OTHER) != _TITLE_OTHER:
            continue
        filled[uid] = type_by_category_id.get(str(row.get("category_id")), FALLBACK_TYPE)
    return filled
