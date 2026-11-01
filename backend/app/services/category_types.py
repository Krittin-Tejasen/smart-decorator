"""A furniture type from IKEA's own category, for products whose title says none.

A title is written in the shop's language, and the regexes in furniture_types.py
and catalog_types.py read a few of them: an Arabic "مكتبة" (bookcase) got no type.
The category is better: `category_id` is the same number on every country's
site, so the English path of an id ("Bookcases & shelving units > Bookcases")
types the products of that id in every language.

`type_from_category` reads that English path. Rules are tried in order and the
first match wins, so specific names come before the general words they contain
("Sofa & armchairs covers" is a textile, not a sofa). Every path gets a type:
what no rule names is "other". Types that are not furniture a room is furnished
with (cookware, tableware, toys...) are named too, so each product has a label
and the catalog can be scaled to other uses later.

Titles stay first: this only fills products whose title gave no type
(see `fill_missing_types`).
"""

import re
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


def type_from_category(path: Iterable[str | None]) -> str:
    """The type a category path names ("Bookcases & shelving units > Bookcases"
    -> "bookshelf"). Never None: a path no rule names is "other"."""
    names = [str(n).strip() for n in path if n and str(n).strip()]
    if names and names[0].lower() == _ROOT:
        names = names[1:]
    text = " > ".join(names).lower()
    if not text:
        return "other"
    # The leaf is the most specific part: try it alone first, then the whole path.
    leaf = names[-1].lower()
    for scope in (leaf, text):
        for name, pattern in _COMPILED:
            if pattern.search(scope):
                return name
    return "other"


def resolve_category_paths(
    paths_by_country: dict[str, dict[str, list[str]]],
    english_sites: Iterable[str],
) -> dict[str, list[str]]:
    """One path per category_id from {category_id: {country: path}}: the path from
    an English-language site when there is one, else the first country's path
    that the rules can read (some Gulf and Czech sites show English names too)."""
    english = tuple(english_sites)
    resolved: dict[str, list[str]] = {}
    for category_id, by_country in paths_by_country.items():
        for country in english:
            if country in by_country:
                resolved[category_id] = by_country[country]
                break
        else:
            for country in sorted(by_country):
                if type_from_category(by_country[country]) != "other":
                    resolved[category_id] = by_country[country]
                    break
    return resolved


def fill_missing_types(
    types: dict[str, str],
    rows: Iterable[dict],
    paths_by_category_id: dict[str, list[str]],
) -> dict[str, str]:
    """`types` (title-derived, unique_id -> type) completed from the category:
    every row without a type gets the type of its `category_id`'s English path,
    or "other" when the id has none. Rows that already have a type are kept,
    except "other" from the title reader: that only says the title matched a
    small-item word, so the category may still name the product."""
    filled = dict(types)
    for row in rows:
        uid = str(row.get("unique_id"))
        if filled.get(uid, "other") != "other":
            continue
        path = paths_by_category_id.get(str(row.get("category_id")))
        filled[uid] = type_from_category(path) if path else "other"
    return filled
