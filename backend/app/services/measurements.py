"""Read a product's size out of the catalog's `measurements` list.

IKEA lists measurements as [{"name": "Width", "value": "5' 3 \\""}, ...] in the
language and unit system of the country page. The old cleaning step looked the
name up in a short alias list and dropped every product with no `width` and no
`height`, which removed whole families: rugs and curtains (width x length, no
height), lamps and mirrors (diameter), sofas ("Height including back cushions"),
beds ("Headboard height"), adjustable desks ("Max. height"), every Arabic page.

Here each dimension is read through a chain: the plain name first, then any
name that contains the word ("Tischbreite", "Height including back cushions"),
then qualified names ("Seat width", "Max. height") as a last resort. A product
with no readable size is kept; `has_dims` says so.
"""

import ast
import re
from typing import Any, Iterable

# name stems per dimension, lower case. Substring match, so "tischbreite" and
# "gesamthöhe" are caught by "breite" and "höhe".
_STEMS: dict[str, tuple[str, ...]] = {
    "width": (
        "width", "largeur", "ancho", "breite", "larghezza", "largura", "breedte", "bredd",
        "širina", "szerokość", "šířka", "šírka", "bredde", "lățime", "lăţime", "zabalera", "amplària", "lebar", "leveys", "szélesség", "العرض", "ความกว้าง", "רוחב",
    ),
    "height": (
        "height", "hauteur", "haut.", "alto", "altura", "höhe", "hoogte", "höjd", "altezza",
        "visina", "višina", "wysokość", "výška", "højde", "înălțime", "înălţime", "altuera", "alçada", "tinggi", "korkeus", "magasság", "الإرتفاع", "الارتفاع", "ความสูง", "גובה",
    ),
    "depth": (
        "depth", "profondeur", "prof.", "profondità", "fondo", "tiefe", "profundidade", "diepte",
        "djup", "dubina", "globina", "głębokość", "hloubka", "hĺbka", "dybde", "adâncime", "sakonera", "fondària", "dalam", "syvyys", "mélység", "العمق", "ความลึก", "עומק",
    ),
    "length": (
        "length", "longueur", "longitud", "länge", "lunghezza", "comprimento", "lengte", "längd",
        "dužina", "dolžina", "długość", "délka", "dĺžka", "lungime", "luzera", "llargada", "panjang", "pituus", "largo", "hossz", "الطول", "ความยาว", "אורך",
    ),
    "diameter": (
        "diameter", "diamètre", "diámetro", "durchmesser", "diametro", "diâmetro", "promjer",
        "średnica", "premer", "průměr", "diametru", "diàmetre", "diametroa", "القطر", "قطر", "เส้นผ่านศูนย์กลาง", "קוטר",
    ),
}

# Names that are exactly the plain word. These are the product's own size.
_EXACT: dict[str, frozenset[str]] = {
    dim: frozenset(stems) for dim, stems in _STEMS.items()
}

# A name that mentions a part ("seat width", "mattress length") or a limit
# ("max. height") rather than the whole product. Used only when nothing better exists.
_QUALIFIER = re.compile(
    r"\b(seat|assise|asiento|assento|sedile|zitting|sitz\w*|armrest|accoudoir|armlehne|mattress|"
    r"matelas|matratze|materasso|colch\w+|matras|headboard|footboard|kopfteil|backrest|"
    r"dossier|rückenlehne|respaldo|encosto|schienale|bed|chair|chaise|stuhl\w*|sedia|silla|"
    r"cadeira|stoel|max|min|load|charge|cord|cable|shade|base|package|free|under|folded|"
    r"clearance|area|thickness|weight|pile)\b\.?",
    re.IGNORECASE,
)

_UNICODE_FRACTIONS = {
    "¼": " 1/4", "½": " 1/2", "¾": " 3/4", "⅓": " 1/3", "⅔": " 2/3",
    "⅛": " 1/8", "⅜": " 3/8", "⅝": " 5/8", "⅞": " 7/8",
}
_ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩٫", "0123456789.")

# unit word -> centimetres
_UNIT_CM = {
    "cm": 1.0, "سم": 1.0, 'ס"מ': 1.0, "mm": 0.1, 'מ"מ': 0.1, "ملم": 0.1, "m": 100.0, "م": 100.0,
    "in": 2.54, "inch": 2.54, "inches": 2.54, '"': 2.54, "″": 2.54, "''": 2.54,
    "ft": 30.48, "feet": 30.48, "foot": 30.48, "'": 30.48, "′": 30.48,
}

_NUM = r"\d+(?:\s+\d+/\d+|/\d+|[.,]\d+)?"
_FEET_INCHES = re.compile(rf"^({_NUM})\s*(?:'|′|ft|feet|foot)\s*(?:({_NUM})\s*(?:\"|″|''|in|inch|inches)?)?\s*$")
_VALUE = re.compile(rf"^({_NUM})\s*([^\d\s].*)?$")


def _number(text: str) -> float:
    text = text.strip().replace(",", ".")
    total = 0.0
    for part in text.split():
        if "/" in part:
            numerator, denominator = part.split("/")
            total += float(numerator) / float(denominator)
        else:
            total += float(part)
    return total


def parse_length_cm(value: Any) -> float | None:
    """A measurement value in centimetres, or None when it is not a length
    ("13 W", "77.50 sq feet", "1.86 kg") or can't be read. A bare number is cm."""
    if value is None:
        return None
    text = str(value).strip().translate(_ARABIC_DIGITS)
    for glyph, replacement in _UNICODE_FRACTIONS.items():
        text = text.replace(glyph, replacement)
    # "75 cm (29 1/2 ")" -> the metric part before the bracket
    text = re.split(r"[(\[]", text)[0].strip()
    if not text:
        return None

    feet = _FEET_INCHES.match(text)
    if feet:
        inches = _number(feet.group(2)) if feet.group(2) else 0.0
        return round(_number(feet.group(1)) * 30.48 + inches * 2.54, 2)

    match = _VALUE.match(text)
    if not match:
        return None
    unit = (match.group(2) or "cm").strip().lower().rstrip(".")
    factor = _UNIT_CM.get(unit)
    if factor is None:
        return None
    try:
        return round(_number(match.group(1)) * factor, 2)
    except (ValueError, ZeroDivisionError):
        return None


def _entries(measurements: Any) -> list[tuple[str, Any]]:
    if isinstance(measurements, str):
        try:
            measurements = ast.literal_eval(measurements)
        except (ValueError, SyntaxError):
            return []
    out: list[tuple[str, Any]] = []
    for item in measurements or []:
        if isinstance(item, dict) and item.get("name"):
            out.append((str(item["name"]).strip().lower(), item.get("value")))
    return out


# Measurements of a cable or of a rug's pile are not the size of the product:
# a lamp's 160 cm "Cord length" is not its depth.
_NOT_THE_PRODUCT = re.compile(r"\b(cord|cable|wire|pile|chain|cordon|kabel|cavo|cabo|cable)\b|الكابل|السلك", re.IGNORECASE)


def _candidates(entries: Iterable[tuple[str, Any]], dim: str) -> list[tuple[tuple[int, int, int], str, Any]]:
    stems = _STEMS[dim]
    found = []
    for name, value in entries:
        if not any(stem in name for stem in stems) or _NOT_THE_PRODUCT.search(name):
            continue
        if name in _EXACT[dim]:
            tier = 0
        elif not _QUALIFIER.search(name):
            tier = 1
        else:
            tier = 2
        # within a tier: a limit ("min.") last, then the shorter, plainer name
        found.append(((tier, int(bool(re.search(r"\bmin\b|mind\.", name))), len(name)), name, value))
    return sorted(found, key=lambda c: c[0])


def _first_length(candidates: list[tuple[tuple[int, int, int], str, Any]]) -> float | None:
    for _, _, value in candidates:
        parsed = parse_length_cm(value)
        if parsed is not None and parsed > 0:
            return parsed
    return None


def extract_dims(measurements: Any) -> dict[str, float | None]:
    """width/height/depth/length/diameter in cm, each None when the product
    doesn't give it. `depth` falls back to `length` (a rug's second side), and
    `width` to `diameter` for round things, so a size can be compared by width."""
    entries = _entries(measurements)
    dims = {dim: _first_length(_candidates(entries, dim)) for dim in _STEMS}
    return {
        "width_cm": dims["width"] if dims["width"] is not None else dims["diameter"],
        "height_cm": dims["height"],
        "depth_cm": dims["depth"] if dims["depth"] is not None else dims["length"],
        "length_cm": dims["length"],
        "diameter_cm": dims["diameter"],
    }
