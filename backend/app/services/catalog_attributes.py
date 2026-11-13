"""Material and colour tags for catalog rows (from clean_data.ipynb, corrected).

Changes from the notebook, each one found by comparing the output with the raw
data:
- melamine/paper foil/plastic edging no longer count as Plastic: they are the
  finish of a wood-based panel, and made Plastic (23,976 rows) outnumber Wood.
- short words that are also common in other languages ("ek", "fer", "pin",
  "ash", "bok", "ull", "lin", "abs") are only read from the materials text,
  never from the title/description fallback.
- colour words are picked by where they appear in the title, not by the order
  of the colour table, and wood names ("oak", "walnut") only name a colour when
  no real colour word is present.
"""

import re
from typing import Any, Iterable

SCORE_THRESHOLD = 1   # a keyword must be found at least this often
MIN_SCORE_GAP = 0.25  # a secondary material needs >= 25% of the main one's score

WOOD_KEYWORDS = [
    'wood', 'oak', 'pine', 'birch', 'beech', 'walnut', 'ash', 'veneer',
    'particleboard', 'fiberboard', 'fibreboard', 'plywood', 'mdf',
    'bamboo', 'rattan', 'melamine', 'hardboard', 'layer-glued', 'solid wood',
    'bois', 'chêne', 'pin', 'bouleau', 'hêtre', 'panneau de particules', 'contreplaqué', 'fibres de bois',
    'holz', 'kiefer', 'buche', 'eiche', 'birke', 'spanplatte', 'massivholz', 'furnier',
    'madera', 'roble', 'abedul', 'haya', 'contrachapado', 'tablero', 'madeira', 'pinho',
    'legno', 'betulla', 'rovere', 'faggio', 'massello', 'compensato', 'truciolato',
    'drvo', 'iverica', 'lesonit',
    'trä', 'ek', 'björk', 'bok', 'furu',
    'خشب', 'صنوبر', 'بتولا', 'بلوط', 'زان', 'خشب مضغوط', 'لوح مضغوط',
    'עץ', 'לוח סיבים',
    'ไม้', 'ไม้อัด', 'ไม้จริง',
]
METAL_KEYWORDS = [
    'steel', 'aluminum', 'aluminium', 'iron', 'chrome', 'chromium',
    'galvanized', 'powder coating', 'powder-coated', 'stainless',
    'brass', 'copper', 'zinc', 'nickel', 'epoxy coating', 'metal',
    'acier', 'fer', 'chromé', 'galvanisé', 'inoxydable', 'métal',
    'stahl', 'eisen', 'verzinkt', 'chrom', 'edelstahl', 'metall',
    'acero', 'aluminio', 'hierro', 'inoxidable', 'acero inoxidable', 'galvanizado',
    'acciaio', 'alluminio', 'ferro', 'inossidabile', 'metallo',
    'čelik', 'aluminij', 'nehrđajući', 'željezo',
    'stål', 'järn', 'galvaniserad',
    'فولاذ', 'حديد', 'ألومنيوم', 'معدن', 'فولاذ مقاوم للصدأ', 'كروم',
    'מתכת', 'פלדה', 'אלומיניום', 'ברזל',
    'เหล็ก', 'โลหะ', 'อลูมิเนียม', 'สแตนเลส',
]
FABRIC_KEYWORDS = [
    'fabric', 'cotton', 'polyester', 'linen', 'velvet', 'wool',
    'leather', 'foam', 'polyurethane', 'upholstery', 'textile',
    'nylon', 'viscose', 'silk', 'suede', 'canvas', 'woven', 'knitted',
    'coton', 'lin', 'velours', 'laine', 'cuir', 'mousse', 'polyuréthane', 'tissu',
    'baumwolle', 'leinen', 'wolle', 'leder', 'schaum', 'polyurethan', 'stoff',
    'algodón', 'lino', 'terciopelo', 'cuero', 'espuma', 'poliuretano',
    'poliester', 'tecido', 'algodão', 'tejido',
    'cotone', 'velluto', 'pelle', 'schiuma', 'tessuto',
    'pamuk', 'vuna', 'koža', 'pjena', 'tkanina',
    'bomull', 'ull', 'läder', 'skum', 'tyg',
    'قطن', 'بوليستر', 'نسيج', 'صوف', 'جلد', 'رغوة', 'قماش',
    'בד', 'כותנה', 'עור',
    'ผ้า', 'ฝ้าย', 'โพลีเอสเตอร์', 'หนัง',
]
GLASS_KEYWORDS = [
    'glass', 'mirror', 'tempered glass', 'safety glass',
    'verre', 'miroir', 'verre trempé',
    'glas', 'spiegel', 'gehärtetes glas',
    'vidrio', 'cristal', 'espejo', 'templado', 'vidro',
    'vetro', 'specchio', 'temperato',
    'staklo', 'ogledalo',
    'spegel', 'härdat glas',
    'زجاج', 'مرآة',
    'זכוכית', 'מראה',
    'กระจก', 'แก้ว',
]
PLASTIC_KEYWORDS = [
    'plastic', 'polypropylene', 'acrylic', 'abs', 'pvc',
    'polyamide', 'polystyrene', 'resin',
    'polypropylène', 'acrylique', 'résine', 'plastique',
    'polypropylen', 'acryl', 'kunststoff', 'plastik',
    'polipropileno', 'acrílico', 'resina', 'plástico',
    'polipropilene', 'acrilico', 'plastica',
    'polipropilen', 'plastična', 'plastika',
    'akryl', 'plast',
    'بلاستيك', 'بولي بروبيلين', 'أكريليك',
    'פלסטיק',
    'พลาสติก', 'โพลีโพรพิลีน',
]

MATERIAL_CATEGORIES: tuple[tuple[str, list[str]], ...] = (
    ('Wood', WOOD_KEYWORDS),
    ('Metal', METAL_KEYWORDS),
    ('Fabric', FABRIC_KEYWORDS),
    ('Plastic', PLASTIC_KEYWORDS),
    ('Glass', GLASS_KEYWORDS),
)

# Words that are also ordinary words or abbreviations in another language of
# the catalog; trusted only in the materials text.
AMBIGUOUS_KEYWORDS = frozenset({'ek', 'fer', 'pin', 'ash', 'bok', 'ull', 'lin', 'abs'})

# Surface finishes and edges of wood-based panels. They name plastic or paper,
# but the product is wood, so they are not read as a material.
_FINISH_PHRASES = re.compile(
    r'plastic edging|plastic edge|edge band\w*|melamine foil|paper foil|melamine|acrylic paint|'
    r'foil|printed (?:paper|pattern)|bordure plastique|kunststoffkante',
    re.IGNORECASE,
)

_THAI = re.compile(r'[฀-๿]')
_PATTERNS: dict[str, re.Pattern[str]] = {}


def _pattern(keyword: str) -> re.Pattern[str]:
    if keyword not in _PATTERNS:
        _PATTERNS[keyword] = re.compile(r'\b' + re.escape(keyword) + r'\b')
    return _PATTERNS[keyword]


def count_keyword_matches(text: str, keywords: Iterable[str], trust_ambiguous: bool = True) -> int:
    """Occurrences of the keywords in text. Whole-word matches, so "iron" is not
    found in "environment"; Thai has no spaces, so it is a plain substring."""
    count = 0
    for kw in keywords:
        if not trust_ambiguous and kw in AMBIGUOUS_KEYWORDS:
            continue
        if _THAI.search(kw):
            count += text.count(kw)
        else:
            count += len(_pattern(kw).findall(text))
    return count


def materials_text(materials_and_care: Any, materials_description: Any) -> str:
    """The product-specific material lines, then the generic description."""
    parts: list[str] = []
    if isinstance(materials_and_care, list):
        parts.extend(
            str(item.get('value') or '')
            for item in materials_and_care
            if isinstance(item, dict) and item.get('type') == 'materials'
        )
    if isinstance(materials_description, str):
        parts.append(materials_description)
    return re.sub(r'\bundefined\b', '', ' '.join(parts).lower()).strip()


def material_tags(mat_text: str, title: str = '', description: str = '') -> str:
    """"Wood|Fabric": every material that scores >= 25% of the best one, best
    first, or "Other". Materials text first; the title and description are only
    read when it names nothing."""
    mat_text = _FINISH_PHRASES.sub(' ', mat_text)
    scores = {name: count_keyword_matches(mat_text, kws) for name, kws in MATERIAL_CATEGORIES}
    if not any(v >= SCORE_THRESHOLD for v in scores.values()):
        fallback = _FINISH_PHRASES.sub(' ', f'{mat_text} {description} {title}'.lower())
        scores = {
            name: count_keyword_matches(fallback, kws, trust_ambiguous=False)
            for name, kws in MATERIAL_CATEGORIES
        }
    if not any(v >= SCORE_THRESHOLD for v in scores.values()):
        return 'Other'
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    needed = max(SCORE_THRESHOLD, ranked[0][1] * MIN_SCORE_GAP)
    return '|'.join(name for name, score in ranked if score >= needed)


# ── Colour ──────────────────────────────────────────────────────────────────

COLOR_MAP = {
    'white':  ['white', 'blanc', 'bianco', 'blanco', 'weiß', 'weiss', 'wit', 'vit', 'hvid',
               'valkoinen', 'off-white', 'off white', 'eggshell', 'cream', 'ivory', 'cassé', 'latte'],
    'black':  ['black', 'noir', 'nero', 'negro', 'schwarz', 'zwart', 'svart', 'sort', 'musta',
               'siyah', 'czarny', 'negru', 'antracite', 'anthracite', 'anthrazit'],
    'gray':   ['gray', 'grey', 'gris', 'grau', 'grigio', 'grijs', 'grå', 'harmaa', 'szary', 'gri',
               'graphite', 'graphit', 'silver', 'argent', 'argento', 'silber', 'cinz'],
    'beige':  ['beige', 'bej', 'ecru', 'sand', 'sandy', 'natural', 'natur', 'natura', 'naturel', 'naturale'],
    'brown':  ['brown', 'marron', 'marrone', 'braun', 'brun', 'bruin', 'ruskea', 'maro'],
    'blue':   ['blue', 'bleu', 'blu', 'blau', 'blauw', 'blå', 'sininen', 'niebieski', 'albastru',
               'mavi', 'navy', 'marine', 'cobalt', 'teal', 'turquoise', 'azure', 'indigo', 'azul'],
    'green':  ['green', 'vert', 'verde', 'grün', 'groen', 'grøn', 'vihreä', 'zielony',
               'olive', 'khaki', 'sage', 'mint', 'forest', 'emerald'],
    'red':    ['red', 'rouge', 'rosso', 'rojo', 'rot', 'rood', 'röd', 'rød', 'punainen',
               'czerwony', 'burgundy', 'bordeaux', 'wine', 'cherry', 'rust'],
    'pink':   ['pink', 'rose', 'rosa', 'blush', 'salmon', 'coral', 'magenta', 'fuchsia',
               'lilac', 'lavender', 'mauve'],
    'yellow': ['yellow', 'jaune', 'giallo', 'amarillo', 'gelb', 'geel', 'gul', 'keltainen',
               'mustard', 'gold', 'golden', 'amber', 'lemon'],
    'purple': ['purple', 'violet', 'viola', 'violett', 'paars', 'lilla', 'fioletowy',
               'plum', 'aubergine'],
    'orange': ['orange', 'arancione', 'naranja', 'oranje', 'appelsin', 'copper'],
}
# A wood tone names a colour only when the title has no colour word.
WOOD_TONES = ['oak', 'chêne', 'roble', 'rovere', 'eiche', 'walnut', 'noyer', 'noce', 'pine', 'birch',
              'beech', 'teak', 'wenge', 'bouleau', 'hêtre', 'walnut']

_SIZE_IN_TITLE = re.compile(r'\b\d[\d\s/x\xd7.]*(?:cm|mm|m|"|\'|inch)\b')


def _compile(words: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r'\b(?:' + '|'.join(re.escape(w) for w in sorted(set(words), key=len, reverse=True)) + r')\b')


_COLOR_PATTERNS = {name: _compile(words) for name, words in COLOR_MAP.items()}
_WOOD_PATTERN = _compile(WOOD_TONES)


def _earliest_color(text: str) -> str | None:
    best: tuple[int, str] | None = None
    for name, pattern in _COLOR_PATTERNS.items():
        match = pattern.search(text)
        if match and (best is None or match.start() < best[0]):
            best = (match.start(), name)
    return best[1] if best else None


def extract_color_from_title(title: Any) -> str | None:
    """"BILLY bookcase, white, 80x202 cm" -> "white". The colour segment (after
    the first comma) is read first, then the whole title; the colour mentioned
    first wins. None when the title names none."""
    if not isinstance(title, str) or not title:
        return None
    text = _SIZE_IN_TITLE.sub('', title.lower())
    parts = text.split(',')
    segment = ' '.join(parts[1:]) if len(parts) > 1 else text
    for scope in (segment, text):
        found = _earliest_color(scope)
        if found:
            return found
    for scope in (segment, text):
        if _WOOD_PATTERN.search(scope):
            return 'brown'
    return None
