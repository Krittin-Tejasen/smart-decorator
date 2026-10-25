"""Extra catalog classifier for products the title regexes in furniture_types.py miss.

The regexes recognise a type only from an English (or a few French/Italian)
word, so a Dutch "kroonluchter", a German "Aufbewahrung" or a Hebrew title got
no type and could never be returned by a typed search. This module reads the
product noun ("base cab f sink", "Aufbewahrung mit Türen") in ~10 languages,
plus the product family, and also names what the regexes ignored: kitchen
cabinets (METOD, SEKTION), cabinet parts (doors, drawer fronts, panels),
bathroom units, appliances and small items.

Measured on the
7,758 image groups that already had a regex type: it agrees on 4,396, moves
3,010 to the new types (mostly kitchen cabinets that were filed as cabinet or
wardrobe), and disagrees on 240 (5.2% of the groups both can name).

Everything here is a heuristic on titles: it never looks at the picture.
"""

import re

KITCHEN_FAM = {"METOD", "MAXIMERA", "SEKTION", "KNOXHULT", "SUNNERSTA", "UTRUSTA"}
BATH_FAM = {"ÄNGSJÖN", "HAVBÄCK", "TÄNNFORSEN", "ORRSJÖN", "GODMORGON", "BROGRUND", "RÄTTVIKEN", "ODENSVIK", "LILLÅNGEN", "SKATSJÖN", "VOXNAN", "SALJEN", "STORKAJEN"}


def R(*p):
    return re.compile("|".join(p), re.I)


PART_START = R(
    r"^(lamp ?shade|shade|vitrinen?tür|vitrine door)",
    r"^(glass |mirror |sliding |pair of |hinged |door |drawer |cover |corner |shaker )?(doors?|portes?|ptes?|tür(en)?|türe|anta|ante|porta|puertas?|portas?|דלת|deur|deuren|fronts?|façade|facade|frontale|frontali|frente|frentes|face|faces)\b",
    r"^(drawer front|face de tiroir|schubladenfront|frontale cassetto|frente gaveta|tiroir|drawer|cassetto|gaveta|schublade|cajón|lade)\b(?! unit| chest)",
    r"^(cover panel|panneau|panel|deckseite|rivestimento|side panel|end panel|plinth|socle|zócalo|sockel|rodapé|zoccolo|cornice|moulding|decor strip|filler|cover strip|finishing)",
    r"^(pull-out|pull out|insert|divider|hinge|handle|knob|legs?\b|armrest|accessor|fitting|organi[sz]er|shelf insert|extra shelf|shelf for|shelf,|mat\b|spare)",
    r"^(wire basket|basket for|cutlery|tray|trays|waste|bin\b)",
    r"^(door with hinges|porte avec|tür mit)",
    r"^(deur|doors? for|ante per)",
    r"^(pair of|par de|coppia di|schiebetürpaar|jeu \d|jeu #|paar|paire)",
    r"^(ladefront|lade|toekick|painel|kantplatte|deckplatte|seitenteil|poggiapiedi$)",
    r"^(accoudoir|bracciolo|braço|armlehne|headrest|appui-tête|kopfstütze|reposacabezas)",
    r"^(mesh basket|clip-on basket|plateau coulissant|plateau|bandeja|tabuleiro|ausziehbar|einsatz)",
    r"^(חזית|מגירה|מגש|חיפוי|מסגרת(?! ארון)|מסד)",
    r"^(geschirrspülerfront|fachada|fronte|frontal)",
    r"^(add-on|corner panel|extension unit|height extension|cover\b|joint|clips?\b|rail with|suspension rail|hook|connector|top and plinth|decorative toekick|bedekkingspaneel|hanging shelf insert|ripiano estraibile|colgador|pannello per|#paneele)",
    r"^#? ?panels?", r"^pannello",
    r"^(driver|transformer|wireless control|extension cord|cable|plug|power supply|remote control)",
    r"^(shower tray|toilet|shower hanger)",
)
KITCHEN_NOUN = R(
    r"\b(base|wall|high|hi|corner|tall)\s*(cab|cb|cabinet)\b", r"^(bc|hc|wc)\b", r"\bbase cb\b", r"\bhi cab\b", r"\bhigh cab\b",
    r"^(élément|élt|el\.?)\s", r"\bunterschr", r"\bhochschr", r"\bwandschr", r"\bpensile\b", r"\bmobile (base|alto|lavello|forno|frigo|microonde|p cottura|angolare|/)",
    r"^mobile/", r"^mob ", r"armário (baixo|alto|de parede|p/|para)", r"^arm (bx|parede|baixo)", r"armoire (four|micro|réfrig|pour réfrig|murale)", r"^rgt bs", r"^ar alt",
    r"^uschr", r"sink cab", r"^(base|high) (cab|cb|cabinet)", r"^wall (cab|cb|cabinet)", r"corner (base|wall)", r"^bs cb", r"gabinete bajo fregadero",
    r"^(kitchen|cuisine|küche|keuken|cucina)$", r"^wall cb",
)
BATH = R(
    r"salle de bain", r"badezimmer", r"badkamer", r"\bbagno\b", r"casa de banho", r"vanity", r"wash-?\s?st(a)?nd", r"wash-basin",
    r"\blavabo", r"\bvasque", r"lavatório", r"waschb", r"mitigeur", r"\btap\b", r"faucet", r"tocador de baño", r"bathroom", r"\bbaño\b", r"lavamanos",
    r"^baño$", r"shower", r"toilet", r"^bath\b",
)
APPLIANCE = R(
    r"\boven\b", r"\bhob\b", r"fridge", r"freezer", r"dishwasher", r"extractor", r"cooker hood", r"washing machine", r"microwave", r"\bhood\b",
    r"refrigerator", r"lave-vaisselle", r"lavavajillas", r"induktionskochfeld", r"induction cooktop", r"cooktop", r"forno", r"^four\b", r"^range with", r"\bcooker\b",
    r"air purifier", r"steam oven", r"fan oven",
)
STORAGE_BOX = R(
    r"^storage box", r"^box\b", r"box with lid", r"^basket", r"^bin\b", r"^crate", r"^boxes", r"^pocket", r"^hanging storage", r"^boîte", r"^korb", r"^kasten",
    r"^mand\b", r"^storage case", r"^container", r"storage basket", r"^moving box", r"^caixa", r"^caja", r"^cesta", r"^scatola",
)
LIGHT_ACC = R(r"led[- ]strip", r"led (strip|lighting|night|decorative|spot)", r"lighting strip", r"night light", r"^bulb", r"led bulb", r"^lighting", r"smart (light|lighting)", r"cabinet lighting", r"worktop lighting", r"éclairage", r"capt")
LAMP = R(r"kroonluchter", r"leuchte", r"\blampe\b", r"\blamp\b", r"pendant", r"\blámpara", r"\blampada", r"chandelier", r"ceiling light", r"spotlight", r"\bluminaire", r"\blustre\b", r"\bplafon")
OTHER_SMALL = R(
    r"^trolley", r"hat and coat", r"changing table", r"tablet stand", r"baby gym", r"speaker", r"hook", r"patère", r"bracket", r"trellis", r"tumble dryer", r"kühlschrank", r"congel", r"jar with", r"roasting", r"pot with", r"^toy", r"christmas tree", r"flower box", r"play kitchen", r"kitchen for children", r"rallonge", r"prolunga", r"sensor", r"organizer for drawer", r"underframe$", r"verbindungskonsole", r"tellerregal", r"peg board", r"display shelf", r"trenner", r"suspension rail", r"susp rail", r"hjälpa", r"closed kitchen unit", r"pegboard", r"charger", r"battery", r"^rack\b", r"^hook", r"^rail", r"^hanger", r"^clothes", r"^shoe", r"^coat", r"^glass$", r"speaker",
    r"alarm clock", r"clock", r"greenhouse", r"gazebo", r"casserole", r"grill", r"lantern", r"hat rack", r"rack with", r"magazine file", r"dish drainer",
    r"cooking", r"^sink\b", r"sink with", r"semi-recessed sink", r"kitchen island", r"utility cart", r"^cart\b", r"desserte", r"kitchen trolley", r"room divider",
    r"paravento", r"paravent", r"raumteiler", r"scheidingswand", r"privacy screen", r"kitchenette", r"cuisine extérieure", r"laundry", r"lavandería", r"lavandaria",
    r"buanderie", r"umbrella", r"colander", r"knife rack", r"spice rack", r"wine (rack|shelf)", r"bottle wine rack", r"drying rack", r"toilet", r"cat house",
    r"children's tent", r"conversation set", r"serving stand", r"carrello", r"carrito", r"carrinho", r"servierwagen", r"cover for", r"housse", r"\bcover\b", r"^hanging",
    r"pêle-mêle", r"photo", r"laptop", r"remote", r"in/outdoor", r"^ikea", r"stainless steel",
)

HEADS = [
    ("tv_bench", R(r"\btv\b.*(bench|unit|stand|storage|bank|combination)", r"banc tv", r"tv-bank", r"tv-möbel", r"tv-kombination", r"media storage", r"porta ?tv", r"meuble tv", r"\btv storage", r"móvel (de )?tv", r"combinazione tv", r"combinaison tv", r"comb arrum tv", r"comb almacenamiento tv", r"tv-meubel")),
    ("wardrobe", R(r"wardrobe", r"schrankkombination", r"roupeiro", r"guardaroba", r"armadio", r"ropero", r"garde-robe", r"\barmoire\b", r"wall-mounted storage frame", r"kleiderschrank", r"kledingkast", r"dressing", r"caisson montage mural", r"add-on (mid|end)-unit", r"ארון", r"kleidst")),
    ("sofa", R(r"sofa-?\s?bed", r"slaapbank", r"chair-bed", r"\bbed settee")),
    ("bed", R(r"matrasbodem", r"foundation", r"bassinette", r"\bbed\b", r"boxbett", r"\bbett", r"divan", r"daybed", r"\blit\b", r"\bletto\b", r"\bcama\b", r"headboard", r"bunk", r"bettgestell", r"boxspring", r"sommier", r"\bcot\b", r"\bcrib\b", r"lattenrost", r"slaapbank", r"footboard")),
    ("sofa", R(r"loveseat", r"sectional", r"\bsofa", r"\bsofá", r"\bcouch", r"seat section", r"chaise longue", r"corner section", r"\bcanapé", r"\d(\.\d)?-seat", r"seat module", r"sitzelement", r"module \d? ?places?", r"\bmodule\b", r"u-shaped", r"méridienne", r"récamiere", r"-zits", r"\bsecção\b", r"\bsección\b", r"\d sections?\b", r"\bdivano", r"conversation")),
    ("armchair", R(r"wing chair", r"armchair", r"\bfauteuil", r"poltrona", r"recliner", r"\bsessel", r"\bsedile\b")),
    ("chair", R(r"\bchair", r"\bstuhl", r"\bsedia", r"\bsilla\b", r"\bchaise\b", r"\bcadeira", r"eetkamerstoel", r"\bstoel", r"כיסא")),
    ("stool", R(r"\bkruk", r"barhocker", r"\bstool", r"\btabouret", r"sgabello", r"\bhocker")),
    ("ottoman", R(r"ottoman", r"footstool", r"pouf", r"repose-pieds", r"repousa", r"הדום", r"poggiapiedi", r"footrest", r"voetensteun", r"fußhocker", r"apoyapiés")),
    ("bench", R(r"\bbench\b", r"\bbank\b", r"\bbanc\b", r"\bpanca\b", r"\bbanca\b", r"-zitsbank")),
    ("coffee_table", R(r"coffee table", r"table basse", r"couchtisch", r"tavolino", r"nesting table", r"nest of tables", r"mesa de centro")),
    ("desk", R(r"table conf[ée]rence", r"monitor stand", r"\bdesk\b", r"bureau", r"conference table", r"schreibtisch", r"escritorio", r"scrivania", r"work ?station", r"computer table", r"שולחן כתיבה", r"laptoptisch", r"secretária", r"laptop stand")),
    ("dining_table", R(r"table (et|and) \d", r"table à abattants", r"table pliante", r"table (et|and) \d", r"table extensible", r"bar table", r"table de bar", r"folding table", r"ausziehtisch", r"picnic table", r"dining table", r"table and \d", r"extendable table", r"table [àa] manger", r"tafel en", r"gateleg", r"drop-leaf", r"dining set", r"שולחן", r"\btisch\b", r"\btafel\b", r"\bmesa\b")),
    ("nightstand", R(r"mesita de noche", r"table chevet", r"nightstand", r"bedside", r"table de chevet", r"nachttisch", r"mesilla", r"comodino", r"mesa de cabeceira")),
    ("side_table", R(r"side table", r"end table", r"console", r"beistelltisch", r"table d'appoint")),
    ("dresser", R(r"\d-drawer chest", r"mini chest", r"\d-drawer chest", r"schubladenelement", r"chest of", r"dresser", r"drawer unit", r"commode", r"cassettiera", r"cómoda", r"kommode", r"caisson [àa] tiroirs", r"gavetero", r"שידת", r"cajones")),
    ("bookshelf", R(r"bücherregal", r"aufsatzregal", r"stellingkast", r"bookcase", r"bookshelf", r"shelving unit", r"shelf unit", r"shelving combination", r"bibliothèque", r"libreria", r"librero", r"estantería", r"\bregal\b", r"étagère", r"\bshelves\b", r"open (shelf|cabinet|storage)", r"estante", r"scaffale", r"יחידת מדפים", r"media shelf", r"book display", r"shelf sections", r"boekenkast", r"gabinete abierto")),
    ("wall_shelf", R(r"^wall shelf", r"floating shelf", r"wall-mounted shelf", r"wall mounted shelf", r"picture ledge", r"^ledge")),
    ("cabinet", R(r"^frame\b", r"cube storage", r"structure tiroir", r"^caixón", r"bovenkast", r"struttura da parete", r"storage combination", r"storage comb", r"combinaison (de )?rangement", r"combinaison structure", r"combinaison rgt", r"combinación", r"combinação", r"comb arr", r"combinazione", r"\bcabinet", r"\bschrank", r"sideboard", r"vitrine", r"\bbuffet", r"highboard", r"\bkast\b", r"armário", r"\bmobile\b", r"\bmeuble\b", r"storage unit", r"wall storage", r"rangement", r"aufbewahrung", r"aufbewkomb", r"opbergcombinatie", r"יחידת אחסון", r"אחסון", r"combi rangement", r"mobili con", r"mueble de almacenamiento", r"storage with", r"laundry combination", r"gabinete", r"\bcab\b", r"tv-kombination", r"mueble", r"ארונית", r"combin")),
    ("rug", R(r"\brug\b", r"\bcarpet\b")),
    ("curtain", R(r"curtain", r"blind", r"rideau", r"gardin", r"vorhang")),
    ("mirror", R(r"table mirror", r"^mirror\b", r"\bmiroir", r"specchio", r"spiegel", r"espejo", r"מראה")),
    ("plant_pot", R(r"plant pot", r"planter", r"plant stand", r"flower pot", r"\bplant\b", r"blumenständer")),
    ("vase", R(r"\bvase")),
    ("artwork", R(r"picture", r"poster", r"painting", r"canvas", r"\bart\b", r"reproduction", r"\bbild\b", r"lienzo", r"afbeelding", r"cuadro", r"quadro")),
    ("textile", R(r"blanket", r"\bthrow\b", r"pillow", r"cushion", r"duvet", r"towel", r"bedspread", r"\bsheet", r"coussin", r"mattress", r"matratze", r"bettpodest")),
]


def classify(noun: str, family: str, rep: str = "") -> str | None:
    n = noun.strip()
    f = (family or "").upper()
    if not n or n in {"#", "white", "black", "grey"}:
        return None
    if f in KITCHEN_FAM:
        if PART_START.search(n):
            return "part"
        if APPLIANCE.search(n) and not re.search(r"cab|cb|élément|schrank|armoire|mobile|pensile|armário", n):
            return "appliance"
        return "kitchen_cabinet"
    if f in BATH_FAM and not PART_START.search(n):
        return "bathroom"
    if PART_START.search(n):
        return "part"
    if re.search(r"mirror cabinet|armoire de toilette|spiegelschrank", n):
        return "bathroom"
    if BATH.search(n):
        return "bathroom"
    if LAMP.search(n) and not re.search(r"led[- ]strip|lighting strip|night light|bulb", n):
        return "lamp"
    if LIGHT_ACC.search(n):
        return "lighting_accessory"
    if KITCHEN_NOUN.search(n) and not re.search(r"wardrobe|bookcase|\btv\b", n):
        return "kitchen_cabinet"
    if APPLIANCE.search(n):
        return "appliance"
    if STORAGE_BOX.search(n):
        return "storage_box"
    for name, rx in HEADS:
        if rx.search(n):
            return name
    if OTHER_SMALL.search(n):
        return "other"
    return None


# ── From a catalog group to a type ──────────────────────────────────────────

_FAMILY_RE = re.compile(r"^[A-ZÅÄÖÆØÜÉ0-9]{3,}(\s*/\s*[A-ZÅÄÖÆØÜÉ0-9]{3,})*\s+")
_COLOUR_WORDS = frozenset(
    "white grey gray black beige oak brown high-gloss matt light dark off-white red blue green yellow pine ash effect "
    "stained anthracite walnut birch pink turquoise orange natural clear lacquered wood veneer stain and with gloss high "
    "mid medium bright pale silver gold".split()
)


class _Colourish:
    """A noun made only of colour/finish words ("light grey", "high-gloss white")."""

    @staticmethod
    def match(text: str) -> bool:
        words = [w for w in re.split(r"[\s/]+", text) if w]
        return bool(words) and len(words) <= 5 and all(w in _COLOUR_WORDS for w in words)


_COLORISH = _Colourish()
_ENGLISH = re.compile(
    r"(with|white|black|and|for|door|drawer|set|frame|unit|table|chair|lamp|grey|gray|brown|oak|pine|wall|storage|bed|sofa)", re.I
)
_SIZE = re.compile(r"\d+([.,/]\d+)?(x\d+([.,/]\d+)?)*\s*(cm|mm|\"|m|ft|'|in)?")

# Types this module adds. When it names one of these it wins over the regex
# type: a METOD kitchen cabinet is not a wardrobe however its title reads.
OVERRIDING_TYPES = frozenset({"kitchen_cabinet", "part", "bathroom", "appliance", "lighting_accessory"})


def _clean(title: str | None) -> str:
    text = re.sub(r"\s*-\s*IKEA.*$", "", title or "")
    return re.sub(r"\(.*?\)", "", text).strip()


def _normalise(segment: str) -> str:
    segment = re.sub(r"\s+-\s+.*$", "", segment)
    return re.sub(r"\s+", " ", _SIZE.sub("#", segment)).strip().lower()


def noun_of(title: str) -> str:
    """The product noun of a catalog title: family name, colours and sizes removed."""
    body = _FAMILY_RE.sub("", title)
    parts = [p.strip() for p in body.split(",")]
    noun = _normalise(parts[0])
    # "LERHYTTAN light grey, drawer front, 80x20 cm": the colour came first.
    if _COLORISH.match(noun) and len(parts) > 1:
        second = _normalise(parts[1])
        if not _COLORISH.match(second):
            noun = second
    return noun


def classify_group(titles: list[str | None]) -> str | None:
    """The type of one product (all the rows that share a picture), from the
    titles of those rows. None when the titles don't say."""
    cleaned = [_clean(t) for t in titles if t]
    english = [t for t in cleaned if _ENGLISH.search(t)]
    representative = (english or cleaned or [""])[0]
    if not representative:
        return None
    return classify(noun_of(representative), representative.split(" ")[0], representative)
