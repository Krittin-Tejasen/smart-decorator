"""
feature_vocab.py
────────────────
The closed vocabularies used by furniture feature extraction, and how they map
onto what the product catalog (`ikea_furniture`) can actually express.

Why two vocabularies: the catalog only has 12 colour words
(`color_extracted`) and 6 material words (`material_tag`: wood / plastic /
metal / fabric / glass / other). Gemini can tell far more than that from a
picture (leather, rattan, ceramic …), and that detail is worth showing to the
user — so items keep the fine words, and `CATALOG_MATERIAL_GROUP` /
`CATALOG_COLORS` translate them for matching.
"""

MATERIALS: tuple[str, ...] = (
    "wood",
    "fabric",
    "leather",
    "metal",
    "glass",
    "ceramic",
    "rattan",
    "paper",
    "stone",
    "plastic",
)

# Fine word -> the catalog's `material_tag` group (lower-case: wood, fabric,
# metal, glass, plastic, other). The catalog has no leather/rattan/ceramic/
# paper/stone tags, so those fold into the nearest group.
CATALOG_MATERIAL_GROUP: dict[str, str] = {
    "wood": "wood",
    "rattan": "wood",
    "fabric": "fabric",
    "leather": "fabric",
    "metal": "metal",
    "glass": "glass",
    "plastic": "plastic",
    "ceramic": "other",
    "stone": "other",
    "paper": "other",
}

# The catalog's `color_extracted` vocabulary (12 words, verified against the
# 59,846-row table on 2026-10-17). Each word has several representative sRGB
# shades — the catalog's "green" covers sage and forest green, "gray" covers
# anthracite — and a measured colour is named after the word of
# its *nearest shade* (ΔE2000). With a single prototype per word, pale sage
# greens came out "gray" and dark greens/browns came out "black" on real
# catalog photos. The first shade of each word is its main one.
CATALOG_COLOR_SHADES: dict[str, tuple[tuple[int, int, int], ...]] = {
    "white": ((245, 245, 240),),
    "black": ((26, 26, 26), (48, 48, 48)),
    "gray": ((140, 140, 140), (95, 95, 97), (192, 192, 190)),
    "brown": ((110, 75, 50), (74, 48, 34), (150, 105, 70)),
    "beige": ((214, 192, 160), (232, 220, 200)),
    "green": ((79, 122, 69), (156, 175, 136), (40, 70, 45)),
    "blue": ((70, 110, 170), (35, 50, 74), (160, 190, 215)),
    "red": ((170, 45, 40), (110, 30, 35)),
    "yellow": ((225, 195, 60),),
    "pink": ((230, 170, 180),),
    "orange": ((225, 125, 40),),
    "purple": ((120, 70, 140), (190, 170, 215)),
}

# Main shade per word (kept for callers that want one swatch per colour word).
CATALOG_COLORS: dict[str, tuple[int, int, int]] = {
    name: shades[0] for name, shades in CATALOG_COLOR_SHADES.items()
}


def material_group(material: str) -> str | None:
    """Catalog material group for a fine material word, or None if unknown."""
    return CATALOG_MATERIAL_GROUP.get(material.strip().lower())
