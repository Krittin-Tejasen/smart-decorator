"""
match_features.py
─────────────────
Turns an item's measured colours and guessed materials into a Pinecone metadata
filter for /match-products (product_matching.py), used only when the user
switches the Colour / Material filter on in the app.

What the filter says
- Colour: keep products whose catalog colour word (`color_extracted`, one of 12)
  is one of the item's colours (each with >= 20% of its pixels, else its main
  colour) or a neighbouring word (similar enough, see COLOR_SIMILARITY), so a
  brown item also keeps beige and dark-brown-ish products.
- Material: keep products tagged with any of the item's materials, using the
  catalog's flags (is_wood / is_fabric / is_metal / is_glass). Fine words are
  folded into those groups (rattan -> wood, leather -> fabric). Plastic,
  ceramic, stone and paper cannot be filtered: the catalog tags plastic on
  ~76% of rows and has no ceramic/stone/paper tag, so they add no condition.

It is a hard filter, so it can leave nothing: the catalog's colour is one word
per product and often only the carcass colour, and its material tags are the
same few words on most products. On 33 real items (5 saved rooms) colour with
neighbours left no item empty, colour + material left 4 of 28. An empty answer is
shown as such in the app, which tells the user to switch the filter off.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from feature_extraction import ciede2000, srgb_to_lab
from feature_vocab import CATALOG_COLOR_SHADES, material_group

# An item colour with at least this share of its pixels is a colour of the item.
MIN_COLOR_SHARE = 0.20

# ΔE2000 at which two colour words stop counting as neighbours.
_NEIGHBOUR_DELTA_E = 30.0
# Words at least this similar (0..1) to an item colour are kept by the filter.
NEIGHBOUR_SIMILARITY = 0.5

# Catalog material group -> the boolean flag Pinecone holds for it.
MATERIAL_FLAG: dict[str, str] = {
    "wood": "is_wood",
    "fabric": "is_fabric",
    "metal": "is_metal",
    "glass": "is_glass",
}


# ── Colour ──────────────────────────────────────────────────────────────────

def _build_similarity() -> dict[tuple[str, str], float]:
    """Similarity of two catalog colour words: 1 for the same word, falling to
    0 as the closest pair of their reference shades gets 30 ΔE2000 apart."""
    names = list(CATALOG_COLOR_SHADES)
    labs = {n: srgb_to_lab(list(CATALOG_COLOR_SHADES[n])) for n in names}
    table: dict[tuple[str, str], float] = {}
    for a in names:
        for b in names:
            if a == b:
                table[(a, b)] = 1.0
                continue
            d = float(ciede2000(labs[a][:, None, :], labs[b][None, :, :]).min())
            table[(a, b)] = round(max(0.0, 1.0 - d / _NEIGHBOUR_DELTA_E), 3)
    return table


COLOR_SIMILARITY = _build_similarity()


def color_similarity(a: str, b: str) -> float:
    return COLOR_SIMILARITY.get((a, b), 0.0)


def parse_item_colors(raw: str | list | None) -> list[tuple[str, float]]:
    """The app's `colors` form field -> [(catalog colour word, share)], shares
    summing to 1, same words merged. Accepts a JSON string or an already
    decoded list of {"catalog_color": ..., "share": ...}. Anything malformed or
    outside the 12 catalog words is ignored."""
    shares: dict[str, float] = {}
    for entry in _decode_list(raw):
        if not isinstance(entry, dict):
            continue
        word = str(entry.get("catalog_color", "")).strip().lower()
        if word not in CATALOG_COLOR_SHADES:
            continue
        share = _number(entry.get("share"), default=0.0)
        if share > 0:
            shares[word] = shares.get(word, 0.0) + share
    total = sum(shares.values())
    if total <= 0:
        return []
    return sorted(((w, s / total) for w, s in shares.items()), key=lambda ws: -ws[1])


def colour_words(item_colors: list[tuple[str, float]]) -> list[str]:
    """The catalog colour words the filter keeps: the item's colours (>= 20%
    share, else its main colour) and every word similar to one of them."""
    colors = list(item_colors)
    if not colors:
        return []
    own = [w for w, share in colors if share >= MIN_COLOR_SHARE] or [colors[0][0]]
    kept = set(own)
    for word in CATALOG_COLOR_SHADES:
        if any(color_similarity(c, word) >= NEIGHBOUR_SIMILARITY for c in own):
            kept.add(word)
    return sorted(kept)


# ── Material ────────────────────────────────────────────────────────────────

def parse_item_materials(raw: str | list | None) -> list[tuple[str, float]]:
    """The app's `materials` form field -> [(catalog material group, confidence)].
    Fine words are folded into the catalog's groups (leather -> fabric, rattan ->
    wood, ceramic/stone/paper -> other); the same group keeps its highest
    confidence; unknown words are ignored."""
    best: dict[str, float] = {}
    for entry in _decode_list(raw):
        if not isinstance(entry, dict):
            continue
        group = material_group(str(entry.get("name", "")))
        if group is None:
            continue
        confidence = min(max(_number(entry.get("confidence"), default=0.5), 0.0), 1.0)
        best[group] = max(best.get(group, 0.0), confidence)
    return sorted(best.items(), key=lambda gc: -gc[1])


def material_flags(item_materials: list[tuple[str, float]]) -> list[str]:
    """The Pinecone flags the filter can use for the item's materials (in order
    of confidence); empty when none of them can be filtered."""
    return [MATERIAL_FLAG[g] for g, _ in item_materials if g in MATERIAL_FLAG]


# ── The filter ──────────────────────────────────────────────────────────────

def metadata_filter(
    item_colors: list[tuple[str, float]] | None = None,
    item_materials: list[tuple[str, float]] | None = None,
) -> dict[str, Any]:
    """Pinecone metadata conditions for the colour and/or material the user
    asked to filter by. {} when there is nothing to filter on, so callers can
    always merge the result into their own filter."""
    out: dict[str, Any] = {}
    words = colour_words(item_colors or [])
    if words:
        out["color"] = {"$in": words}
    flags = material_flags(item_materials or [])
    if flags:
        out["$or"] = [{flag: {"$eq": True}} for flag in flags]
    return out


# ── Helpers ─────────────────────────────────────────────────────────────────

def _decode_list(raw: Any) -> list:
    if isinstance(raw, list):
        return raw
    if not isinstance(raw, str) or not raw.strip():
        return []
    try:
        value = json.loads(raw)
    except ValueError:
        return []
    return value if isinstance(value, list) else []


def _number(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default
