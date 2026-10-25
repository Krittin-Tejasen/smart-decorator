"""
match_features.py
─────────────────
How much an item's measured colour and guessed material agree with a catalog
product, as a *small ranking nudge* for /match-products (product_matching.py).

Why a nudge and not a filter (same lesson as the width filter in
product_matching.py, which fell from 84.5% to ~24% with a 10% error):

- The catalog's colour is one word per product (`color_extracted`) and often
  describes only the carcass: ~1/3 of sampled "white" products have a black,
  oak or green front. That is worst for kitchen/wardrobe/storage systems
  (68% "white") and tables (60%), mildest for sofas (6%) and chairs (21%).
- The catalog's material tags are the same six words on almost everything
  (plastic is on 76% of rows) and say nothing about which part is which.
- The item's colour is measured from an AI-generated picture and its material is
  a model's guess from that picture.

So both fits only move candidates around by a few hundredths of cosine
similarity. They never remove a candidate, and never touch `score`, the picture
similarity that MATCH_MIN_SCORE and the confidence labels use. How much each
fit counts depends on the furniture type (see TYPE_WEIGHT).

The numbers below (penalty sizes, type weights, the colour-distance scale) are
provisional: set from a 240-row catalog sample and 66 cut-outs, not yet tuned on
ranked results.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import numpy as np

from feature_extraction import ciede2000, srgb_to_lab
from feature_vocab import CATALOG_COLOR_SHADES, material_group

# Largest ranking cost of a total mismatch, before the type weight. Cosine
# scores of real matches sit around 0.6–1.0; the width penalty tops out at 0.10.
COLOR_PENALTY_MAX = 0.04
MATERIAL_PENALTY_MAX = 0.03

# How much of those penalties applies per furniture type
# (app.services.furniture_types.FURNITURE_TYPES). Types whose catalog colour is
# mostly the carcass get little; seating gets the full amount.
_FULL = ("sofa", "armchair", "chair", "stool", "ottoman", "bench")
_LOW = ("wardrobe", "cabinet", "bookshelf", "tv_bench", "wall_shelf")
_TABLES = ("coffee_table", "dining_table", "side_table", "desk", "nightstand", "dresser")
TYPE_WEIGHT: dict[str, float] = {
    **{t: 1.0 for t in _FULL},
    **{t: 0.4 for t in _TABLES},
    **{t: 0.25 for t in _LOW},
}
DEFAULT_TYPE_WEIGHT = 0.6  # bed, lamp, rug, curtain, mirror, decor, unknown

# Material groups that say nothing: on 76% of rows (plastic) / "unknown" (other).
_NEUTRAL_GROUPS = frozenset({"plastic", "other"})

# A product whose colour/material the catalog doesn't tell us is not a perfect
# match, so it counts as a half match rather than as no penalty. With "no
# penalty" such products (14% of candidates, no colour label) jumped ahead of
# known-but-imperfect ones and filled 30% of the top-3 slots instead of the 13%
# the picture alone gives them (33 items, real index).
UNKNOWN_FIT = 0.5


# ── Colour ──────────────────────────────────────────────────────────────────

# ΔE2000 at which two colour words stop counting as neighbours.
_NEIGHBOUR_DELTA_E = 30.0


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
    entries = _decode_list(raw)
    shares: dict[str, float] = {}
    for entry in entries:
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


def color_fit(item_colors: Iterable[tuple[str, float]], product_color: Any) -> float | None:
    """0..1: share-weighted similarity between the item's colours and the
    product's single colour word. None when either side is unknown."""
    colors = list(item_colors)
    word = str(product_color or "").strip().lower()
    if not colors or word not in CATALOG_COLOR_SHADES:
        return None
    total = sum(share for _, share in colors)
    if total <= 0:
        return None
    return sum(share * color_similarity(c, word) for c, share in colors) / total


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


def product_material_groups(row: dict[str, Any]) -> set[str]:
    """The product's material groups: every tag in `material_tags` ("Wood|Metal"),
    else the single `material_tag`. Empty when the catalog has none."""
    raw = row.get("material_tags") or row.get("material_tag") or ""
    return {part.strip().lower() for part in str(raw).split("|") if part.strip()}


def material_fit(item_materials: Iterable[tuple[str, float]], row: dict[str, Any]) -> float | None:
    """0..1: confidence-weighted share of the item's informative materials that
    the product also lists. None when there is nothing to compare: the item has
    only plastic/other guesses, or the product's tags are missing or just
    "other"."""
    informative = [(g, c) for g, c in item_materials if g not in _NEUTRAL_GROUPS]
    groups = product_material_groups(row)
    if not informative or not (groups - {"other"}):
        return None
    total = sum(c for _, c in informative)
    if total <= 0:
        return None
    return sum(c for g, c in informative if g in groups) / total


# ── Penalty ─────────────────────────────────────────────────────────────────

def type_weight(furniture_type: str | None) -> float:
    return TYPE_WEIGHT.get(furniture_type or "", DEFAULT_TYPE_WEIGHT)


def has_informative_material(item_materials: Iterable[tuple[str, float]]) -> bool:
    """Whether the item has any material guess other than plastic/other."""
    return any(group not in _NEUTRAL_GROUPS for group, _ in item_materials)


def feature_penalty(furniture_type: str | None, color: float | None, material: float | None) -> float:
    """What the colour and material fits cost a candidate in ranking score, at
    most (COLOR_PENALTY_MAX + MATERIAL_PENALTY_MAX) * the type's weight. A fit
    of None means "no opinion" and costs nothing; callers pass UNKNOWN_FIT for
    a product whose side is unknown while the item's side is known."""
    cost = 0.0
    if color is not None:
        cost += COLOR_PENALTY_MAX * (1.0 - color)
    if material is not None:
        cost += MATERIAL_PENALTY_MAX * (1.0 - material)
    return cost * type_weight(furniture_type)


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
