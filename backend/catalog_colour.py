"""
catalog_colour.py
──────────────────
Measures a catalog product's colour from its own photo, instead of trusting
`color_extracted` (one word per product, often the colour of a kitchen
cabinet's carcass rather than its door — see the 2026-10-17 audit: ~1/3 of
"white"-labelled rows had a non-white front). Part of the colour clean-up
plan (catalog-colour-cleanup branch); does not touch the live table or index.

IKEA's `main_image` is a plain studio photo: in a sample of 24 products,
including ones shot for outdoor furniture (patio sets, a greenhouse, outdoor
storage), every single border pixel was flat white/near-white — never grass,
sky or a lifestyle scene — so a simple flood fill from the image border is
enough to separate the product from its background (verified 2026-10-31,
bg_test.py in the project's scratchpad). The one real failure mode found:
a white or pale, thin product (a white wire drying rack) blends into the white
backdrop and the flood fill eats most of it, leaving too little of the item to
measure — `measure_product_colour` reports that as `low_confidence` instead of
guessing, and callers should keep the catalog's existing colour for those.

Reuses feature_extraction.extract_swatches — the same CIELAB k-means swatch
extractor already used on room cut-outs — so a product's measured colour and a
detected item's measured colour are directly comparable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from feature_extraction import ColorSwatch, extract_swatches

# A product whose isolated foreground is smaller than this share of the image
# is not trustworthy to measure (white-on-white erosion, a near-empty photo, a
# decorative prop taking up most of the frame) — callers should fall back to
# the catalog's own `color_extracted` instead of a measured colour.
MIN_FOREGROUND_SHARE = 0.08

# How close to the border's own colour a pixel must be to count as background,
# as a sum of absolute differences over R+G+B (0-765). IKEA's studio backdrop
# is flat, so this only needs to be generous enough for JPEG noise and a soft
# shadow gradient, not an actual second colour.
_BG_TOLERANCE = 54


@dataclass
class ProductColourMeasurement:
    swatches: list[ColorSwatch]   # up to 3, most common first (empty if low_confidence)
    foreground_share: float       # 0..1, how much of the photo was the product
    low_confidence: bool          # True: caller should keep the existing catalog colour


def isolate_foreground(image: Image.Image, bg_tolerance: int = _BG_TOLERANCE) -> np.ndarray:
    """Boolean (H, W) mask of the product's own pixels: True where NOT part of
    the background region that touches the image border. Flood fill from the
    four edges over pixels close to the border's own (median) colour.

    A product whose own colour also falls inside that tolerance (a white
    product on the white backdrop) is not distinguishable from the background
    by colour alone and gets swallowed along with it — `measure_product_colour`
    reports that case as `low_confidence` rather than measuring it wrong."""
    arr = np.asarray(image.convert("RGB")).astype(int)
    h, w, _ = arr.shape
    border_px = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg_color = np.median(border_px, axis=0)
    near_bg = np.abs(arr - bg_color).sum(axis=2) < bg_tolerance

    # Flood fill from the border over `near_bg`, iteratively (no recursion
    # limit issues on a few-hundred-pixel photo): repeatedly grow a frontier
    # seeded at the border until it stops spreading.
    reached = np.zeros((h, w), dtype=bool)
    reached[0, :] |= near_bg[0, :]
    reached[-1, :] |= near_bg[-1, :]
    reached[:, 0] |= near_bg[:, 0]
    reached[:, -1] |= near_bg[:, -1]
    frontier = reached.copy()
    while frontier.any():
        grown = frontier.copy()
        grown[1:, :] |= frontier[:-1, :]
        grown[:-1, :] |= frontier[1:, :]
        grown[:, 1:] |= frontier[:, :-1]
        grown[:, :-1] |= frontier[:, 1:]
        grown &= near_bg
        frontier = grown & ~reached
        reached |= grown

    return ~reached


def measure_product_colour(image: Image.Image) -> ProductColourMeasurement:
    """The product's own colour(s), isolating it from its studio background
    first. Never raises on a normal photo; `low_confidence=True` with empty
    swatches means "could not tell, keep the existing catalog colour"."""
    fg = isolate_foreground(image)
    share = float(fg.mean())
    if share < MIN_FOREGROUND_SHARE:
        return ProductColourMeasurement(swatches=[], foreground_share=share, low_confidence=True)

    rgba = np.dstack([np.asarray(image.convert("RGB")), (fg * 255).astype("uint8")])
    swatches = extract_swatches(Image.fromarray(rgba, "RGBA"))
    if not swatches:
        return ProductColourMeasurement(swatches=[], foreground_share=share, low_confidence=True)
    return ProductColourMeasurement(swatches=swatches, foreground_share=share, low_confidence=False)
