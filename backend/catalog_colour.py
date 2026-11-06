"""
catalog_colour.py
──────────────────
Measures a catalog product's colour from its own photo, instead of trusting
`color_extracted` (one word per product, often the colour of a kitchen
cabinet's carcass rather than its door — see the 2026-10-17 audit: ~1/3 of
"white"-labelled rows had a non-white front). Part of the colour clean-up
plan (catalog-colour-cleanup branch); does not touch the live table or index.

IKEA's `main_image` is a plain, flat JPEG (no alpha channel — confirmed on the
cached photos: 3-component JFIF, no transparency) of a studio photo: in a
sample of 24 products, including ones shot for outdoor furniture (patio sets,
a greenhouse, outdoor storage), every single border pixel was flat
white/near-white — never grass, sky or a lifestyle scene (verified
2026-10-31, bg_test.py in the project's scratchpad) — so a product's own
pixels are simply whatever is NOT close in colour to that backdrop.

An earlier version of this flood-filled from the image border instead of
testing every pixel, on the idea that a product's own pale pixels shouldn't be
treated as background just because they're a similar shade. That backfired
badly: open shelving, chair legs, a table's underside — anything with a gap —
shows backdrop *through* the gap, and that patch of backdrop doesn't touch the
image border, so the flood fill never reached it and counted it as part of the
product instead (found 2026-11-06 on IVAR shelving units: measured "white
60-77%" against a catalog colour of "brown", because most of what the mask
called "product" was daylight showing through the shelf, not the pine frame).
Colour alone can't tell "backdrop seen through a gap" apart from "a pale bit of
the product" either way — the flood fill's connectivity rule didn't solve that
ambiguity, it just picked the wrong default, and far more products have gaps
than have a stray pale highlight. Testing every pixel by colour distance alone
fixes the gap case outright, at the cost of also excluding a product's own
near-background-coloured patches (a white highlight, say) from the swatch
measurement — an existing, accepted limitation, not a new one.

The one real failure mode this doesn't fix: a white or pale, thin product (a
white wire drying rack) blends into the white backdrop almost everywhere, not
just at a gap, leaving too little of the item to measure —
`measure_product_colour` reports that as `low_confidence` instead of guessing,
and callers should keep the catalog's existing colour for those.

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
    """Boolean (H, W) mask of the product's own pixels: True wherever a pixel's
    colour is NOT close to the backdrop's (the border pixels' median colour).
    Every pixel is judged independently by colour distance alone — not by
    whether it is connected to the image border — specifically so that
    backdrop showing through a gap in the product (open shelving, the space
    between chair legs or under a table) is correctly treated as background
    even though it sits in the middle of the photo.

    A product whose own colour also falls inside that tolerance (a white
    product on the white backdrop) is not distinguishable from the background
    by colour alone and gets excluded along with it — `measure_product_colour`
    reports that case as `low_confidence` rather than measuring it wrong."""
    arr = np.asarray(image.convert("RGB")).astype(int)
    border_px = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    bg_color = np.median(border_px, axis=0)
    near_bg = np.abs(arr - bg_color).sum(axis=2) < bg_tolerance
    return ~near_bg


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
