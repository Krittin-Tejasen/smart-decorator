"""Unit tests for catalog_colour.py (measuring a product's colour from its own
studio photo instead of trusting the catalog's single `color_extracted` word).

Run from backend/:  python -m unittest discover -s tests -v
No network: every image here is synthesised.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

import catalog_colour as cc


def studio_photo(size, product_rgb, product_box, bg_rgb=(255, 255, 255)):
    """A plain background with a solid rectangle 'product' on it, the shape
    every IKEA catalog photo in the 2026-10-31 audit turned out to be."""
    img = Image.new("RGB", size, bg_rgb)
    ImageDraw.Draw(img).rectangle(product_box, fill=product_rgb)
    return img


class IsolateForegroundTests(unittest.TestCase):
    def test_a_coloured_product_on_a_white_background_is_isolated(self):
        img = studio_photo((100, 100), (200, 30, 30), (20, 20, 79, 79))
        fg = cc.isolate_foreground(img)
        self.assertTrue(fg[50, 50])           # inside the product
        self.assertFalse(fg[2, 2])            # corner, background
        self.assertAlmostEqual(fg.mean(), 0.36, delta=0.02)   # 60x60 of 100x100

    def test_an_enclosed_gap_inside_the_product_is_background_not_foreground(self):
        # the real bug found 2026-11-06 on IVAR open shelving (measured "white
        # 60-77%" against a catalog colour of "brown"): backdrop visible through
        # a gap in the product — open shelving, the space between chair legs,
        # under a table — must count as background even with NO path back to
        # the image border, because by colour it simply IS the backdrop.
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle((10, 10, 89, 89), fill=(110, 75, 50))     # a solid wood "frame"...
        d.rectangle((30, 30, 69, 69), fill=(255, 255, 255))   # ...with an enclosed backdrop-coloured gap
        fg = cc.isolate_foreground(img)
        self.assertFalse(fg[50, 50])          # inside the gap: background, unreachable from the border or not
        self.assertTrue(fg[15, 15])           # the solid frame itself is still the product

    def test_background_is_measured_from_the_border_not_assumed_white(self):
        img = studio_photo((100, 100), (255, 255, 255), (20, 20, 79, 79), bg_rgb=(10, 10, 10))
        fg = cc.isolate_foreground(img)
        self.assertTrue(fg[50, 50])
        self.assertFalse(fg[2, 2])

    def test_a_product_just_outside_the_tolerance_band_survives_even_though_it_is_pale(self):
        # the background tolerance (sum of |dR|+|dG|+|dB|) is 54; a shade just
        # past it is a different colour, not noise in the same backdrop, and
        # must not be eaten just for being light
        img = studio_photo((100, 100), (220, 220, 215), (20, 20, 79, 79), bg_rgb=(255, 255, 255))
        fg = cc.isolate_foreground(img)
        self.assertTrue(fg[50, 50])

    def test_a_product_within_the_tolerance_band_is_treated_as_background(self):
        # the known, documented limitation (same mechanism as the real white
        # drying-rack case): a product whose colour the flood fill cannot tell
        # apart from the backdrop is swallowed by it, border-connected or not —
        # measure_product_colour's low_confidence flag exists for exactly this
        img = studio_photo((100, 100), (250, 250, 248), (20, 20, 79, 79), bg_rgb=(255, 255, 255))
        fg = cc.isolate_foreground(img)
        self.assertFalse(fg[50, 50])

    def test_an_all_background_photo_has_no_foreground(self):
        img = Image.new("RGB", (60, 60), (255, 255, 255))
        self.assertEqual(cc.isolate_foreground(img).mean(), 0.0)


class MeasureProductColourTests(unittest.TestCase):
    def test_a_clear_product_photo_is_measured_with_high_confidence(self):
        img = studio_photo((100, 100), (110, 75, 50), (10, 10, 89, 89))
        result = cc.measure_product_colour(img)
        self.assertFalse(result.low_confidence)
        self.assertGreater(result.foreground_share, 0.5)
        self.assertEqual(result.swatches[0].catalog_color, "brown")

    def test_a_product_that_blends_into_the_background_is_flagged_low_confidence(self):
        # a thin, near-white shape on a white background (the drying-rack case
        # found on 2026-10-31: foreground share 2%)
        img = Image.new("RGB", (100, 100), (255, 255, 255))
        ImageDraw.Draw(img).line((50, 5, 50, 95), fill=(248, 248, 248), width=1)
        result = cc.measure_product_colour(img)
        self.assertTrue(result.low_confidence)
        self.assertEqual(result.swatches, [])

    def test_the_threshold_is_exactly_what_min_foreground_share_says(self):
        side = int(100 * (cc.MIN_FOREGROUND_SHARE ** 0.5))
        just_under = studio_photo((100, 100), (0, 100, 0), (0, 0, side - 2, side - 2))
        just_over = studio_photo((100, 100), (0, 100, 0), (0, 0, side + 10, side + 10))
        self.assertTrue(cc.measure_product_colour(just_under).low_confidence)
        self.assertFalse(cc.measure_product_colour(just_over).low_confidence)

    def test_never_raises_on_a_tiny_or_blank_image(self):
        for img in (Image.new("RGB", (1, 1), (255, 255, 255)), Image.new("RGB", (4, 4), (0, 0, 0))):
            result = cc.measure_product_colour(img)
            self.assertIsInstance(result, cc.ProductColourMeasurement)

    def test_result_is_deterministic(self):
        img = studio_photo((120, 120), (90, 60, 140), (15, 15, 105, 105))
        a, b = cc.measure_product_colour(img), cc.measure_product_colour(img)
        self.assertEqual([s.model_dump() for s in a.swatches], [s.model_dump() for s in b.swatches])
        self.assertEqual(a.foreground_share, b.foreground_share)


if __name__ == "__main__":
    unittest.main()
