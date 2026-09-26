"""Unit tests for the pure helpers in segmentation.py.

Run from backend/:  python -m unittest discover -s tests -v
(stdlib unittest on purpose — no extra dev dependency needed.)

These cover exactly the bugs found on 2026-09-26 in the live pipeline: the
Gemini fallback's swapped axes, Grounding DINO's glued-together labels,
duplicate boxes for one object, and the transformers 5.x argument rename.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw

import segmentation as seg


class GeminiBoxTests(unittest.TestCase):
    def test_native_gemini_order_is_y_first_on_a_1000_grid(self):
        # [ymin, xmin, ymax, xmax] — a wide, short sofa near the bottom.
        self.assertEqual(
            seg.gemini_box_to_xyxy([530, 300, 670, 710]),
            (0.3, 0.53, 0.71, 0.67),
        )

    def test_unit_range_boxes_are_not_rescaled(self):
        self.assertEqual(seg.gemini_box_to_xyxy([0.5, 0.2, 0.9, 0.6]), (0.2, 0.5, 0.6, 0.9))

    def test_inverted_corners_are_sorted(self):
        x1, y1, x2, y2 = seg.gemini_box_to_xyxy([700, 800, 100, 200])
        self.assertLess(x1, x2)
        self.assertLess(y1, y2)

    def test_garbage_returns_none(self):
        self.assertIsNone(seg.gemini_box_to_xyxy(["a", 1, 2, 3]))
        self.assertIsNone(seg.gemini_box_to_xyxy([1, 2, 3]))

    def test_parse_uses_box_2d_and_lowercases_labels(self):
        dets = seg.parse_gemini_items(
            [
                {"label": "Sofa", "box_2d": [530, 300, 670, 710], "confidence": 0.9},
                {"label": "junk", "box_2d": "nope"},
                "not a dict",
                {"label": "rug", "bbox": [650, 50, 800, 970]},  # legacy key still accepted
            ]
        )
        self.assertEqual([d["label"] for d in dets], ["sofa", "rug"])
        self.assertEqual(dets[0]["box"], [0.3, 0.53, 0.71, 0.67])


class LabelTests(unittest.TestCase):
    def test_glued_dino_phrases_collapse_to_the_first_vocabulary_phrase(self):
        cases = {
            "armchair chair chair office chair": "armchair",
            "rug carpet": "rug",
            "sofa couch": "sofa",
            "table lamp pendant lamp": "table lamp",
            "console table desk cabinet tv": "console table",
            "office chair": "office chair",
            "vase": "vase",
        }
        for raw, expected in cases.items():
            self.assertEqual(seg.canonicalize_label(raw), expected, raw)

    def test_a_door_or_window_match_wins_over_the_furniture_word_beside_it(self):
        # Real raw labels from Grounding DINO on a generated room.
        self.assertEqual(seg.canonicalize_label("blinds window"), "window")
        self.assertEqual(seg.canonicalize_label("wardrobe door"), "door")
        self.assertFalse(seg.is_furniture_label(seg.canonicalize_label("blinds window")))
        self.assertFalse(seg.is_furniture_label(seg.canonicalize_label("wardrobe door")))

    def test_unknown_and_empty_labels_pass_through(self):
        self.assertEqual(seg.canonicalize_label("Floating Shelf"), "floating shelf")
        self.assertEqual(seg.canonicalize_label(""), "furniture")

    def test_doors_and_windows_are_not_furniture(self):
        self.assertFalse(seg.is_furniture_label("door"))
        self.assertFalse(seg.is_furniture_label(" Window "))
        self.assertTrue(seg.is_furniture_label("sofa"))
        self.assertTrue(seg.is_furniture_label("curtain"))

    def test_decoys_are_in_the_dino_prompt(self):
        self.assertIn("door", seg._VOCAB_SET)
        self.assertIn("window", seg._VOCAB_SET)
        self.assertTrue(seg._FURNITURE_PROMPT.rstrip().endswith("."))


class DedupeTests(unittest.TestCase):
    @staticmethod
    def det(label, conf, box):
        return {"label": label, "confidence": conf, "box": box}

    def test_same_chair_reported_twice_keeps_the_more_confident(self):
        kept = seg.dedupe_detections(
            [
                self.det("office chair", 0.30, (0.71, 0.54, 0.92, 0.75)),
                self.det("armchair", 0.42, (0.73, 0.56, 0.89, 0.73)),
            ]
        )
        self.assertEqual([d["label"] for d in kept], ["armchair"])

    def test_pillow_inside_a_sofa_box_survives(self):
        kept = seg.dedupe_detections(
            [
                self.det("sofa", 0.8, (0.30, 0.50, 0.70, 0.70)),
                self.det("pillow", 0.6, (0.35, 0.50, 0.42, 0.58)),
            ]
        )
        self.assertEqual(sorted(d["label"] for d in kept), ["pillow", "sofa"])

    def test_two_separate_vases_survive(self):
        kept = seg.dedupe_detections(
            [
                self.det("vase", 0.5, (0.02, 0.50, 0.05, 0.57)),
                self.det("vase", 0.5, (0.05, 0.53, 0.08, 0.56)),
            ]
        )
        self.assertEqual(len(kept), 2)

    def test_table_and_cabinet_boxes_on_one_piece_merge(self):
        kept = seg.dedupe_detections(
            [
                self.det("console table", 0.41, (0.0, 0.55, 0.12, 0.71)),
                self.det("bookshelf", 0.31, (0.05, 0.57, 0.12, 0.69)),
            ]
        )
        self.assertEqual([d["label"] for d in kept], ["console table"])

    def test_near_identical_boxes_merge_even_across_families(self):
        kept = seg.dedupe_detections(
            [
                self.det("mirror", 0.9, (0.1, 0.1, 0.5, 0.5)),
                self.det("rug", 0.4, (0.1, 0.1, 0.5, 0.51)),
            ]
        )
        self.assertEqual(len(kept), 1)


class ThresholdKwargTests(unittest.TestCase):
    def test_uses_threshold_when_the_installed_transformers_renamed_it(self):
        class New:
            def post_process_grounded_object_detection(self, outputs, input_ids=None, threshold=0.25,
                                                       text_threshold=0.25, target_sizes=None):
                pass

        self.assertEqual(
            seg._post_process_threshold_kwargs(New(), 0.3, 0.2),
            {"threshold": 0.3, "text_threshold": 0.2},
        )

    def test_keeps_box_threshold_for_older_transformers(self):
        class Old:
            def post_process_grounded_object_detection(self, outputs, input_ids, box_threshold=0.25,
                                                       text_threshold=0.25, target_sizes=None):
                pass

        self.assertEqual(
            seg._post_process_threshold_kwargs(Old(), 0.3, 0.2),
            {"box_threshold": 0.3, "text_threshold": 0.2},
        )


class BuildItemTests(unittest.TestCase):
    def setUp(self):
        # 400x400 room: red "sofa" square on a blue wall.
        self.img = Image.new("RGB", (400, 400), (30, 60, 200))
        ImageDraw.Draw(self.img).rectangle((100, 200, 300, 300), fill=(200, 30, 30))
        self.mask = Image.new("L", (400, 400), 0)
        ImageDraw.Draw(self.mask).rectangle((100, 200, 300, 300), fill=255)

    def build(self, mask):
        # detection box deliberately a bit larger than the object
        return seg._build_item(self.img, "sofa", 0.9, 0, 0.2, 0.4, 0.8, 0.85, mask)

    def test_real_mask_gives_a_trimmed_transparent_cutout_and_object_colours(self):
        item = self.build(self.mask)
        self.assertTrue(item.mask_precise)
        self.assertIsNotNone(item.cutout_image)
        self.assertTrue(item.cutout_image.startswith("data:image/png;base64,"))
        self.assertTrue(item.crop_image.startswith("data:image/jpeg;base64,"))

        import base64, io
        cut = Image.open(io.BytesIO(base64.b64decode(item.cutout_image.split(",", 1)[1])))
        self.assertEqual(cut.mode, "RGBA")
        # trimmed to the mask, not the 240x180 detection box (PIL's rectangle()
        # is end-inclusive, hence 201x101 for a (100,200)-(300,300) fill)
        self.assertEqual(cut.size, (201, 101))
        self.assertEqual(cut.getpixel((5, 5))[3], 255)

        # dominant colours come from the sofa's pixels, not the blue wall in the bbox
        first = item.features.dominant_colors[0]
        r, g, b = (int(first[i : i + 2], 16) for i in (1, 3, 5))
        self.assertGreater(r, b)

    def test_no_mask_falls_back_to_a_rectangle_and_no_cutout(self):
        item = self.build(None)
        self.assertFalse(item.mask_precise)
        self.assertIsNone(item.cutout_image)

    def test_tiny_mask_is_treated_as_no_mask(self):
        tiny = Image.new("L", (400, 400), 0)
        ImageDraw.Draw(tiny).rectangle((150, 250, 153, 253), fill=255)
        item = self.build(tiny)
        self.assertFalse(item.mask_precise)
        self.assertIsNone(item.cutout_image)

    def test_thumbnails_are_capped(self):
        big = Image.new("RGB", (2000, 2000), (10, 200, 10))
        item = seg._build_item(big, "rug", 0.9, 0, 0.0, 0.0, 1.0, 1.0, None)

        import base64, io
        crop = Image.open(io.BytesIO(base64.b64decode(item.crop_image.split(",", 1)[1])))
        self.assertLessEqual(max(crop.size), seg._THUMB_PX)


if __name__ == "__main__":
    unittest.main()
