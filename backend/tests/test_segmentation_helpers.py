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

    def test_every_phrase_is_separated(self):
        # A missing " . " once glued "cabinet" onto "floor lamp", so DINO was
        # never asked for either.
        self.assertIn("cabinet", seg._VOCAB_SET)
        self.assertIn("floor lamp", seg._VOCAB_SET)
        self.assertNotIn("cabinetfloor lamp", seg._VOCAB_SET)


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

    def test_chairs_one_behind_another_survive_when_their_masks_are_apart(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        front = self.det("armchair", 0.41, (0.20, 0.50, 0.45, 0.95))
        back = self.det("armchair", 0.38, (0.25, 0.45, 0.47, 0.80))
        front["mask"] = img(_rect_mask(100, 100, 20, 60, 45, 95))
        back["mask"] = img(_rect_mask(100, 100, 30, 45, 47, 59))
        self.assertEqual(len(seg.dedupe_detections([front, back])), 2)

    def test_the_same_chair_twice_keeps_the_more_complete_mask(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        partial = self.det("armchair", 0.39, (0.30, 0.17, 0.38, 0.26))
        whole = self.det("armchair", 0.32, (0.30, 0.17, 0.38, 0.27))
        partial["mask"] = img(_rect_mask(100, 100, 30, 17, 38, 24))
        whole["mask"] = img(_rect_mask(100, 100, 30, 17, 38, 27))
        kept = seg.dedupe_detections([partial, whole])
        self.assertEqual(kept, [whole])

    def test_a_bench_that_sam_painted_as_the_bed_base_is_dropped(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = self.det("bed", 0.8, (0.1, 0.3, 0.8, 0.9))
        base = self.det("bench", 0.31, (0.3, 0.8, 0.75, 0.9))
        bed["mask"] = img(_rect_mask(100, 100, 10, 30, 80, 90))
        base["mask"] = img(_rect_mask(100, 100, 30, 80, 75, 90))
        self.assertEqual(seg.drop_parts_of_hosts([bed, base]), [bed])

    def test_a_real_bench_at_the_foot_of_the_bed_stays(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = self.det("bed", 0.8, (0.1, 0.3, 0.8, 0.8))
        bench = self.det("bench", 0.5, (0.3, 0.75, 0.7, 0.95))
        bed["mask"] = img(_rect_mask(100, 100, 10, 30, 80, 80))
        bench["mask"] = img(_rect_mask(100, 100, 30, 80, 70, 95))
        self.assertEqual(len(seg.drop_parts_of_hosts([bed, bench])), 2)

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


def _rect_mask(h, w, x1, y1, x2, y2):
    import numpy as np

    m = np.zeros((h, w), dtype=bool)
    m[y1:y2, x1:x2] = True
    return m


class SamMaskTests(unittest.TestCase):
    """Mask selection and clean-up between SAM 2 and _build_item."""

    def test_pad_box_grows_and_clips_to_the_image(self):
        self.assertEqual(seg.pad_box([10, 10, 110, 60], 200, 100, 0.1), [0.0, 5.0, 120.0, 65.0])
        self.assertEqual(seg.pad_box([0, 0, 200, 100], 200, 100, 0.1), [0.0, 0.0, 200.0, 100.0])

    def test_picks_the_whole_piece_when_nearly_as_confident_as_a_part(self):
        box = [10, 10, 90, 90]
        part = _rect_mask(100, 100, 30, 30, 60, 60)    # the cushions
        whole = _rect_mask(100, 100, 12, 12, 88, 88)   # the sofa
        self.assertEqual(seg.select_best_mask([part, whole], [0.95, 0.88], box), 1)

    def test_keeps_the_part_when_the_whole_is_much_less_likely(self):
        box = [10, 10, 90, 90]
        part = _rect_mask(100, 100, 30, 30, 60, 60)
        whole = _rect_mask(100, 100, 12, 12, 88, 88)
        self.assertEqual(seg.select_best_mask([part, whole], [0.95, 0.5], box), 0)

    def test_a_mask_spilling_out_of_the_box_is_not_picked(self):
        box = [40, 40, 60, 60]
        inside = _rect_mask(100, 100, 42, 42, 58, 58)
        wall = _rect_mask(100, 100, 0, 0, 100, 100)
        self.assertEqual(seg.select_best_mask([inside, wall], [0.8, 0.9], box), 0)

    def test_clean_mask_clips_drops_strays_and_fills_pinholes(self):
        mask = _rect_mask(100, 100, 20, 20, 80, 80)
        mask[50, 50] = False                     # SAM 2 pin-hole
        mask[0:5, 0:5] = True                    # blob outside the box
        mask[22:24, 85:87] = True                # small stray inside the box
        cleaned = seg.clean_mask(mask, [10, 10, 90, 90])
        self.assertTrue(cleaned[50, 50])
        self.assertFalse(cleaned[0:5, 0:5].any())
        self.assertFalse(cleaned[22:24, 85:87].any())
        self.assertTrue(cleaned[20:80, 20:80].all())

    def test_clean_mask_keeps_a_big_real_gap(self):
        mask = _rect_mask(100, 100, 20, 20, 80, 80)
        mask[35:65, 35:65] = False               # the gap in a chair back
        cleaned = seg.clean_mask(mask, [10, 10, 90, 90])
        self.assertFalse(cleaned[50, 50])

    def test_lamp_pixels_leave_the_nightstand(self):
        nightstand = {"box": (0.2, 0.5, 0.6, 0.9), "mask": _rect_mask(100, 100, 20, 30, 60, 90)}
        lamp = {"box": (0.3, 0.2, 0.45, 0.6), "mask": _rect_mask(100, 100, 30, 20, 45, 60)}
        seg.resolve_overlaps([nightstand, lamp])
        self.assertFalse(nightstand["mask"][40, 35])
        self.assertTrue(lamp["mask"][40, 35])

    def test_pillows_leave_the_bed_they_sit_on(self):
        bed = {"box": (0.1, 0.4, 0.9, 0.9), "mask": _rect_mask(100, 100, 10, 40, 90, 90)}
        pillow = {"box": (0.2, 0.45, 0.4, 0.55), "mask": _rect_mask(100, 100, 20, 45, 40, 55)}
        seg.resolve_overlaps([bed, pillow])
        self.assertFalse(bed["mask"][50, 30])
        self.assertTrue(bed["mask"][70, 60])
        self.assertTrue(pillow["mask"][50, 30])

    def test_the_piece_in_front_gains_nothing(self):
        lamp = {"label": "floor lamp", "box": (0.4, 0.1, 0.6, 0.9), "mask": _rect_mask(100, 100, 40, 10, 60, 90)}
        pot = {"label": "plant pot", "box": (0.35, 0.7, 0.55, 0.9), "mask": _rect_mask(100, 100, 35, 70, 55, 90)}
        pot_before = pot["mask"].copy()
        seg.resolve_overlaps([lamp, pot])
        self.assertTrue((pot["mask"] == pot_before).all())
        self.assertFalse(lamp["mask"][80, 45])
        self.assertTrue(lamp["mask"][30, 50])

    def test_a_chair_keeps_its_seat_cushion(self):
        chair = {"label": "armchair", "box": (0.2, 0.2, 0.8, 0.9), "mask": _rect_mask(100, 100, 20, 20, 80, 90)}
        seat = {"label": "cushion", "box": (0.3, 0.5, 0.7, 0.65), "mask": _rect_mask(100, 100, 30, 50, 70, 65)}
        seg.resolve_overlaps([chair, seat])
        self.assertTrue(chair["mask"][55, 50])

    def test_a_piece_that_loses_most_of_its_mask_is_marked_swallowed(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        lamp = {"label": "floor lamp", "box": (0.4, 0.1, 0.6, 0.5), "mask": img(_rect_mask(100, 100, 40, 10, 60, 50))}
        shade = {"label": "pendant lamp", "box": (0.4, 0.1, 0.6, 0.3), "mask": img(_rect_mask(100, 100, 40, 10, 60, 30))}
        sofa = {"label": "sofa", "box": (0.0, 0.5, 1.0, 1.0), "mask": img(_rect_mask(100, 100, 0, 50, 100, 100))}
        seg._settle_masks([lamp, shade, sofa], 100, 100)
        self.assertFalse(lamp["swallowed"])
        self.assertFalse(sofa["swallowed"])
        lamp2 = {"label": "floor lamp", "box": (0.4, 0.1, 0.6, 0.5), "mask": img(_rect_mask(100, 100, 40, 10, 60, 50))}
        vase = {"label": "vase", "box": (0.45, 0.2, 0.55, 0.45), "mask": img(_rect_mask(100, 100, 40, 12, 60, 50))}
        seg._settle_masks([lamp2, vase], 100, 100)
        self.assertTrue(lamp2["swallowed"])

    def test_a_duvet_sized_pillow_box_is_not_cut_out_of_the_bed(self):
        bed = {"label": "bed", "box": (0.0, 0.3, 0.6, 0.9), "mask": _rect_mask(100, 100, 0, 30, 60, 90)}
        fold = {"label": "cushion", "box": (0.25, 0.4, 0.45, 0.8), "mask": _rect_mask(100, 100, 25, 40, 45, 80)}
        seg.resolve_overlaps([bed, fold])
        self.assertTrue(bed["mask"][60, 35])

    def test_blanket_is_neither_a_card_nor_cut_out(self):
        self.assertFalse(seg.is_furniture_label("blanket"))
        self.assertFalse(seg.is_mask_only_label("blanket"))

    def test_pillows_are_mask_only(self):
        self.assertTrue(seg.is_mask_only_label("cushion"))
        self.assertFalse(seg.is_furniture_label("throw pillow"))
        self.assertTrue(seg.is_furniture_label("sofa"))

    def test_clean_mask_leaves_shelf_openings_open_without_hole_filling(self):
        shelf = _rect_mask(100, 100, 20, 20, 80, 80)
        shelf[25:35, 25:75] = False              # one opening between shelves
        cleaned = seg.clean_mask(shelf, [10, 10, 90, 90], fill_holes=False)
        self.assertFalse(cleaned[30, 50])

    def test_the_bed_keeps_its_base_where_the_rug_runs_under_it(self):
        bed = {"label": "bed", "box": (0.1, 0.3, 0.6, 0.8), "mask": _rect_mask(100, 100, 10, 30, 60, 80)}
        rug = {"label": "rug", "box": (0.15, 0.7, 0.7, 0.95), "mask": _rect_mask(100, 100, 15, 70, 70, 95)}
        seg.resolve_overlaps([bed, rug])
        self.assertTrue(bed["mask"][75, 30])
        self.assertFalse(rug["mask"][75, 30])

    def test_similar_sized_boxes_keep_their_own_pixels(self):
        table = {"label": "dining table", "box": (0.2, 0.4, 0.8, 0.8), "mask": _rect_mask(100, 100, 20, 40, 80, 80)}
        sideboard = {"label": "bookshelf", "box": (0.3, 0.3, 0.8, 0.6), "mask": _rect_mask(100, 100, 30, 30, 80, 60)}
        seg.resolve_overlaps([table, sideboard])
        self.assertTrue(table["mask"][50, 50])
        self.assertTrue(sideboard["mask"][50, 50])

    def test_settle_masks_grows_the_box_to_cover_the_mask(self):
        mask = Image.fromarray(_rect_mask(100, 100, 18, 20, 80, 95).astype("uint8") * 255, mode="L")
        dets = [{"box": (0.2, 0.2, 0.8, 0.9), "mask": mask}]
        seg._settle_masks(dets, 100, 100)
        self.assertEqual(dets[0]["box"], (0.18, 0.2, 0.8, 0.95))


if __name__ == "__main__":
    unittest.main()
