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

import numpy as np
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

    def test_tvs_and_appliances_are_never_cards(self):
        for label in ("tv", "TV", "television", "flat screen tv", "wall-mounted tv",
                      "refrigerator", "stainless steel fridge", "microwave", "oven",
                      "washing machine", "air conditioner", "ceiling fan", "speaker",
                      "soundbar", "computer monitor"):
            self.assertFalse(seg.is_candidate_label(label), label)
            self.assertFalse(seg.is_furniture_label(label), label)

    def test_tv_furniture_is_still_furniture(self):
        for label in ("tv stand", "tv bench", "tv cabinet", "media console", "sofa"):
            self.assertTrue(seg.is_furniture_label(label), label)
        self.assertEqual(seg.canonicalize_label("tv"), "tv")

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
        self.assertEqual(len(kept), 1)
        self.assertIs(kept[0]["mask"], whole["mask"])
        self.assertEqual(kept[0]["confidence"], 0.39)

    def test_a_bench_that_sam_painted_as_the_bed_base_is_dropped(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = self.det("bed", 0.8, (0.1, 0.3, 0.8, 0.9))
        base = self.det("bench", 0.31, (0.3, 0.8, 0.75, 0.9))
        bed["mask"] = img(_rect_mask(100, 100, 10, 30, 80, 90))
        base["mask"] = img(_rect_mask(100, 100, 30, 80, 75, 90))
        self.assertEqual(seg.merge_parts_into_hosts([bed, base]), [bed])

    def test_an_armchair_takes_back_its_seat_read_as_an_ottoman(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        chair = self.det("armchair", 0.38, (0.72, 0.65, 0.92, 0.97))
        seat = self.det("ottoman", 0.36, (0.72, 0.76, 0.86, 0.87))
        chair["mask"] = img(_rect_mask(100, 100, 72, 65, 92, 75))
        seat["mask"] = img(_rect_mask(100, 100, 72, 76, 86, 87))
        kept = seg.merge_parts_into_hosts([chair, seat])
        self.assertEqual(kept, [chair])
        self.assertTrue((np.asarray(chair["mask"]) > 127)[80, 80])

    def test_a_footstool_in_front_of_the_armchair_stays(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        chair = self.det("armchair", 0.5, (0.5, 0.4, 0.8, 0.8))
        stool = self.det("ottoman", 0.5, (0.55, 0.7, 0.75, 0.95))
        chair["mask"] = img(_rect_mask(100, 100, 50, 40, 80, 80))
        stool["mask"] = img(_rect_mask(100, 100, 55, 82, 75, 95))
        self.assertEqual(len(seg.merge_parts_into_hosts([chair, stool])), 2)

    def test_a_real_bench_at_the_foot_of_the_bed_stays(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = self.det("bed", 0.8, (0.1, 0.3, 0.8, 0.8))
        bench = self.det("bench", 0.5, (0.3, 0.75, 0.7, 0.95))
        bed["mask"] = img(_rect_mask(100, 100, 10, 30, 80, 80))
        bench["mask"] = img(_rect_mask(100, 100, 30, 80, 70, 95))
        self.assertEqual(len(seg.merge_parts_into_hosts([bed, bench])), 2)

    def test_a_headboard_above_the_bed_box_joins_the_bed(self):
        # DINO's bed box stops at the mattress; the headboard stands out of it.
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = self.det("bed", 0.8, (0.2, 0.5, 0.8, 0.9))
        head = self.det("headboard", 0.5, (0.22, 0.3, 0.78, 0.6))
        bed["mask"] = img(_rect_mask(100, 100, 20, 50, 80, 90))
        head["mask"] = img(_rect_mask(100, 100, 22, 30, 78, 60))
        self.assertEqual(seg.merge_parts_into_hosts([bed, head]), [bed])
        self.assertTrue((np.asarray(bed["mask"]) > 127)[35, 50])

    def test_a_headboard_is_never_a_card(self):
        self.assertFalse(seg.is_furniture_label("headboard"))
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        # No bed under it, or only a sofa: dropped, like a pillow on nothing.
        head = {"label": "headboard", "confidence": 0.5, "box": (0.2, 0.2, 0.6, 0.4), "mask": img(_rect_mask(100, 100, 20, 20, 60, 40))}
        sofa = {"label": "sofa", "confidence": 0.8, "box": (0.1, 0.1, 0.7, 0.8), "mask": img(_rect_mask(100, 100, 10, 10, 70, 80))}
        furniture, _ = seg.select_detections([head, sofa], [head["mask"], sofa["mask"]], 100, 100)
        self.assertEqual([d["label"] for d in furniture], ["sofa"])

    def test_a_headboard_beside_the_bed_is_not_its_head(self):
        self.assertFalse(seg.is_bed_head((0.0, 0.3, 0.15, 0.6), (0.2, 0.5, 0.8, 0.9)))

    def test_a_mirror_inside_the_bed_box_is_its_headboard(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = self.det("bed", 0.8, (0.1, 0.3, 0.8, 0.9))
        headboard = self.det("mirror", 0.34, (0.08, 0.3, 0.3, 0.55))
        bed["mask"] = img(_rect_mask(100, 100, 25, 45, 80, 90))
        headboard["mask"] = img(_rect_mask(100, 100, 10, 30, 30, 55))
        self.assertEqual(seg.merge_parts_into_hosts([bed, headboard]), [bed])
        self.assertTrue((np.asarray(bed["mask"]) > 127)[35, 15])

    def test_a_mirror_above_the_bed_stays(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = self.det("bed", 0.8, (0.1, 0.5, 0.8, 0.9))
        mirror = self.det("mirror", 0.6, (0.3, 0.1, 0.6, 0.45))
        bed["mask"] = img(_rect_mask(100, 100, 10, 50, 80, 90))
        mirror["mask"] = img(_rect_mask(100, 100, 30, 10, 60, 45))
        self.assertEqual(len(seg.merge_parts_into_hosts([bed, mirror])), 2)

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

    def test_a_small_unlabelled_box_in_front_takes_its_pixels(self):
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

    def test_a_duvet_mask_as_big_as_the_bed_does_not_replace_the_bed(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = {"label": "bed", "confidence": 0.8, "box": (0.3, 0.4, 0.7, 0.8), "mask": img(_rect_mask(100, 100, 30, 40, 70, 80))}
        duvet = {"label": "blanket", "confidence": 0.33, "box": (0.3, 0.55, 0.7, 0.75), "mask": img(_rect_mask(100, 100, 30, 40, 70, 81))}
        furniture, _ = seg.select_detections([bed, duvet], [bed["mask"], duvet["mask"]], 100, 100)
        self.assertEqual([d["label"] for d in furniture], ["bed"])

    def test_a_bed_deduped_against_its_headboard_still_gets_the_duvet(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        # SAM 2 painted only the headboard for the bed box; the headboard box
        # got a slightly bigger mask of the same thing.
        bed = {"label": "bed", "confidence": 0.77, "box": (0.3, 0.37, 0.7, 0.6), "mask": img(_rect_mask(100, 100, 38, 37, 62, 44))}
        head = {"label": "headboard", "confidence": 0.54, "box": (0.38, 0.36, 0.62, 0.44), "mask": img(_rect_mask(100, 100, 38, 36, 62, 44))}
        duvet = {"label": "blanket", "confidence": 0.32, "box": (0.31, 0.43, 0.69, 0.52), "mask": img(_rect_mask(100, 100, 31, 43, 69, 52))}
        furniture, mask_only = seg.select_detections(
            [bed, head, duvet], [bed["mask"], head["mask"], duvet["mask"]], 100, 100
        )
        self.assertEqual([d["label"] for d in furniture], ["bed"])
        self.assertEqual(mask_only, [])
        self.assertTrue((np.asarray(furniture[0]["mask"]) > 127)[48, 50])

    def test_a_four_poster_reported_twice_is_one_bed(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        frame = {"label": "bed", "confidence": 0.45, "box": (0.3, 0.2, 0.7, 0.8), "mask": img(_rect_mask(100, 100, 30, 20, 32, 80))}
        mattress = {"label": "bed", "confidence": 0.53, "box": (0.31, 0.5, 0.69, 0.79), "mask": img(_rect_mask(100, 100, 33, 50, 68, 79))}
        furniture, _ = seg.select_detections([frame, mattress], [frame["mask"], mattress["mask"]], 100, 100)
        self.assertEqual(len(furniture), 1)
        merged = np.asarray(furniture[0]["mask"]) > 127
        self.assertTrue(merged[60, 31] and merged[60, 50])

    def test_the_garden_chair_seen_through_the_doorway_is_dropped(self):
        doorway = (0.15, 0.27, 0.30, 0.74)
        self.assertTrue(seg.seen_through_opening((0.22, 0.57, 0.30, 0.70), [doorway]))
        # A chair in front of the door reaches below its sill.
        self.assertFalse(seg.seen_through_opening((0.18, 0.55, 0.28, 0.90), [doorway]))

    def test_glass_walls_and_bed_frames_are_not_doorways(self):
        doorway = (0.15, 0.27, 0.30, 0.74)
        glass_wall = (0.80, 0.0, 1.0, 0.96)
        four_poster = (0.33, 0.23, 0.68, 0.78)
        bed_box = (0.326, 0.236, 0.679, 0.779)
        self.assertEqual(seg.real_openings([doorway, glass_wall, four_poster], [bed_box]), [doorway])

    def test_tables_beside_the_bed_are_nightstands(self):
        bed = {"label": "bed", "box": (0.28, 0.50, 0.73, 0.82)}
        left = {"label": "console table", "box": (0.277, 0.575, 0.373, 0.688)}
        right = {"label": "desk", "box": (0.633, 0.576, 0.728, 0.689)}
        bench = {"label": "side table", "box": (0.426, 0.684, 0.586, 0.93)}     # at the foot, below the bed
        sideboard = {"label": "dresser", "box": (0.756, 0.564, 0.89, 0.726)}    # too big
        far = {"label": "side table", "box": (0.031, 0.631, 0.15, 0.824)}       # by the chaise
        dets = [bed, left, right, bench, sideboard, far]
        seg.relabel_bedside_tables(dets)
        self.assertEqual([d["label"] for d in dets],
                         ["bed", "nightstand", "nightstand", "side table", "dresser", "side table"])

    def test_clip_moves_a_lone_armchair_off_sofa_when_sure(self):
        cands = seg.clip_candidates("sofa")
        probs = [0.01 if c != "armchair" else 0.97 for c in cands]
        self.assertEqual(seg.clip_verdict("sofa", cands, probs), "armchair")

    def test_clip_does_not_shuffle_names_inside_a_product_type(self):
        cands = seg.clip_candidates("console table")
        probs = [0.9 if c == "desk" else 0.02 for c in cands]
        self.assertEqual(seg.clip_verdict("console table", cands, probs), "console table")

    def test_clip_needs_to_be_far_more_sure_than_of_dinos_label(self):
        cands = seg.clip_candidates("chandelier")
        probs = [0.62 if c == "decoration" else (0.2 if c == "chandelier" else 0.01) for c in cands]
        self.assertEqual(seg.clip_verdict("chandelier", cands, probs), "chandelier")

    def test_clip_leaves_labels_without_known_swaps_alone(self):
        for label in ("nightstand", "bed", "rug", "desk", "artwork"):
            self.assertIsNone(seg.clip_candidates(label))

    def test_word_pieces_are_rejoined(self):
        self.assertEqual(seg.canonicalize_label("##board"), "headboard")
        self.assertEqual(seg.canonicalize_label("head ##board"), "headboard")
        self.assertEqual(seg.canonicalize_label("plant"), "plant pot")
        self.assertEqual(seg.canonicalize_label("doorway"), "doorway")

    def test_blanket_is_never_a_card(self):
        self.assertFalse(seg.is_furniture_label("blanket"))
        self.assertTrue(seg.is_candidate_label("blanket"))
        self.assertFalse(seg.is_candidate_label("door"))

    def test_a_blanket_on_the_bed_joins_the_bed(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        bed = {"label": "bed", "confidence": 0.8, "box": (0.1, 0.3, 0.9, 0.9), "mask": img(_rect_mask(100, 100, 10, 30, 60, 90))}
        duvet = {"label": "blanket", "confidence": 0.37, "box": (0.4, 0.5, 0.88, 0.85), "mask": img(_rect_mask(100, 100, 40, 50, 88, 85))}
        furniture, mask_only = seg.select_detections([bed, duvet], [bed["mask"], duvet["mask"]], 100, 100)
        self.assertEqual([d["label"] for d in furniture], ["bed"])
        self.assertEqual(mask_only, [])
        self.assertTrue((np.asarray(furniture[0]["mask"]) > 127)[70, 80])

    def test_a_bench_at_the_foot_keeps_its_pixels_from_the_bed(self):
        bed = {"label": "bed", "box": (0.30, 0.49, 0.71, 0.80), "mask": _rect_mask(100, 100, 30, 49, 71, 85)}
        bench = {"label": "bench", "box": (0.33, 0.67, 0.67, 0.85), "mask": _rect_mask(100, 100, 33, 67, 67, 85)}
        seg.resolve_overlaps([bed, bench])
        self.assertFalse(bed["mask"][80, 50])
        self.assertTrue(bench["mask"][80, 50])

    def test_pillows_and_throws_join_the_sofa(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        sofa = {"label": "sofa", "confidence": 0.7, "box": (0.1, 0.4, 0.9, 0.8), "mask": img(_rect_mask(100, 100, 10, 50, 90, 80))}
        throw = {"label": "blanket", "confidence": 0.4, "box": (0.6, 0.45, 0.75, 0.7), "mask": img(_rect_mask(100, 100, 60, 45, 75, 70))}
        pillow = {"label": "pillow", "confidence": 0.5, "box": (0.2, 0.42, 0.35, 0.55), "mask": img(_rect_mask(100, 100, 20, 42, 35, 55))}
        stray = {"label": "pillow", "confidence": 0.4, "box": (0.0, 0.9, 0.05, 0.98), "mask": img(_rect_mask(100, 100, 0, 90, 5, 98))}
        dets = [sofa, throw, pillow, stray]
        furniture, mask_only = seg.select_detections(dets, [d["mask"] for d in dets], 100, 100)
        self.assertEqual([d["label"] for d in furniture], ["sofa"])
        self.assertEqual(mask_only, [])
        seg._settle_masks(furniture, 100, 100)
        merged = np.asarray(furniture[0]["mask"]) > 127
        self.assertTrue(merged[46, 65] and merged[44, 25])
        self.assertFalse(merged[94, 2])

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

    def test_the_desk_front_goes_to_the_desk_not_the_chair_behind_it(self):
        # room31: the office chair's box covers the desk front, and both masks
        # hold it. Asked at the front, SAM 2's whole-object answer is the desk.
        desk = {"label": "desk", "confidence": 0.45, "box": (0.1, 0.5, 0.9, 0.9), "mask": _rect_mask(100, 100, 10, 50, 90, 90)}
        chair = {"label": "armchair", "confidence": 0.6, "box": (0.1, 0.2, 0.4, 0.9), "mask": _rect_mask(100, 100, 10, 20, 40, 90)}
        asked = []

        def ask(x, y):
            asked.append((x, y))
            return np.stack([_rect_mask(100, 100, 10, 50, 40, 90), _rect_mask(100, 100, 10, 50, 90, 90)])

        seg.resolve_overlaps([desk, chair])
        seg.split_contested([desk, chair], ask)
        self.assertEqual(len(asked), 1)
        self.assertTrue(10 <= asked[0][0] < 40 and 50 <= asked[0][1] < 90)
        self.assertFalse((desk["mask"] & chair["mask"]).any())
        self.assertTrue(desk["mask"][70, 25])
        self.assertTrue(chair["mask"][30, 25])

    def test_a_shared_patch_goes_to_the_piece_sam_answers_with(self):
        # The front dining chair's back hides part of the one behind it.
        behind = {"label": "armchair", "confidence": 0.8, "box": (0.2, 0.2, 0.5, 0.8), "mask": _rect_mask(100, 100, 20, 20, 50, 80)}
        front = {"label": "armchair", "confidence": 0.4, "box": (0.4, 0.3, 0.8, 0.9), "mask": _rect_mask(100, 100, 40, 30, 80, 90)}
        seg.split_contested([behind, front], lambda x, y: front["mask"][None].copy())
        self.assertTrue(front["mask"][50, 45])
        self.assertFalse(behind["mask"][50, 45])

    def test_without_sam_the_more_confident_piece_keeps_the_patch(self):
        a = {"label": "desk", "confidence": 0.3, "box": (0.1, 0.1, 0.6, 0.6), "mask": _rect_mask(100, 100, 10, 10, 60, 60)}
        b = {"label": "armchair", "confidence": 0.7, "box": (0.4, 0.4, 0.9, 0.9), "mask": _rect_mask(100, 100, 40, 40, 90, 90)}

        def broken(x, y):
            raise RuntimeError("no SAM")

        seg.split_contested([a, b], broken)
        self.assertTrue(b["mask"][50, 50])
        self.assertFalse(a["mask"][50, 50])

    def test_the_loser_drops_slivers_by_the_patch_but_keeps_its_own_parts(self):
        # room10: one box on both frames, one on the right frame. The left
        # box keeps the left frame; the strip left beside the right frame goes,
        # a separate small piece away from the patch (a chair's seat) stays.
        both = {"label": "artwork", "confidence": 0.5, "box": (0.0, 0.0, 1.0, 0.6), "mask": _rect_mask(100, 100, 5, 10, 95, 50)}
        both["mask"][80:84, 5:9] = True
        right = {"label": "artwork", "confidence": 0.4, "box": (0.55, 0.1, 0.95, 0.5), "mask": _rect_mask(100, 100, 55, 10, 93, 50)}
        seg.split_contested([both, right], lambda x, y: right["mask"][None].copy())
        self.assertTrue(both["mask"][30, 20])
        self.assertFalse(both["mask"][30, 94])
        self.assertTrue(both["mask"][82, 7])
        self.assertTrue(right["mask"][30, 70])

    def test_a_chair_left_with_only_its_back_stays_a_card(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        desk = {"label": "desk", "confidence": 0.45, "box": (0.1, 0.5, 0.9, 0.9), "mask": img(_rect_mask(100, 100, 10, 50, 90, 90))}
        chair = {"label": "armchair", "confidence": 0.6, "box": (0.1, 0.4, 0.4, 0.9), "mask": img(_rect_mask(100, 100, 10, 40, 40, 90))}
        seg._settle_masks([desk, chair], 100, 100, ask=lambda x, y: _rect_mask(100, 100, 10, 50, 90, 90)[None])
        self.assertFalse(chair["swallowed"])
        self.assertEqual(int((np.asarray(chair["mask"]) > 127).sum()), 300)

    def test_settle_masks_leaves_no_pixel_in_two_masks(self):
        img = lambda m: Image.fromarray(m.astype("uint8") * 255, mode="L")
        table = {"label": "dining table", "confidence": 0.5, "box": (0.2, 0.4, 0.8, 0.8), "mask": img(_rect_mask(100, 100, 20, 40, 80, 80))}
        sideboard = {"label": "bookshelf", "confidence": 0.4, "box": (0.3, 0.3, 0.8, 0.6), "mask": img(_rect_mask(100, 100, 30, 30, 80, 60))}
        rug = {"label": "rug", "confidence": 0.5, "box": (0.0, 0.7, 1.0, 1.0), "mask": img(_rect_mask(100, 100, 0, 70, 100, 100))}
        dets = [table, sideboard, rug]
        seg._settle_masks(dets, 100, 100, ask=lambda x, y: sideboard["mask"][None].copy())
        self.assertLessEqual(int(np.max(sum((np.asarray(d["mask"]) > 127).astype(int) for d in dets))), 1)
        self.assertTrue(np.asarray(sideboard["mask"])[50, 50] > 127)

    def test_settle_masks_grows_the_box_to_cover_the_mask(self):
        mask = Image.fromarray(_rect_mask(100, 100, 18, 20, 80, 95).astype("uint8") * 255, mode="L")
        dets = [{"box": (0.2, 0.2, 0.8, 0.9), "mask": mask}]
        seg._settle_masks(dets, 100, 100)
        self.assertEqual(dets[0]["box"], (0.18, 0.2, 0.8, 0.95))


if __name__ == "__main__":
    unittest.main()


class ReflectionTests(unittest.TestCase):
    @staticmethod
    def det(label, box, confidence=0.5):
        # Mirrors at the confidence real ones scored in the test rooms.
        if label == "mirror" and confidence == 0.5:
            confidence = 0.77
        return {"label": label, "confidence": confidence, "box": box, "mask": None}

    def labels(self, dets):
        return [d["label"] for d in seg.drop_reflections(dets)]

    def test_the_lamp_in_room09s_floor_mirror_is_dropped(self):
        # Boxes from compare/all_rooms/room09: the real lamp stands on the desk
        # (DINO: "floor lamp"), its reflection is inside the leaning mirror.
        dets = [
            self.det("mirror", (0.0351, 0.1718, 0.1383, 0.5734)),
            self.det("desk", (0.0224, 0.6029, 0.2776, 0.8896)),
            self.det("floor lamp", (0.144, 0.4211, 0.2319, 0.6445)),
            self.det("table lamp", (0.045, 0.4416, 0.1072, 0.5657)),
        ]
        self.assertEqual(self.labels(dets), ["mirror", "desk", "floor lamp"])

    def test_a_reflection_with_no_twin_outside_stays(self):
        dets = [self.det("mirror", (0.1, 0.1, 0.4, 0.9)), self.det("armchair", (0.15, 0.5, 0.35, 0.88))]
        self.assertEqual(self.labels(dets), ["mirror", "armchair"])

    def test_what_stands_below_a_wall_mirror_stays(self):
        # A vase on the console under the mirror reaches below its bottom edge.
        dets = [
            self.det("mirror", (0.3, 0.2, 0.6, 0.5)),
            self.det("vase", (0.4, 0.4, 0.45, 0.56)),
            self.det("vase", (0.8, 0.5, 0.85, 0.6)),
        ]
        self.assertEqual(len(seg.drop_reflections(dets)), 3)

    def test_a_mirror_headboard_does_not_hide_the_bed_pillows(self):
        dets = [
            self.det("bed", (0.1, 0.3, 0.8, 0.9)),
            self.det("mirror", (0.15, 0.3, 0.75, 0.6)),
            self.det("pillow", (0.2, 0.45, 0.35, 0.58)),
            self.det("pillow", (0.5, 0.45, 0.65, 0.58)),
        ]
        self.assertEqual(len(seg.drop_reflections(dets)), 4)

    def test_bedroom_lamp_beside_weak_mirrors_stays(self):
        # compare/all_rooms/bedroom: DINO called the dark bedside lamp (0.37)
        # and the headboard (0.30) "mirror"; the lamp is real.
        dets = [
            self.det("bed", (0.071, 0.515, 0.547, 0.875), 0.81),
            self.det("mirror", (0.05, 0.517, 0.115, 0.725), 0.37),
            self.det("mirror", (0.053, 0.514, 0.206, 0.722), 0.3),
            self.det("table lamp", (0.093, 0.522, 0.117, 0.716), 0.37),
            self.det("floor lamp", (0.914, 0.416, 0.961, 0.79), 0.64),
        ]
        self.assertIn("table lamp", self.labels(dets))

    def test_a_confident_mirror_at_the_bed_head_is_the_headboard(self):
        dets = [
            self.det("bed", (0.2, 0.5, 0.8, 0.9)),
            self.det("mirror", (0.22, 0.3, 0.78, 0.6)),
            self.det("table lamp", (0.3, 0.4, 0.36, 0.55)),
            self.det("table lamp", (0.85, 0.4, 0.9, 0.55)),
        ]
        self.assertEqual(len(seg.drop_reflections(dets)), 4)

    def test_select_detections_drops_reflections(self):
        raw = [
            {"label": "mirror", "confidence": 0.77, "box": [0.0351, 0.1718, 0.1383, 0.5734]},
            {"label": "floor lamp", "confidence": 0.45, "box": [0.144, 0.4211, 0.2319, 0.6445]},
            {"label": "table lamp", "confidence": 0.39, "box": [0.045, 0.4416, 0.1072, 0.5657]},
        ]
        furniture, _ = seg.select_detections(raw, [None] * 3, 100, 100)
        self.assertEqual(sorted(d["label"] for d in furniture), ["floor lamp", "mirror"])


def img(mask):
    return Image.fromarray(mask.astype("uint8") * 255, mode="L")


class AnswerKeyFixTests(unittest.TestCase):
    """Rules added after scoring against the 24-image Roboflow answer key."""

    @staticmethod
    def det(label, box, conf=0.5, mask=None):
        return {"label": label, "confidence": conf, "box": box, "mask": mask}

    def test_curtains_and_artwork_in_a_doorway_box_stay(self):
        # DINO boxes a curtained window or a gallery wall as "doorway".
        raw = [
            {"label": "doorway", "confidence": 0.4, "box": [0.0, 0.22, 0.25, 0.74]},
            {"label": "artwork", "confidence": 0.45, "box": [0.04, 0.36, 0.16, 0.48]},
            {"label": "curtain", "confidence": 0.4, "box": [0.02, 0.25, 0.08, 0.73]},
            {"label": "armchair", "confidence": 0.4, "box": [0.10, 0.57, 0.20, 0.70]},
        ]
        furniture, _ = seg.select_detections(raw, [None] * 4, 100, 100)
        self.assertEqual(sorted(d["label"] for d in furniture), ["artwork", "curtain"])

    def test_a_vase_keeps_the_container_not_the_flowers(self):
        vessel = img(_rect_mask(100, 100, 45, 60, 55, 80))
        with_flowers = img(_rect_mask(100, 100, 35, 30, 65, 80))
        kept = seg.dedupe_detections([
            self.det("vase", (0.35, 0.30, 0.65, 0.80), 0.66, with_flowers),
            self.det("vase", (0.45, 0.60, 0.55, 0.80), 0.50, vessel),
        ])
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["box"], (0.45, 0.60, 0.55, 0.80))
        self.assertEqual(int((np.asarray(kept[0]["mask"]) > 127).sum()), 200)

    def test_a_box_over_a_gallery_wall_is_dropped_for_its_frames(self):
        group = self.det("artwork", (0.1, 0.1, 0.9, 0.5), 0.45, img(_rect_mask(100, 100, 10, 10, 90, 50)))
        a = self.det("artwork", (0.12, 0.12, 0.4, 0.45), 0.4, img(_rect_mask(100, 100, 12, 12, 40, 45)))
        b = self.det("artwork", (0.5, 0.12, 0.85, 0.45), 0.4, img(_rect_mask(100, 100, 50, 12, 85, 45)))
        kept = seg.dedupe_detections([group, a, b])
        self.assertEqual(sorted(d["box"] for d in kept), [a["box"], b["box"]])

    def test_a_bookshelf_is_not_a_group_of_its_cabinets(self):
        shelf = self.det("bookshelf", (0.1, 0.1, 0.9, 0.9), 0.5, img(_rect_mask(100, 100, 10, 10, 90, 90)))
        c1 = self.det("cabinet", (0.12, 0.6, 0.4, 0.88), 0.4, img(_rect_mask(100, 100, 12, 60, 40, 88)))
        c2 = self.det("cabinet", (0.5, 0.6, 0.85, 0.88), 0.4, img(_rect_mask(100, 100, 50, 60, 85, 88)))
        kept = seg.dedupe_detections([shelf, c1, c2])
        self.assertEqual([d["label"] for d in kept], ["bookshelf"])

    def test_low_storage_under_a_wall_tv_is_a_tv_stand(self):
        bench = self.det("bookshelf", (0.09, 0.55, 0.31, 0.66))
        seg.relabel_by_position([bench], [(0.11, 0.33, 0.28, 0.49)], 4 / 3)
        self.assertEqual((bench["label"], bench["dino_label"]), ("tv stand", "bookshelf"))

    def test_a_desk_with_a_monitor_on_it_stays(self):
        desk = self.det("console table", (0.02, 0.59, 0.32, 0.94))
        seg.relabel_by_position([desk], [(0.12, 0.45, 0.23, 0.63)], 16 / 9)
        self.assertEqual(desk["label"], "console table")

    def test_a_table_in_front_of_the_sofa_is_the_coffee_table(self):
        sofa = self.det("sofa", (0.31, 0.53, 0.71, 0.71))
        table = self.det("console table", (0.37, 0.65, 0.64, 0.78))
        seg.relabel_by_position([sofa, table], [], 4 / 3)
        self.assertEqual(table["label"], "coffee table")

    def test_a_side_table_beside_an_armchair_called_sofa_stays(self):
        chair = self.det("sofa", (0.80, 0.55, 0.95, 0.85))
        table = self.det("side table", (0.87, 0.71, 0.99, 0.96))
        seg.relabel_by_position([chair, table], [], 16 / 9)
        self.assertEqual(table["label"], "side table")

    def test_a_stool_in_front_of_the_sofa_is_an_ottoman(self):
        sofa = self.det("sofa", (0.2, 0.4, 0.8, 0.65))
        pouf = self.det("stool", (0.3, 0.6, 0.45, 0.85))
        seg.relabel_by_position([sofa, pouf], [], 4 / 3)
        self.assertEqual(pouf["label"], "ottoman")
