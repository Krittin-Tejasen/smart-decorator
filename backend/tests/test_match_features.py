"""Unit tests for match_features.py (the colour / material filter) and how product_matching.py uses it.

Run from backend/:  python -m unittest discover -s tests -v
No network: Pinecone, CLIP and Supabase are replaced by stand-ins.
"""

import json
import sys
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import match_features as mf
import product_matching as pm


# ── colour words ────────────────────────────────────────────────────────────

class ColorSimilarityTests(unittest.TestCase):
    def test_same_word_is_one_and_the_table_is_symmetric_and_bounded(self):
        for a in mf.CATALOG_COLOR_SHADES:
            self.assertEqual(mf.color_similarity(a, a), 1.0)
            for b in mf.CATALOG_COLOR_SHADES:
                self.assertEqual(mf.color_similarity(a, b), mf.color_similarity(b, a), (a, b))
                self.assertTrue(0.0 <= mf.color_similarity(a, b) <= 1.0)

    def test_neighbours_score_higher_than_unrelated_colours(self):
        self.assertGreater(mf.color_similarity("white", "beige"), 0.5)
        self.assertGreater(mf.color_similarity("beige", "gray"), 0.4)
        self.assertLess(mf.color_similarity("green", "red"), 0.05)
        self.assertLess(mf.color_similarity("white", "black"), 0.05)

    def test_unknown_words_are_not_similar(self):
        self.assertEqual(mf.color_similarity("white", "chartreuse"), 0.0)


class ParseItemColorsTests(unittest.TestCase):
    def test_reads_the_json_the_app_sends_and_normalises_shares(self):
        raw = json.dumps([{"catalog_color": "brown", "share": 0.6}, {"catalog_color": "beige", "share": 0.2}])
        got = mf.parse_item_colors(raw)
        self.assertEqual([w for w, _ in got], ["brown", "beige"])
        self.assertAlmostEqual(sum(s for _, s in got), 1.0)
        self.assertAlmostEqual(got[0][1], 0.75)

    def test_same_word_is_merged_so_three_browns_count_as_one(self):
        raw = [{"catalog_color": "brown", "share": 0.38}, {"catalog_color": "brown", "share": 0.34},
               {"catalog_color": "brown", "share": 0.28}]
        self.assertEqual(mf.parse_item_colors(raw), [("brown", 1.0)])

    def test_words_outside_the_catalog_and_bad_values_are_ignored(self):
        raw = json.dumps([{"catalog_color": "terracotta", "share": 0.5}, {"catalog_color": "gray", "share": "x"},
                          {"catalog_color": "gray", "share": -1}, "gray", {"catalog_color": "Green", "share": 1}])
        self.assertEqual(mf.parse_item_colors(raw), [("green", 1.0)])

    def test_garbage_input_gives_nothing(self):
        for junk in (None, "", "not json", "{}", "5", 7, [], [None]):
            self.assertEqual(mf.parse_item_colors(junk), [], junk)


class ColourWordsTests(unittest.TestCase):
    def test_the_item_colour_and_its_close_neighbours_are_kept(self):
        words = mf.colour_words([("brown", 1.0)])
        self.assertIn("brown", words)
        self.assertIn("black", words)            # dark brown is close to black
        self.assertNotIn("blue", words)
        self.assertNotIn("white", words)

    def test_light_neutrals_keep_each_other(self):
        words = mf.colour_words([("white", 1.0)])
        self.assertTrue({"white", "beige", "gray"} <= set(words))
        self.assertNotIn("black", words)

    def test_a_minor_colour_is_not_one_of_the_items_colours(self):
        words = mf.colour_words([("brown", 0.9), ("blue", 0.1)])    # blue is only 10% of the pixels
        self.assertIn("brown", words)
        self.assertNotIn("blue", words)

    def test_two_real_colours_are_both_kept(self):
        words = mf.colour_words([("brown", 0.64), ("beige", 0.36)])
        self.assertTrue({"brown", "beige"} <= set(words))

    def test_with_no_dominant_colour_the_main_one_is_used(self):
        six = [("red", 0.19), ("blue", 0.18), ("green", 0.17), ("pink", 0.16), ("orange", 0.15), ("purple", 0.15)]
        words = mf.colour_words(six)
        self.assertIn("red", words)
        self.assertNotIn("purple", words)

    def test_result_is_sorted_and_empty_without_colours(self):
        words = mf.colour_words([("brown", 1.0)])
        self.assertEqual(words, sorted(words))
        self.assertEqual(mf.colour_words([]), [])


# ── materials ───────────────────────────────────────────────────────────────

class ParseItemMaterialsTests(unittest.TestCase):
    def test_fine_words_fold_into_catalog_groups(self):
        raw = json.dumps([{"name": "leather", "confidence": 0.7}, {"name": "rattan", "confidence": 0.8},
                          {"name": "ceramic", "confidence": 0.9}])
        self.assertEqual(dict(mf.parse_item_materials(raw)), {"fabric": 0.7, "wood": 0.8, "other": 0.9})

    def test_same_group_keeps_the_higher_confidence(self):
        raw = [{"name": "fabric", "confidence": 0.4}, {"name": "leather", "confidence": 0.9}]
        self.assertEqual(mf.parse_item_materials(raw), [("fabric", 0.9)])

    def test_unknown_words_and_garbage_are_ignored_and_confidence_is_clamped(self):
        raw = [{"name": "unobtainium", "confidence": 1}, {"name": "wood", "confidence": 7}, {"name": "metal"}, 3]
        self.assertEqual(dict(mf.parse_item_materials(raw)), {"wood": 1.0, "metal": 0.5})
        for junk in (None, "", "nope", "{}", 9):
            self.assertEqual(mf.parse_item_materials(junk), [])


class MaterialFlagsTests(unittest.TestCase):
    def test_filterable_groups_map_to_the_catalog_flags_in_confidence_order(self):
        mats = [("wood", 0.9), ("other", 0.9), ("plastic", 0.5), ("fabric", 0.8), ("metal", 0.1), ("glass", 0.05)]
        self.assertEqual(mf.material_flags(mats), ["is_wood", "is_fabric", "is_metal", "is_glass"])

    def test_plastic_and_other_cannot_be_filtered(self):
        self.assertEqual(mf.material_flags([("plastic", 0.9), ("other", 0.8)]), [])
        self.assertEqual(mf.material_flags([]), [])

    def test_every_flag_is_a_pinecone_metadata_boolean_the_ingest_writes(self):
        self.assertEqual(set(mf.MATERIAL_FLAG.values()), {"is_wood", "is_fabric", "is_metal", "is_glass"})


# ── the metadata filter ─────────────────────────────────────────────────────

class MetadataFilterTests(unittest.TestCase):
    brown = [("brown", 1.0)]
    wood_fabric = [("wood", 0.9), ("fabric", 0.8)]

    def test_colour_only(self):
        f = mf.metadata_filter(self.brown, None)
        self.assertEqual(set(f), {"color"})
        self.assertIn("brown", f["color"]["$in"])

    def test_material_only_any_of_the_items_materials(self):
        f = mf.metadata_filter(None, self.wood_fabric)
        self.assertEqual(f, {"$or": [{"is_wood": {"$eq": True}}, {"is_fabric": {"$eq": True}}]})

    def test_both_are_combined(self):
        f = mf.metadata_filter(self.brown, self.wood_fabric)
        self.assertEqual(set(f), {"color", "$or"})

    def test_nothing_to_filter_on_gives_an_empty_filter(self):
        self.assertEqual(mf.metadata_filter(), {})
        self.assertEqual(mf.metadata_filter([], []), {})
        self.assertEqual(mf.metadata_filter(None, [("plastic", 1.0), ("other", 1.0)]), {})

    def test_the_filter_is_json_serialisable_for_pinecone(self):
        json.dumps(mf.metadata_filter(self.brown, self.wood_fabric))


class FilterLadderTests(unittest.TestCase):
    extra = {"color": {"$in": ["brown", "black"]}, "$or": [{"is_wood": {"$eq": True}}]}

    def test_without_an_extra_filter_the_levels_are_unchanged(self):
        levels = pm.build_filter_ladder("armchair", 100, 900)
        self.assertEqual([lv.name for lv in levels], ["type"])
        self.assertEqual(levels[0].filter, {"thb_price": {"$gte": 100.0, "$lte": 900.0}, "furniture_type": {"$eq": "armchair"}})

    def test_the_extra_filter_is_part_of_the_type_level(self):
        (level,) = pm.build_filter_ladder("armchair", extra_filter=self.extra)
        self.assertEqual(level.filter["furniture_type"], {"$eq": "armchair"})
        self.assertEqual(level.filter["color"], {"$in": ["brown", "black"]})
        self.assertEqual(level.filter["$or"], [{"is_wood": {"$eq": True}}])

    def test_it_is_in_every_level_never_relaxed_away(self):
        levels = pm.build_filter_ladder("armchair", in_stock_only=True, extra_filter=self.extra)
        self.assertEqual([lv.name for lv in levels], ["type+stock", "type"])
        for level in levels:
            self.assertIn("color", level.filter)
            self.assertIn("$or", level.filter)

    def test_it_also_applies_to_the_image_only_levels(self):
        levels = pm.build_filter_ladder(None, in_stock_only=True, extra_filter=self.extra)
        self.assertEqual([lv.name for lv in levels], ["stock", "image_only"])
        self.assertTrue(all("color" in lv.filter for lv in levels))

    def test_the_callers_dict_is_not_modified(self):
        extra = {"color": {"$in": ["brown"]}}
        pm.build_filter_ladder("armchair", 1, 2, extra_filter=extra)
        self.assertEqual(extra, {"color": {"$in": ["brown"]}})


class FindCandidatesFilterTests(unittest.TestCase):
    """The filter reaches Pinecone, together with the type and price conditions."""

    def run_match(self, **kwargs):
        sent = []

        def query_similar(vector, top_k, metadata_filter, dedupe):
            sent.append(metadata_filter)
            return []

        with patch("app.services.embedding.encode_image", return_value=[0.0] * 512), \
                patch("app.services.pinecone_client.query_similar", side_effect=query_similar), \
                patch("app.services.product_repo.fetch_products_by_ids", return_value={}):
            pm.find_candidates(Image.new("RGB", (8, 8)), "armchair", **kwargs)
        return sent

    def test_the_colour_condition_is_sent_with_the_type(self):
        sent = self.run_match(max_price=5000, extra_filter={"color": {"$in": ["brown"]}})
        self.assertEqual(len(sent), 1)
        self.assertEqual(sent[0]["color"], {"$in": ["brown"]})
        self.assertEqual(sent[0]["furniture_type"], {"$eq": "armchair"})
        self.assertEqual(sent[0]["thb_price"], {"$lte": 5000.0})

    def test_without_one_nothing_extra_is_sent(self):
        sent = self.run_match()
        self.assertEqual(set(sent[0]), {"furniture_type"})

    def test_an_empty_answer_is_not_retried_without_the_filter(self):
        sent = self.run_match(extra_filter={"color": {"$in": ["brown"]}})
        self.assertEqual(len(sent), 1)           # one query: nothing is relaxed


# ── the endpoint ────────────────────────────────────────────────────────────

class EndpointFilterTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(pm.router)
        self.client = TestClient(app)
        buf = BytesIO()
        Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, "PNG")
        self.png = buf.getvalue()
        self.empty = pm.MatchResult("armchair", "type", [], 0)

    def post(self, **data):
        return self.client.post("/match-products", files={"image": ("item.png", self.png, "image/png")},
                                data={"label": "armchair", **data})

    def sent_filter(self, **data):
        with patch.object(pm, "find_candidates", return_value=self.empty) as find:
            resp = self.post(**data)
        self.assertEqual(resp.status_code, 200, resp.text)
        return find.call_args.kwargs["extra_filter"], resp.json()

    def test_colours_become_a_colour_filter(self):
        colors = json.dumps([{"catalog_color": "brown", "share": 1}])
        extra, body = self.sent_filter(colors=colors)
        self.assertIn("brown", extra["color"]["$in"])
        self.assertNotIn("$or", extra)
        self.assertEqual(body["filters"], ["color"])

    def test_materials_become_a_material_filter(self):
        materials = json.dumps([{"name": "leather", "confidence": 0.8}, {"name": "wood", "confidence": 0.5}])
        extra, body = self.sent_filter(materials=materials)
        self.assertEqual(extra, {"$or": [{"is_fabric": {"$eq": True}}, {"is_wood": {"$eq": True}}]})
        self.assertEqual(body["filters"], ["material"])

    def test_both_filters_together(self):
        colors = json.dumps([{"catalog_color": "beige", "share": 1}])
        materials = json.dumps([{"name": "metal", "confidence": 0.9}])
        extra, body = self.sent_filter(colors=colors, materials=materials)
        self.assertEqual(set(extra), {"color", "$or"})
        self.assertEqual(body["filters"], ["color", "material"])

    def test_without_the_fields_there_is_no_filter(self):
        extra, body = self.sent_filter()
        self.assertIsNone(extra)
        self.assertEqual(body["filters"], [])

    def test_a_filter_with_nothing_usable_in_it_filters_nothing(self):
        for data in ({"colors": "not json"}, {"materials": "{}"}, {"materials": json.dumps([{"name": "plastic"}])}):
            extra, body = self.sent_filter(**data)
            self.assertIsNone(extra, data)
            self.assertEqual(body["filters"], [], data)

    def test_matches_come_back_without_extra_fields_in_each_card(self):
        candidate = pm.Candidate(row={"unique_id": "a", "main_title": "POANG", "sku": "s", "main_image": "i"}, score=0.9)
        result = pm.MatchResult("armchair", "type", [candidate], 0)
        with patch.object(pm, "find_candidates", return_value=result), patch.object(pm, "check_link", return_value=True):
            body = self.post().json()
        self.assertNotIn("fit", body["matches"][0])

    def test_a_missing_package_is_a_503_that_names_it(self):
        error = ModuleNotFoundError("No module named 'sentence_transformers'", name="sentence_transformers")
        with patch.object(pm, "find_candidates", side_effect=error):
            resp = self.post()
        self.assertEqual(resp.status_code, 503)
        self.assertIn("sentence_transformers", resp.json()["detail"])
        self.assertIn("pip install -r requirements.txt", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
