"""Unit tests for match_features.py and how product_matching.py uses it.

Run from backend/:  python -m unittest discover -s tests -v
No Pinecone/CLIP needed: ranking is tested on collapse_products() with fake rows.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import match_features as mf
import product_matching as pm


# ── colour similarity ───────────────────────────────────────────────────────

class ColorSimilarityTests(unittest.TestCase):
    def test_same_word_is_one_and_the_table_is_symmetric_and_bounded(self):
        for a in mf.CATALOG_COLOR_SHADES:
            self.assertEqual(mf.color_similarity(a, a), 1.0)
            for b in mf.CATALOG_COLOR_SHADES:
                self.assertEqual(mf.color_similarity(a, b), mf.color_similarity(b, a), (a, b))
                self.assertTrue(0.0 <= mf.color_similarity(a, b) <= 1.0)

    def test_neighbours_score_higher_than_unrelated_colours(self):
        # the lighting-confusable trio, and brown near beige, versus opposites
        self.assertGreater(mf.color_similarity("white", "beige"), 0.5)
        self.assertGreater(mf.color_similarity("beige", "gray"), 0.4)
        self.assertGreater(mf.color_similarity("brown", "beige"), mf.color_similarity("brown", "blue") - 0.15)
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


class ColorFitTests(unittest.TestCase):
    stool = [("brown", 0.6), ("beige", 0.34), ("orange", 0.06)]

    def test_matching_the_main_colour_beats_matching_a_minor_one(self):
        brown = mf.color_fit(self.stool, "brown")
        beige = mf.color_fit(self.stool, "beige")
        blue = mf.color_fit(self.stool, "blue")
        self.assertGreater(brown, beige)
        self.assertGreater(beige, blue)
        self.assertGreaterEqual(brown, 0.6)            # the 60% colour matches outright
        self.assertLess(blue, 0.2)

    def test_a_tiny_swatch_barely_helps(self):
        only_a_sliver = [("blue", 0.94), ("orange", 0.06)]
        self.assertLess(mf.color_fit(only_a_sliver, "orange"), 0.15)

    def test_unknown_on_either_side_means_no_opinion(self):
        self.assertIsNone(mf.color_fit(self.stool, None))
        self.assertIsNone(mf.color_fit(self.stool, ""))
        self.assertIsNone(mf.color_fit(self.stool, "chartreuse"))
        self.assertIsNone(mf.color_fit([], "brown"))

    def test_product_word_is_case_insensitive(self):
        self.assertEqual(mf.color_fit(self.stool, "Brown"), mf.color_fit(self.stool, "brown"))


# ── material ────────────────────────────────────────────────────────────────

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


class MaterialFitTests(unittest.TestCase):
    wood_fabric = [("wood", 0.9), ("fabric", 0.8)]

    def test_all_informative_materials_present_is_a_full_fit(self):
        self.assertEqual(mf.material_fit(self.wood_fabric, {"material_tags": "Fabric|Wood"}), 1.0)

    def test_partial_fit_is_weighted_by_confidence(self):
        got = mf.material_fit(self.wood_fabric, {"material_tags": "Wood|Metal"})
        self.assertAlmostEqual(got, 0.9 / 1.7)

    def test_no_overlap_is_zero(self):
        self.assertEqual(mf.material_fit(self.wood_fabric, {"material_tags": "Metal|Glass"}), 0.0)

    def test_falls_back_to_the_single_tag_when_the_list_is_missing(self):
        self.assertEqual(mf.material_fit([("wood", 1.0)], {"material_tag": "Wood"}), 1.0)

    def test_plastic_and_other_guesses_carry_no_information(self):
        self.assertIsNone(mf.material_fit([("plastic", 0.9)], {"material_tags": "Wood"}))
        self.assertIsNone(mf.material_fit([("other", 0.9)], {"material_tags": "Wood"}))   # ceramic/stone/paper
        # ...but they do not hurt the informative ones
        self.assertEqual(mf.material_fit([("wood", 0.9), ("plastic", 0.9)], {"material_tags": "Wood"}), 1.0)

    def test_products_without_usable_tags_give_no_opinion(self):
        for row in ({}, {"material_tags": ""}, {"material_tags": "Other"}, {"material_tag": "Other"}):
            self.assertIsNone(mf.material_fit([("wood", 1.0)], row), row)

    def test_nothing_to_compare_when_the_item_has_no_materials(self):
        self.assertIsNone(mf.material_fit([], {"material_tags": "Wood"}))


# ── penalty ─────────────────────────────────────────────────────────────────

class FeaturePenaltyTests(unittest.TestCase):
    def test_no_information_costs_nothing(self):
        self.assertEqual(mf.feature_penalty("sofa", None, None), 0.0)

    def test_perfect_fit_costs_nothing(self):
        self.assertEqual(mf.feature_penalty("sofa", 1.0, 1.0), 0.0)

    def test_total_mismatch_costs_at_most_the_two_maximums_times_the_type_weight(self):
        full = mf.COLOR_PENALTY_MAX + mf.MATERIAL_PENALTY_MAX
        self.assertAlmostEqual(mf.feature_penalty("sofa", 0.0, 0.0), full * 1.0)
        self.assertAlmostEqual(mf.feature_penalty("wardrobe", 0.0, 0.0), full * 0.25)
        self.assertAlmostEqual(mf.feature_penalty("coffee_table", 0.0, 0.0), full * 0.4)
        self.assertAlmostEqual(mf.feature_penalty(None, 0.0, 0.0), full * mf.DEFAULT_TYPE_WEIGHT)

    def test_seating_counts_more_than_storage(self):
        self.assertGreater(mf.feature_penalty("armchair", 0.0, None), mf.feature_penalty("cabinet", 0.0, None))

    def test_the_penalty_is_always_smaller_than_the_width_penalty_ceiling(self):
        worst = max(mf.feature_penalty(t, 0.0, 0.0) for t in list(mf.TYPE_WEIGHT) + [None])
        self.assertLess(worst, pm._SIZE_PENALTY_MAX)

    def test_every_type_weight_names_a_real_furniture_type(self):
        from app.services.furniture_types import FURNITURE_TYPES
        self.assertTrue(set(mf.TYPE_WEIGHT) <= set(FURNITURE_TYPES), set(mf.TYPE_WEIGHT) - set(FURNITURE_TYPES))


# ── ranking in product_matching ─────────────────────────────────────────────

def row(n, color=None, tags=None, width=None):
    return {"unique_id": str(n), "sku": f"sku{n}", "main_image": f"https://img/{n}.jpg",
            "color_extracted": color, "material_tags": tags, "width_cm": width}


def rank(scores_and_rows, **kw):
    matches = [{"id": str(r["unique_id"]), "score": s} for s, r in scores_and_rows]
    rows = {str(r["unique_id"]): r for _, r in scores_and_rows}
    return pm.collapse_products(matches, rows, **kw)


class RankingTests(unittest.TestCase):
    brown_item = (("brown", 1.0),)
    wood_item = (("wood", 0.9),)

    def test_without_item_features_nothing_changes(self):
        got = rank([(0.80, row(1, "blue")), (0.82, row(2, "brown"))])
        self.assertEqual([c.row["unique_id"] for c in got], ["2", "1"])
        self.assertEqual([c.rank_score for c in got], [c.score for c in got])

    def test_equal_picture_scores_are_split_by_colour(self):
        got = rank([(0.85, row(1, "blue")), (0.85, row(2, "brown"))],
                   item_colors=self.brown_item, furniture_type="sofa")
        self.assertEqual([c.row["unique_id"] for c in got], ["2", "1"])

    def test_a_clearly_better_picture_match_still_wins_over_a_better_colour(self):
        got = rank([(0.92, row(1, "blue")), (0.80, row(2, "brown"))],
                   item_colors=self.brown_item, furniture_type="sofa")
        self.assertEqual(got[0].row["unique_id"], "1")

    def test_material_breaks_ties_too(self):
        got = rank([(0.85, row(1, tags="Metal|Glass")), (0.85, row(2, tags="Wood|Plastic"))],
                   item_materials=self.wood_item, furniture_type="chair")
        self.assertEqual(got[0].row["unique_id"], "2")

    def test_score_is_never_changed_so_thresholds_and_confidence_stay_about_the_picture(self):
        got = rank([(0.80, row(1, "blue", "Metal"))], item_colors=self.brown_item,
                   item_materials=self.wood_item, furniture_type="sofa")
        self.assertEqual(got[0].score, 0.80)
        self.assertLess(got[0].rank_score, 0.80)
        self.assertEqual(pm.confidence_of(got[0].score), pm.confidence_of(0.80))

    def test_a_mismatch_never_removes_a_candidate(self):
        got = rank([(0.80, row(1, "blue", "Metal")), (0.79, row(2, "red", "Glass"))],
                   item_colors=self.brown_item, item_materials=self.wood_item, furniture_type="sofa")
        self.assertEqual(len(got), 2)

    def test_storage_types_barely_care_about_colour(self):
        sofa = rank([(0.85, row(1, "blue"))], item_colors=self.brown_item, furniture_type="sofa")[0]
        cabinet = rank([(0.85, row(1, "blue"))], item_colors=self.brown_item, furniture_type="cabinet")[0]
        self.assertGreater(cabinet.rank_score, sofa.rank_score)

    def test_unknown_product_colour_is_not_punished(self):
        got = rank([(0.85, row(1, None)), (0.85, row(2, "blue"))],
                   item_colors=self.brown_item, furniture_type="sofa")
        self.assertEqual(got[0].row["unique_id"], "1")

    def test_works_together_with_the_width_nudge(self):
        got = rank([(0.85, row(1, "brown", width=300)), (0.85, row(2, "brown", width=100))],
                   target_width_cm=100, item_colors=self.brown_item, furniture_type="sofa")
        self.assertEqual(got[0].row["unique_id"], "2")

    def test_the_card_json_reports_the_fits(self):
        cand = rank([(0.85, row(1, "brown", "Wood|Metal"))], item_colors=self.brown_item,
                    item_materials=self.wood_item, furniture_type="sofa")[0]
        fit = pm.to_product_match(cand, True)["fit"]
        self.assertEqual(fit, {"color": 1.0, "material": 1.0})
        bare = pm.to_product_match(rank([(0.85, row(2))])[0], True)["fit"]
        self.assertEqual(bare, {"color": None, "material": None})


class EndpointFeatureTests(unittest.TestCase):
    """/match-products: the new form fields reach the matcher, and a missing
    package is a clear 503 instead of a bare 500."""

    def setUp(self):
        from io import BytesIO
        from unittest.mock import patch
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from PIL import Image

        app = FastAPI()
        app.include_router(pm.router)
        self.client = TestClient(app)
        buf = BytesIO()
        Image.new("RGB", (8, 8), (200, 30, 30)).save(buf, "PNG")
        self.png = buf.getvalue()
        self.patch = patch

    def post(self, **data):
        return self.client.post("/match-products", files={"image": ("item.png", self.png, "image/png")},
                                data={"label": "armchair", **data})

    def test_colours_and_materials_are_parsed_and_forwarded(self):
        result = pm.MatchResult("armchair", "type", [], 0)
        colors = json.dumps([{"catalog_color": "brown", "share": 3}, {"catalog_color": "beige", "share": 1}])
        materials = json.dumps([{"name": "leather", "confidence": 0.8}])
        with self.patch.object(pm, "find_candidates", return_value=result) as find:
            resp = self.post(colors=colors, materials=materials)
        self.assertEqual(resp.status_code, 200, resp.text)
        kwargs = find.call_args.kwargs
        self.assertEqual(kwargs["item_colors"], (("brown", 0.75), ("beige", 0.25)))
        self.assertEqual(kwargs["item_materials"], (("fabric", 0.8),))

    def test_without_the_fields_nothing_is_forwarded(self):
        result = pm.MatchResult("armchair", "type", [], 0)
        with self.patch.object(pm, "find_candidates", return_value=result) as find:
            self.post()
        self.assertEqual(find.call_args.kwargs["item_colors"], ())
        self.assertEqual(find.call_args.kwargs["item_materials"], ())

    def test_malformed_fields_are_ignored_not_an_error(self):
        result = pm.MatchResult("armchair", "type", [], 0)
        with self.patch.object(pm, "find_candidates", return_value=result):
            self.assertEqual(self.post(colors="not json", materials="{}").status_code, 200)

    def test_a_missing_package_is_a_503_that_names_it(self):
        error = ModuleNotFoundError("No module named 'sentence_transformers'", name="sentence_transformers")
        with self.patch.object(pm, "find_candidates", side_effect=error):
            resp = self.post()
        self.assertEqual(resp.status_code, 503)
        self.assertIn("sentence_transformers", resp.json()["detail"])
        self.assertIn("pip install -r requirements.txt", resp.json()["detail"])


if __name__ == "__main__":
    unittest.main()
