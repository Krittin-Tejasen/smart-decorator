"""Tests for product matching: the furniture-type vocabulary shared by the
catalog and the queries, and the /match-products ranking, fallback, threshold
and link-check logic.

Run from backend/:  python -m unittest discover -s tests -v

CLIP, Pinecone, Supabase and the shop's web pages are all patched out, so
these run offline, need no API keys, and never touch the real index.
"""

import sys
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import product_matching as pm
from app.services import catalog_types as ct
from app.services import furniture_types as ft
from segmentation import _VOCAB, is_furniture_label


def png_bytes() -> bytes:
    buf = BytesIO()
    Image.new("RGBA", (16, 16), (120, 80, 40, 255)).save(buf, format="PNG")
    return buf.getvalue()


def row(uid, title="Chair", image_id="0001_pe1", sku="001", **extra):
    return {
        "unique_id": uid,
        "main_title": title,
        "main_image": f"https://www.ikea.com/images/x__{image_id}_s5.jpg",
        "sku": sku,
        "thb_price": 1990,
        "url": f"https://www.ikea.com/p/{uid}",
        **extra,
    }


class ClassifyTitleTests(unittest.TestCase):
    def test_measured_types_keep_their_patterns(self):
        self.assertEqual(ft.classify_title("EKTORP 3-seat sofa, Hallarp grey"), "sofa")
        self.assertEqual(ft.classify_title("MALM bed frame, high, white"), "bed")
        self.assertEqual(ft.classify_title("STRANDMON fauteuil à oreilles"), "armchair")

    def test_specific_names_win_over_the_words_they_contain(self):
        self.assertEqual(ft.classify_title("LACK TV bench, black-brown"), "tv_bench")
        # Not a plain bench; its image group's English rows make it a tv_bench.
        self.assertIsNone(ft.classify_title("BESTÅ banc TV avec tiroirs"))
        self.assertEqual(ft.classify_title("HEMNES bedside table, white"), "nightstand")
        self.assertEqual(ft.classify_title("LACK coffee table, white"), "coffee_table")

    def test_accessories_are_not_furniture(self):
        self.assertIsNone(ft.classify_title("EKTORP cover for 3-seat sofa"))
        self.assertIsNone(ft.classify_title("PAX wardrobe door hinge"))
        self.assertIsNone(ft.classify_title(""))

    def test_found_by_real_queries(self):
        # PAX wardrobes titled in Italian were typed "mirror" via "specchio".
        self.assertEqual(ft.classify_title("PAX combinazione di guardaroba, bianco specchio"), "wardrobe")
        # A mirror door is a part, not a mirror.
        self.assertIsNone(ft.classify_title("AULI 4 panneaux pour porte coulissante, miroir"))
        self.assertEqual(ft.classify_title("LETTAN mirror, 100x95 cm"), "mirror")
        # Kitchen tall cabinets are titled "armoire" in French.
        self.assertIsNone(ft.classify_title("METOD armoire réfrigérateur/congél+3ptes, blanc"))
        # The catalog's only "tapis" are drawer mats.
        self.assertIsNone(ft.classify_title("KOMPLEMENT plateau couliss av tapis de tiroir, blanc"))

    def test_added_types(self):
        self.assertEqual(ft.classify_title("STOCKHOLM rug, flatwoven"), "rug")
        self.assertEqual(ft.classify_title("HEMNES chest of 8 drawers"), "dresser")
        self.assertEqual(ft.classify_title("LINDBYN mirror, black"), "mirror")


class LabelToTypeTests(unittest.TestCase):
    def test_every_segmentation_furniture_label_is_handled(self):
        # A new word in segmentation's prompt must get a decision here: either a
        # type, or listed below as one the catalog can't be filtered by.
        untyped = {"decoration"}
        for label in _VOCAB:
            if not is_furniture_label(label) or label in untyped:
                continue
            with self.subTest(label=label):
                self.assertIn(ft.label_to_type(label), ft.FURNITURE_TYPES)

    def test_gemini_free_text_labels(self):
        self.assertEqual(ft.label_to_type("Floating_Planter"), "plant_pot")
        self.assertEqual(ft.label_to_type("throw blanket"), None)
        self.assertEqual(ft.label_to_type(None), None)


class GroupTypeTests(unittest.TestCase):
    def test_a_group_takes_the_majority_type_of_its_titles(self):
        rows = [
            row("1", "MALM bed frame, white", image_id="0749130_pe745499"),
            row("2", "MALM Bettgestell, weiß", image_id="0749130_pe745499"),  # German: no pattern
            row("3", "MALM lit, blanc", image_id="0749130_pe745499"),
            row("4", "Something else", image_id="9999_pe9"),
        ]
        self.assertEqual(ft.types_by_unique_id(rows), {"1": "bed", "2": "bed", "3": "bed"})

    def test_rows_without_an_image_id_group_by_sku(self):
        self.assertEqual(ft.image_group_key("https://x/no-id.jpg", "S299.316.00"), "sku:29931600")


class CatalogTypesTests(unittest.TestCase):
    """app.services.catalog_types: the multilingual reader that fills the groups
    the title regexes cannot name."""

    def test_names_products_whose_titles_the_regexes_miss(self):
        group = ct.classify_group
        self.assertEqual(group(["STORSLINGA led-kroonluchter, 8 armen, zwart/wit, 70 cm"]), "lamp")
        self.assertEqual(group(["PILSKOTT Standleuchte, LED, smart schwarz"]), "lamp")
        self.assertEqual(group(["PAX / ÅHEIM Schrankkombination, weiß/Spiegelglas"]), "wardrobe")
        self.assertEqual(group(["BESTÅ Aufbewahrung mit Türen, schwarzbraun"]), "cabinet")

    def test_kitchen_cabinets_and_parts_are_not_furniture_for_a_room(self):
        group = ct.classify_group
        self.assertEqual(group(["METOD armoire réfrigérateur/congél+3ptes, blanc"]), "kitchen_cabinet")
        self.assertEqual(group(["METOD base cab f sink+2 fronts/2 drawers, white"]), "kitchen_cabinet")
        self.assertEqual(group(["VOXTORP door, high-gloss dark grey-brown, 60x40 cm"]), "part")
        self.assertEqual(group(["LERHYTTAN light grey, drawer front, 80x20 cm"]), "part")
        self.assertEqual(group(["HAVBÄCK bathroom vanity with sink & faucet, white"]), "bathroom")

    def test_a_pair_of_doors_is_not_a_wardrobe_but_a_wardrobe_with_doors_is(self):
        self.assertEqual(ct.classify_group(["HASVIK pair of sliding doors, grey-beige"]), "part")
        self.assertEqual(ct.classify_group(["PAX wardrobe with sliding doors, white"]), "wardrobe")

    def test_a_bedside_table_is_not_a_side_table_and_a_sofa_bed_is_not_a_bed(self):
        self.assertEqual(ct.classify_group(["BRIMNES Bedside table - white"]), "nightstand")
        self.assertEqual(ct.classify_group(["FRIHETEN Three-seat sofa-bed - Bomstad black"]), "sofa")

    def test_led_strips_are_accessories_but_led_lamps_are_lamps(self):
        self.assertEqual(ct.classify_group(["MITTLED LED kitchen lighting strip, dimmable"]), "lighting_accessory")
        self.assertEqual(ct.classify_group(["UPPLYST LED wall lamp, cloud white"]), "lamp")

    def test_titles_that_say_nothing_stay_untyped(self):
        self.assertIsNone(ct.classify_group(["Illustration with measurements for the product"]))
        self.assertIsNone(ct.classify_group([None, ""]))

    def test_colour_first_titles_use_the_product_noun(self):
        self.assertEqual(ct.noun_of("LERHYTTAN light grey, drawer front, 80x20 cm"), "drawer front")

    def test_kitchen_cabinet_wins_over_a_regex_type_but_other_types_do_not(self):
        rows = [row("1", "METOD armoire réfrigérateur/congél+3ptes, blanc", image_id="1_pe1"),
                row("2", "MALM bed frame, white", image_id="2_pe2")]
        self.assertEqual(ft.types_by_unique_id(rows), {"1": "kitchen_cabinet", "2": "bed"})

    def test_a_group_the_regexes_cannot_name_gets_the_readers_type(self):
        rows = [row("1", "STORSLINGA led-kroonluchter, 8 armen, zwart/wit", image_id="3_pe3")]
        self.assertEqual(ft.types_by_unique_id(rows), {"1": "lamp"})


class FilterLadderTests(unittest.TestCase):
    def test_typed_label_is_searched_only_within_its_type_and_keeps_the_budget(self):
        levels = pm.build_filter_ladder("sofa", min_price=1000, max_price=5000)
        self.assertEqual([l.name for l in levels], ["type"])
        self.assertEqual(levels[0].filter["furniture_type"], {"$eq": "sofa"})
        self.assertEqual(levels[0].filter["thb_price"], {"$gte": 1000.0, "$lte": 5000.0})

    def test_untyped_label_without_budget_is_unfiltered(self):
        levels = pm.build_filter_ladder(None)
        self.assertEqual([(l.name, l.filter) for l in levels], [("image_only", None)])

    def test_in_stock_is_the_strictest_level(self):
        names = [l.name for l in pm.build_filter_ladder("bed", in_stock_only=True)]
        self.assertEqual(names, ["type+stock", "type"])
        self.assertEqual([l.name for l in pm.build_filter_ladder(None, in_stock_only=True)], ["stock", "image_only"])


class RankingTests(unittest.TestCase):
    def test_size_is_free_within_tolerance_and_capped_beyond(self):
        self.assertEqual(pm.size_penalty(200, 230), 0.0)
        self.assertAlmostEqual(pm.size_penalty(200, 260), 0.05)
        self.assertEqual(pm.size_penalty(200, 1000), pm._SIZE_PENALTY_MAX)
        self.assertEqual(pm.size_penalty(None, 120), 0.0)
        self.assertEqual(pm.size_penalty(200, ""), 0.0)

    def test_one_card_per_product_preferring_the_english_listing(self):
        rows = {
            "fr": row("fr", image_id="1_pe1", country_code="fr"),
            "gb": row("gb", image_id="1_pe1", country_code="gb"),
            "other": row("other", image_id="2_pe2"),
        }
        matches = [{"id": "fr", "score": 0.9101}, {"id": "gb", "score": 0.9099}, {"id": "other", "score": 0.8}]
        result = pm.collapse_products(matches, rows)
        self.assertEqual([c.row["unique_id"] for c in result], ["gb", "other"])

    def test_ids_missing_from_the_catalog_are_skipped(self):
        self.assertEqual(pm.collapse_products([{"id": "gone", "score": 0.99}], {}), [])

    def test_size_reorders_but_reports_the_image_score(self):
        rows = {"wide": row("wide", image_id="1_pe1", width_cm=300), "fits": row("fits", image_id="2_pe2", width_cm=200)}
        matches = [{"id": "wide", "score": 0.90}, {"id": "fits", "score": 0.88}]
        result = pm.collapse_products(matches, rows, target_width_cm=200)
        self.assertEqual([c.row["unique_id"] for c in result], ["fits", "wide"])
        self.assertEqual(result[1].score, 0.90)


def fake_pinecone(responses_by_level):
    """query_similar stand-in returning canned matches per filter."""
    calls = []

    def query_similar(vector, top_k, metadata_filter, dedupe):
        calls.append(metadata_filter)
        typed = bool(metadata_filter and "furniture_type" in metadata_filter)
        return responses_by_level["type" if typed else "image_only"]

    return query_similar, calls


class FindCandidatesTests(unittest.TestCase):
    CATALOG = {
        "a": row("a", "POÄNG armchair", image_id="1_pe1"),
        "b": row("b", "STRANDMON armchair", image_id="2_pe2"),
        "c": row("c", "LACK table", image_id="3_pe3"),
    }

    def run_match(self, responses, **kwargs):
        query, calls = fake_pinecone(responses)
        with patch("app.services.embedding.encode_image", return_value=[0.0] * 512), \
                patch("app.services.pinecone_client.query_similar", side_effect=query), \
                patch("app.services.product_repo.fetch_products_by_ids",
                      side_effect=lambda ids: {i: self.CATALOG[i] for i in ids if i in self.CATALOG}):
            result = pm.find_candidates(Image.new("RGB", (8, 8)), "armchair", min_score=0.7, **kwargs)
        return result, calls

    def test_typed_results_above_threshold_are_used(self):
        result, calls = self.run_match({
            "type": [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.65}],
            "image_only": [{"id": "c", "score": 0.99}],
        })
        self.assertEqual(result.furniture_type, "armchair")
        self.assertEqual(result.filter_level, "type")
        self.assertEqual([c.row["unique_id"] for c in result.candidates], ["a"])
        self.assertEqual(result.below_threshold, 1)
        self.assertEqual(len(calls), 1)

    def test_nothing_of_that_type_is_not_padded_with_other_types(self):
        # The catalog has no rugs: say so instead of showing drawer fronts.
        result, calls = self.run_match({"type": [], "image_only": [{"id": "c", "score": 0.99}]})
        self.assertEqual(result.candidates, [])
        self.assertEqual(len(calls), 1)
        self.assertTrue(all("furniture_type" in (f or {}) for f in calls))

    def test_nothing_confident_returns_no_candidates(self):
        result, _ = self.run_match({"type": [{"id": "a", "score": 0.5}], "image_only": [{"id": "c", "score": 0.6}]})
        self.assertEqual(result.candidates, [])
        self.assertEqual(result.below_threshold, 1)


class ConfidenceTests(unittest.TestCase):
    def test_bands(self):
        self.assertEqual([pm.confidence_of(x) for x in (0.84, 0.75, 0.70, 0.65, 0.60)], ["high", "high", "medium", "medium", "low"])

    def test_a_right_but_low_scoring_match_is_shown_with_a_low_label(self):
        # A correct table lamp scores ~0.67 and a correct bed ~0.84: one cut-off
        # cannot serve both, so the match is shown and labelled, not hidden.
        card = pm.to_product_match(pm.Candidate(row=row("a", "FLOTTILJ desk lamp"), score=0.62), True)
        self.assertEqual(card["confidence"], "low")

    def test_the_floor_still_drops_unrelated_products(self):
        self.assertLess(pm.settings.match_min_score, pm.settings.match_medium_score)


class LinkCheckTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        pm._link_cache.clear()

    async def check(self, handler):
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await pm.check_link(client, "https://www.ikea.com/p/x")

    async def test_status_codes(self):
        self.assertIs(await self.check(lambda r: httpx.Response(200)), True)
        pm._link_cache.clear()
        self.assertIs(await self.check(lambda r: httpx.Response(404)), False)
        pm._link_cache.clear()
        # Shops often refuse automated HEAD requests; that isn't a dead link.
        self.assertIsNone(await self.check(lambda r: httpx.Response(403)))

    async def test_network_errors_are_unknown_not_dead(self):
        def boom(request):
            raise httpx.ConnectError("offline")

        self.assertIsNone(await self.check(boom))

    async def test_dead_links_are_replaced_by_the_next_candidate(self):
        candidates = [pm.Candidate(row=row(str(i), image_id=f"{i}_pe{i}"), score=0.9 - i / 100) for i in range(4)]
        dead = {"https://www.ikea.com/p/0"}

        async def fake_check(client, url):
            return url not in dead

        with patch.object(pm, "check_link", side_effect=fake_check):
            picked = await pm.pick_with_live_links(candidates, top_k=2)
        self.assertEqual([p["id"] for p in picked], ["1", "2"])


class EndpointTests(unittest.TestCase):
    def setUp(self):
        app = FastAPI()
        app.include_router(pm.router)
        self.client = TestClient(app)

    def post(self, **data):
        return self.client.post(
            "/match-products",
            files={"image": ("item.png", png_bytes(), "image/png")},
            data={"label": "armchair", **data},
        )

    def test_returns_product_match_json(self):
        candidate = pm.Candidate(row=row("a", "POÄNG armchair", width_cm="68", in_stock=True), score=0.912345)
        result = pm.MatchResult("armchair", "type", [candidate], 3)
        with patch.object(pm, "find_candidates", return_value=result) as find, \
                patch.object(pm, "check_link", return_value=True):
            resp = self.post(min_price="500", max_price="5000", top_k="3")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["furniture_type"], body["filter_level"], body["below_threshold"]), ("armchair", "type", 3))
        match = body["matches"][0]
        self.assertEqual(match["name"], "POÄNG armchair")
        self.assertEqual(match["width_cm"], 68.0)
        self.assertEqual(match["score"], 0.9123)
        self.assertIs(match["link_ok"], True)
        self.assertEqual(find.call_args.kwargs["max_price"], 5000.0)

    def test_inverted_price_range_is_rejected(self):
        self.assertEqual(self.post(min_price="5000", max_price="100").status_code, 422)

    def test_missing_configuration_is_a_503(self):
        with patch.object(pm, "find_candidates", side_effect=RuntimeError("PINECONE_API_KEY is not configured.")):
            resp = self.post()
        self.assertEqual(resp.status_code, 503)

    def test_a_label_with_no_furniture_type_is_not_searched_by_picture(self):
        with patch.object(pm, "find_candidates") as find:
            resp = self.post(label="window")
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual((body["matches"], body["needs_type"], body["filter_level"]), ([], True, "no_type"))
        self.assertIn("lamp", body["available_types"])
        find.assert_not_called()

    def test_a_type_chosen_by_the_user_is_used_instead_of_the_label(self):
        result = pm.MatchResult("lamp", "type", [], 0)
        with patch.object(pm, "find_candidates", return_value=result) as find:
            resp = self.post(label="window", furniture_type="lamp")
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(find.call_args.kwargs["furniture_type"], "lamp")
        self.assertIs(resp.json()["needs_type"], False)

    def test_an_unknown_chosen_type_is_a_422(self):
        self.assertEqual(self.post(furniture_type="spaceship").status_code, 422)

    def test_non_image_upload_is_a_400(self):
        resp = self.client.post(
            "/match-products", files={"image": ("x.png", b"not an image", "image/png")}, data={"label": "sofa"}
        )
        self.assertEqual(resp.status_code, 400)



if __name__ == "__main__":
    unittest.main()
