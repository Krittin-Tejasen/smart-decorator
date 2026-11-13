"""The pure parts of the generated-room evaluation: test-case choice, room plans,
tying detections to answers, and the metrics."""

import random
import sys
import tempfile
import unittest
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.room_eval import (
    app_prompt,
    composer_text,
    compute_metrics,
    critic_loop,
    parse_composer_choice,
    judge_items,
    product_clause,
    reference_board,
    reference_kind,
    review_items,
    room_prompt,
    valid_judgment,
    valid_review,
    write_json,
    write_report,
)
from scripts.room_eval_metrics import (
    ROOM_TEMPLATES,
    bootstrap_ci,
    calibration_points,
    choose_threshold,
    canonical_sku,
    detected_only,
    detections_of,
    equivalent_rank,
    human_judgments,
    one_per_kind,
    rank_by_similarity,
    faithful_only,
    is_usable_product,
    item_outcomes,
    label_precision,
    plan_rooms,
    precision_curve,
    rank_of,
    same_line_rank,
    split_rooms,
    summarize,
    threshold_grid,
)

ALL_TYPES = {t for types in ROOM_TEMPLATES.values() for t in types}


def product_row(title="MALM bed frame, white", country="gb", source="title", image="https://x/a.jpg", furniture_type="bed"):
    return {"main_title": title, "country_code": country, "type_source": source, "main_image": image, "furniture_type": furniture_type}


def result(group, score=0.8, title="X"):
    return {"group": group, "score": score, "title": title}


def room(room_id="room001", products=None, detections=None):
    """A room with one bed; its detections are tied to the bed unless they say otherwise."""
    return {
        "id": room_id,
        "kind": "bedroom",
        "products": products or [{"type": "bed", "group": "img:1", "title": "MALM bed"}],
        "detections": [{"group": "img:1", **d} for d in detections or []],
    }


class UsableProductTests(unittest.TestCase):
    def test_a_titled_english_listing_is_usable(self):
        self.assertTrue(is_usable_product(product_row()))

    def test_other_countries_category_types_and_missing_pictures_are_not(self):
        self.assertFalse(is_usable_product(product_row(country="de")))
        self.assertFalse(is_usable_product(product_row(source="category")))
        self.assertFalse(is_usable_product(product_row(image=None)))

    def test_parts_outdoor_pieces_and_sets_are_not(self):
        for furniture_type, title in (
            ("curtain", "BEKRÄFTA curtain rod set, black"),
            ("artwork", "MÅLERÅS Bamboo Picture Ledge, 75cm"),
            ("stool", "SEGERÖN bar stool, outdoor, white"),
            ("dining_table", "GRÅSALA / LIDÅS table and 2 chairs, grey"),
            ("coffee_table", "SUNDVIK Children's coffee table - white"),
        ):
            self.assertFalse(is_usable_product(product_row(title=title, furniture_type=furniture_type)), title)

    def test_the_title_must_really_name_the_type(self):
        for furniture_type, title in (
            ("bed", "INNERLIG Spring mattress for extendable bed"),
            ("bed", "GULLIVER Crib - white"),
            ("bed", "MALM Under Bed Storage Drawers for High Bed Frame"),
            ("bench", "BROR Work bench - black"),
            ("dining_table", "IDÅSEN Sit/stand underframe for table top"),
            ("bookshelf", "BERGSHULT / KROKSHULT Estante de pared"),
            ("lamp", "DYKARKLOCKA / HAVSDJUP Lámpara colgante"),
            ("desk", "LAGKAPTEN / ALEX Escritorio - blanco"),
        ):
            self.assertFalse(is_usable_product(product_row(title=title, furniture_type=furniture_type)), title)

    def test_real_products_of_each_kind_stay(self):
        for furniture_type, title in (
            ("bed", "HEMNES bed frame with 2 storage boxes"),
            ("dining_table", "RÖNNINGE extendable table, birch"),
            ("nightstand", "IDANÄS Nightstand - white"),
            ("sofa", "BACKSÄLEN loveseat, Hallarp gray"),
            ("tv_bench", "LOMMARP TV unit, black"),
            ("bookshelf", "KALLAX Shelf unit - black-brown"),
            ("dresser", "MALM chest of 2 drawers, white"),
        ):
            self.assertTrue(is_usable_product(product_row(title=title, furniture_type=furniture_type)), title)

    def test_a_type_without_a_rule_is_not_usable(self):
        self.assertFalse(is_usable_product(product_row(furniture_type="kitchen_cabinet")))


class PlanRoomsTests(unittest.TestCase):
    def test_every_type_in_a_room_differs_and_the_anchor_comes_first(self):
        for planned in plan_rooms(40, ALL_TYPES, random.Random(1)):
            self.assertEqual(len(planned["types"]), 4)
            self.assertEqual(len(set(planned["types"])), 4)
            self.assertEqual(planned["types"][0], ROOM_TEMPLATES[planned["kind"]][0])

    def test_the_same_seed_gives_the_same_plan(self):
        self.assertEqual(plan_rooms(20, ALL_TYPES, random.Random(7)), plan_rooms(20, ALL_TYPES, random.Random(7)))

    def test_types_are_spread_evenly(self):
        used = Counter(t for planned in plan_rooms(120, ALL_TYPES, random.Random(1)) for t in planned["types"])
        self.assertEqual(set(used), ALL_TYPES)
        self.assertGreaterEqual(min(used.values()), 15)

    def test_a_missing_type_is_never_planned(self):
        rooms = plan_rooms(20, ALL_TYPES - {"mirror"}, random.Random(1))
        self.assertFalse(any("mirror" in r["types"] for r in rooms))

    def test_a_template_without_its_anchor_is_skipped(self):
        rooms = plan_rooms(8, ALL_TYPES - {"bed"}, random.Random(1))
        self.assertFalse(any(r["kind"] == "bedroom" for r in rooms))

    def test_given_kinds_are_kept_in_order(self):
        kinds = ["bedroom", "living room", "bedroom"]
        rooms = plan_rooms(3, ALL_TYPES, random.Random(1), kinds=kinds)
        self.assertEqual([r["kind"] for r in rooms], kinds)
        self.assertEqual(rooms[0]["types"][0], "bed")

    def test_a_kind_that_cant_be_filled_is_an_error(self):
        with self.assertRaises(ValueError):
            plan_rooms(1, ALL_TYPES - {"bed"}, random.Random(1), kinds=["bedroom"])

    def test_no_fillable_template_is_an_error(self):
        with self.assertRaises(ValueError):
            plan_rooms(1, {"mirror"}, random.Random(1))


class MatchingTests(unittest.TestCase):
    def test_detections_are_tied_by_the_reviewed_product_not_the_label(self):
        detections = [
            {"id": "a", "type": "bed", "group": "img:1"},
            {"id": "b", "type": "desk", "group": "img:1"},
            {"id": "c", "type": "bed", "group": "img:2"},
            {"id": "d", "type": "bed", "group": None},
        ]
        self.assertEqual([d["id"] for d in detections_of("img:1", detections)], ["a", "b"])
        self.assertEqual(detections_of("img:3", detections), [])

    def test_rank_is_one_based_and_none_when_absent(self):
        results = [result("img:9"), result("img:1")]
        self.assertEqual(rank_of(results, "img:1"), 2)
        self.assertIsNone(rank_of(results, "img:2"))

    def test_article_numbers_compare_by_their_digits(self):
        self.assertEqual(canonical_sku("S994.433.05"), canonical_sku("99443305"))

    def test_another_listing_of_the_answer_counts_as_equivalent(self):
        results = [result("x"), result("y"), result("img:1")]
        self.assertEqual(equivalent_rank(results, "img:1", None), 3)
        same_sku = [{"same_sku": False, "similarity": 0.9}, {"same_sku": True, "similarity": 0.8}]
        self.assertEqual(equivalent_rank(results, "img:1", same_sku), 2)
        same_picture = [{"same_sku": False, "similarity": 0.9995}]
        self.assertEqual(equivalent_rank(results, "img:1", same_picture), 1)

    def test_a_merely_similar_product_is_not_equivalent(self):
        results = [result("x")]
        self.assertIsNone(equivalent_rank(results, "img:1", [{"same_sku": False, "similarity": 0.99}]))

    def test_same_line_ignores_case_and_the_rest_of_the_title(self):
        results = [result("a", title="HEMNES bed"), result("b", title="malm desk")]
        self.assertEqual(same_line_rank(results, "MALM bed frame"), 2)
        self.assertIsNone(same_line_rank(results, "KIVIK sofa"))
        self.assertIsNone(same_line_rank(results, ""))


class OutcomeTests(unittest.TestCase):
    def test_a_product_detected_twice_counts_at_its_best_rank(self):
        detections = [
            {"type": "bed", "verdict": "same", "results": [result("x"), result("x2"), result("img:1")]},
            {"type": "bed", "verdict": "same", "results": [result("img:1")]},
        ]
        summary = summarize(item_outcomes([room(detections=detections)], floor=0.5))
        self.assertEqual(summary["hit1"], 1.0)
        self.assertEqual(summary["mrr"], 1.0)

    def test_an_undetected_product_is_a_miss_but_not_a_search_failure(self):
        items = item_outcomes([room()], floor=0.5)
        self.assertEqual(summarize(items)["hit5"], 0.0)
        self.assertEqual(summarize(items)["not_in_pool"], 1.0)
        self.assertEqual(detected_only(items), [])

    def test_hit_at_5_stops_at_5_and_mrr_uses_the_rank(self):
        results = [result(f"x{i}") for i in range(5)] + [result("img:1")]
        items = item_outcomes([room(detections=[{"type": "bed", "verdict": "same", "results": results}])], floor=0.5)
        summary = summarize(items)
        self.assertEqual(summary["hit5"], 0.0)
        self.assertAlmostEqual(summary["mrr"], 1 / 6)
        self.assertEqual(summary["not_in_pool"], 0.0)

    def test_the_floor_hides_an_answer_below_it(self):
        detections = [{"type": "bed", "verdict": "same", "results": [result("img:1", score=0.4)]}]
        items = item_outcomes([room(detections=detections)], floor=0.5)
        self.assertEqual(summarize(items)["hit1"], 1.0)
        self.assertEqual(summarize(items, rank_key="rank_with_floor")["hit1"], 0.0)

    def test_faithful_only_keeps_detections_reviewed_as_the_same(self):
        detections = [
            {"type": "bed", "verdict": "different", "results": [result("img:1")]},
            {"type": "bed", "verdict": "same", "results": [result("x"), result("img:1")]},
        ]
        items = item_outcomes([room(detections=detections)], floor=0.5)
        self.assertEqual(summarize(items)["hit1"], 1.0)
        self.assertEqual(summarize(faithful_only(items))["hit1"], 0.0)

    def test_an_item_with_no_faithful_detection_drops_out(self):
        detections = [{"type": "bed", "verdict": "different", "results": [result("img:1")]}]
        self.assertEqual(faithful_only(item_outcomes([room(detections=detections)], floor=0.5)), [])

    def test_empty_summary_has_no_rates(self):
        self.assertIsNone(summarize([])["hit1"])

    def test_a_wrong_label_is_kept_and_flagged(self):
        detections = [{"type": "desk", "label": "desk", "verdict": "same", "results": [result("x")]}]
        outcome = item_outcomes([room(detections=detections)], floor=0.5)[0]["detections"][0]
        self.assertEqual((outcome["label"], outcome["label_ok"]), ("desk", False))

    def test_the_right_type_search_is_scored_from_its_own_results(self):
        detections = [{
            "type": "desk", "verdict": "same",
            "results": [result("x")], "filter_level": "type",
            "oracle_results": [result("img:1")], "oracle_filter_level": "subtype",
        }]
        rooms = [room(detections=detections)]
        self.assertEqual(summarize(item_outcomes(rooms, floor=0.5))["hit1"], 0.0)
        oracle = item_outcomes(rooms, floor=0.5, results_key="oracle_results")
        self.assertEqual(summarize(oracle)["hit1"], 1.0)
        self.assertEqual(oracle[0]["detections"][0]["filter_level"], "subtype")

    def test_same_line_is_reported_beside_the_exact_hit(self):
        detections = [{"type": "bed", "verdict": "same", "results": [result("x", title="MALM bed, oak")]}]
        summary = summarize(item_outcomes([room(detections=detections)], floor=0.5))
        self.assertEqual(summary["hit5"], 0.0)
        self.assertEqual(summary["line_hit5"], 1.0)


class BootstrapTests(unittest.TestCase):
    def test_interval_is_flat_when_every_room_agrees(self):
        items = [{"room": f"r{i}", "value": 1.0} for i in range(10)]
        statistic = lambda sample: sum(i["value"] for i in sample) / len(sample)
        self.assertEqual(bootstrap_ci(items, statistic, random.Random(1), draws=50), (1.0, 1.0))

    def test_interval_contains_the_mean_and_has_width_when_rooms_differ(self):
        items = [{"room": f"r{i}", "value": float(i % 2)} for i in range(40)]
        statistic = lambda sample: sum(i["value"] for i in sample) / len(sample)
        low, high = bootstrap_ci(items, statistic, random.Random(1), draws=300)
        self.assertLess(low, 0.5)
        self.assertGreater(high, 0.5)

    def test_no_items_means_no_interval(self):
        self.assertIsNone(bootstrap_ci([], lambda s: 1.0, random.Random(1)))


class ThresholdTests(unittest.TestCase):
    def points(self):
        return [{"room": "r", "type": "bed", "score": s, "correct": c} for s, c in
                [(0.9, True)] * 10 + [(0.7, True)] * 5 + [(0.7, False)] * 5 + [(0.55, False)] * 10]

    def test_calibration_uses_only_reviewed_searches_with_results(self):
        items = item_outcomes(
            [room(detections=[
                {"type": "bed", "verdict": "same", "results": [result("img:1", score=0.9)]},
                {"type": "bed", "verdict": "different", "results": [result("img:1", score=0.8)]},
                {"type": "bed", "verdict": "unsure", "results": [result("img:1", score=0.7)]},
                {"type": "bed", "verdict": "same", "results": []},
            ])],
            floor=0.5,
        )
        self.assertEqual(calibration_points(items), [{"room": "room001", "type": "bed", "score": 0.9, "correct": True}])

    def test_split_is_by_room_without_overlap_and_repeatable(self):
        rooms = [f"r{i}" for i in range(10)]
        fit, check = split_rooms(rooms, random.Random(3))
        self.assertEqual(len(fit), 7)
        self.assertEqual(fit | check, set(rooms))
        self.assertFalse(fit & check)
        self.assertEqual((fit, check), split_rooms(rooms, random.Random(3)))

    def test_grid_covers_both_ends_in_hundredths(self):
        grid = threshold_grid()
        self.assertEqual((grid[0], grid[-1], grid[15]), (0.5, 0.95, 0.65))
        self.assertEqual(len(grid), 46)

    def test_curve_counts_and_precision(self):
        curve = {row["threshold"]: row for row in precision_curve(self.points(), [0.5, 0.6, 0.8, 0.95])}
        self.assertEqual((curve[0.5]["n"], curve[0.5]["precision"]), (30, 0.5))
        self.assertEqual((curve[0.6]["n"], curve[0.6]["precision"]), (20, 0.75))
        self.assertEqual(curve[0.8]["precision"], 1.0)
        self.assertIsNone(curve[0.95]["precision"])

    def test_the_lowest_threshold_reaching_the_target_is_chosen(self):
        curve = precision_curve(self.points(), [0.5, 0.6, 0.8])
        self.assertEqual(choose_threshold(curve, 0.5), 0.5)
        self.assertEqual(choose_threshold(curve, 0.8), 0.8)

    def test_too_few_searches_cant_set_a_threshold(self):
        curve = precision_curve(self.points(), [0.5, 0.8])
        self.assertIsNone(choose_threshold(curve, 0.8, min_n=11))

    def test_labels_partition_the_searches(self):
        table = label_precision(self.points(), high=0.8, medium=0.6)
        self.assertEqual({k: v["n"] for k, v in table.items()}, {"high": 10, "medium": 10, "low": 10})
        self.assertEqual(table["high"]["precision"], 1.0)
        self.assertEqual(table["medium"]["precision"], 0.5)
        self.assertEqual(table["low"]["precision"], 0.0)

    def test_a_missing_threshold_leaves_its_label_empty(self):
        table = label_precision(self.points(), high=None, medium=0.6)
        self.assertEqual(table["high"]["n"], 0)
        self.assertEqual(table["medium"]["n"], 20)


class PromptAndBoardTests(unittest.TestCase):
    def test_prompt_numbers_every_product_in_reference_order(self):
        planned = {
            "kind": "bedroom",
            "style": "Japandi",
            "products": [{"title": "MALM bed frame - IKEA"}, {"title": "KULLEN dresser"}],
        }
        prompt = room_prompt(planned)
        self.assertIn("Reference image 1: MALM bed frame\n", prompt)
        self.assertIn("Reference image 2: KULLEN dresser\n", prompt)
        self.assertIn("exactly these 2 products", prompt)
        self.assertIn("bedroom in a Japandi style", prompt)

    def test_board_fits_every_product_on_one_picture(self):
        images = [Image.new("RGB", (900, 300), "red") for _ in range(3)]
        board = Image.open(BytesIO(reference_board(images, cell=100)))
        self.assertEqual(board.size, (200, 200))


class AppFlowTests(unittest.TestCase):
    products = [{"title": "MALM bed frame - IKEA"}, {"title": "KULLEN dresser"}]

    def test_composer_gets_the_apps_fields_plus_numbered_products_to_choose_from(self):
        text = composer_text("bedroom", "Japandi desc", "oat desc", self.products)
        self.assertTrue(text.startswith("Room type: bedroom\nStyle: Japandi desc\nColor: oat desc\n\n"))
        self.assertIn("1. MALM bed frame\n2. KULLEN dresser\n", text)
        self.assertIn("PRODUCTS: 1, 3", text)
        self.assertTrue(text.endswith("Here is the room photo:"))

    def test_the_composers_picks_are_read_from_its_last_line(self):
        paragraph, picked = parse_composer_choice("A calm room with a bed.\n\nPRODUCTS: 1, 3", 6)
        self.assertEqual((paragraph, picked), ("A calm room with a bed.", [0, 2]))

    def test_picks_in_markdown_or_lower_case_are_read_too(self):
        self.assertEqual(parse_composer_choice("Text.\n**Products:** 2", 3)[1], [1])

    def test_a_doubtful_pick_is_an_error(self):
        for answer in ("No line at all.", "Text.\nPRODUCTS: 7", "Text.\nPRODUCTS: 2, 2", "Text.\nPRODUCTS: 0", "PRODUCTS: 1"):
            with self.assertRaises(ValueError, msg=answer):
                parse_composer_choice(answer, 6)

    def test_products_are_ranked_by_how_alike_their_picture_is_to_the_style(self):
        vectors = {"a": [1.0, 0.0], "b": [0.0, 1.0], "c": [0.7, 0.7]}
        self.assertEqual(rank_by_similarity(vectors, [1.0, 0.1]), ["a", "c", "b"])

    def test_one_product_per_kind_keeps_the_first(self):
        products = [{"type": "chair", "id": 1}, {"type": "desk", "id": 2}, {"type": "chair", "id": 3}]
        self.assertEqual([p["id"] for p in one_per_kind(products)], [1, 2])

    def test_the_critic_loop_stops_at_the_first_pass(self):
        attempts = critic_loop("P", lambda p: (b"img", 0.01), lambda image: (True, "PASS"), 2, lambda v: " retry")
        self.assertEqual(len(attempts), 1)

    def test_a_failing_critic_retries_with_its_reason_and_keeps_the_last(self):
        verdicts = iter([(False, "FAIL: door"), (False, "FAIL: empty"), (False, "FAIL: again")])
        prompts = []

        def generate(prompt):
            prompts.append(prompt)
            return f"img{len(prompts)}".encode(), 0.01

        attempts = critic_loop("P", generate, lambda image: next(verdicts), 2, lambda v: f" [{v}]")
        self.assertEqual(len(attempts), 3)
        self.assertEqual(prompts, ["P", "P [FAIL: door]", "P [FAIL: door] [FAIL: empty]"])
        self.assertEqual(attempts[-1]["image"], b"img3")
        self.assertFalse(attempts[-1]["passed"])

    def test_a_retry_that_passes_ends_the_loop(self):
        verdicts = iter([(False, "FAIL: door"), (True, "PASS")])
        attempts = critic_loop("P", lambda p: (b"x", 0.01), lambda image: next(verdicts), 2, lambda v: "!")
        self.assertEqual([a["passed"] for a in attempts], [False, True])

    def test_products_follow_the_room_photo_in_the_references(self):
        clause = product_clause(self.products)
        self.assertIn("Reference image 1 is the room", clause)
        self.assertIn("Reference image 2: MALM bed frame\nReference image 3: KULLEN dresser\n", clause)

    def test_prompt_is_the_composers_paragraph_then_products_then_the_apps_structure_clause(self):
        import generation

        prompt = app_prompt("A calm bedroom.", "bedroom", "japandi", "warm_oat_cream", self.products)
        self.assertTrue(prompt.startswith("A calm bedroom.\n\nReference image 1"))
        self.assertTrue(prompt.endswith(generation._STRUCTURE_PRESERVATION_CLAUSE))

    def test_without_the_composer_the_apps_static_prompt_is_used(self):
        import generation

        prompt = app_prompt(None, "bedroom", "japandi", "warm_oat_cream", self.products)
        self.assertTrue(prompt.startswith("Redesign this uploaded bedroom"))
        self.assertIn("Reference image 2: MALM bed frame", prompt)
        self.assertTrue(prompt.endswith(generation._STRUCTURE_PRESERVATION_CLAUSE))
        self.assertEqual(prompt.count(generation._STRUCTURE_PRESERVATION_CLAUSE), 1)


class ReferenceKindTests(unittest.TestCase):
    def test_a_product_on_a_light_background_is_a_studio_picture(self):
        image = Image.new("RGB", (200, 200), (245, 245, 245))
        image.paste((120, 60, 20), (40, 40, 160, 160))
        self.assertEqual(reference_kind(image), "studio")

    def test_a_picture_filled_to_its_edges_is_a_scene(self):
        self.assertEqual(reference_kind(Image.new("RGB", (200, 200), (150, 140, 130))), "scene")


class ReviewItemsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_path = Path(directory.name)
        self.manifest = {
            "seed": 1,
            "rooms": [{"id": "room001", "products": [
                {"type": "bed", "ref": "bed.jpg", "group": "img:1"},
                {"type": "lamp", "ref": "lamp.jpg", "group": "img:2"},
            ]}],
        }
        write_json(
            self.run_path / "segments" / "room001.json",
            {
                "detections": [
                    {"id": "room001_00", "type": "bed", "cutout": "a.png", "crop": "a.jpg", "score": 0.9},
                    {"id": "room001_01", "type": "desk", "cutout": None, "crop": "b.jpg"},
                    {"id": "room001_02", "type": "sofa", "cutout": None, "crop": None},
                ]
            },
        )

    def test_every_detection_with_a_picture_is_reviewed_cutout_first(self):
        items = {i["id"]: i for i in review_items(self.run_path, self.manifest)}
        self.assertEqual(set(items), {"room001_00", "room001_01"})
        self.assertEqual(items["room001_00"]["cutout"], "a.png")
        self.assertEqual(items["room001_01"]["cutout"], "b.jpg")

    def test_an_item_shows_every_product_of_its_room_and_nothing_about_the_search(self):
        for item in review_items(self.run_path, self.manifest):
            self.assertEqual(set(item), {"id", "cutout", "products"})
            self.assertEqual({p["group"] for p in item["products"]}, {"img:1", "img:2"})

    def test_an_answer_names_a_product_of_the_room_with_a_verdict_or_none_without(self):
        item = review_items(self.run_path, self.manifest)[0]
        self.assertTrue(valid_review({"group": "img:2", "verdict": "different"}, item))
        self.assertTrue(valid_review({"group": None, "verdict": None}, item))
        self.assertFalse(valid_review({"group": "img:9", "verdict": "same"}, item))
        self.assertFalse(valid_review({"group": "img:1", "verdict": None}, item))
        self.assertFalse(valid_review({"group": None, "verdict": "same"}, item))
        self.assertFalse(valid_review("same", item))


class ReportTests(unittest.TestCase):
    def rooms(self):
        products = [
            {"type": "bed", "group": "img:1", "title": "MALM bed", "ref_kind": "studio"},
            {"type": "desk", "group": "img:2", "title": "LINNMON desk", "ref_kind": "scene"},
        ]
        detections = [
            {"id": "d0", "label": "bed", "type": "bed", "group": "img:1", "verdict": "same",
             "results": [result("img:1", 0.9)], "filter_level": "type",
             "oracle_results": [result("img:1", 0.9)], "oracle_filter_level": "type"},
            {"id": "d1", "label": "console table", "type": "side_table", "group": "img:2", "verdict": "same",
             "results": [result("x", 0.7)], "filter_level": "type",
             "oracle_results": [result("img:2", 0.8)], "oracle_filter_level": "type"},
            {"id": "d2", "label": "vase", "type": None, "group": None, "verdict": None},
        ]
        return [{"id": "room001", "kind": "home office", "products": products, "detections": detections, "segment_seconds": 30.0}]

    def test_the_wrong_label_costs_the_system_but_not_the_search_alone(self):
        metrics = compute_metrics(self.rooms(), {"seed": 1}, {"room001": {"cost": 0.01}}, floor=0.5)
        self.assertEqual(metrics["system"]["hit1"]["value"], 0.5)
        self.assertEqual(metrics["search_only"]["hit1"]["value"], 1.0)
        self.assertEqual(metrics["label_accuracy"], 0.5)
        self.assertEqual(metrics["verdicts"]["none_of_them"], 1)
        self.assertEqual(metrics["label_confusions"], [{"type": "desk", "label": "console table", "count": 1}])
        self.assertEqual(metrics["by_reference"]["scene"]["search_only"]["hit1"], 1.0)
        self.assertIsNone(metrics["thresholds"]["calibrated"])

    def test_an_equivalent_listing_counts_as_right_but_not_strictly(self):
        rooms = self.rooms()
        rooms[0]["detections"][1]["oracle_equivalence"] = [{"same_sku": True, "similarity": 0.95}]
        rooms[0]["detections"][1]["oracle_results"] = [result("img:2-ae", 0.8)]
        metrics = compute_metrics(rooms, {"seed": 1}, {}, floor=0.5)
        self.assertEqual(metrics["search_only"]["hit1"]["value"], 1.0)
        self.assertEqual(metrics["search_only"]["strict_hit1"]["value"], 0.5)

    def test_the_report_is_written(self):
        metrics = compute_metrics(self.rooms(), {"seed": 1}, {}, floor=0.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.md"
            write_report(path, "test", metrics)
            text = path.read_text(encoding="utf-8")
        self.assertIn("การค้นหาอย่างเดียว", text)
        self.assertIn("| desk | console table | 1 |", text)


class JudgeTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run_path = Path(directory.name)
        self.manifest = {"seed": 1, "rooms": [{"id": "room001", "products": []}]}
        write_json(self.run_path / "segments" / "room001.json", {"detections": [
            {"id": "room001_00", "cutout": "a.png", "crop": "a.jpg"},
            {"id": "room001_01", "cutout": None, "crop": "b.jpg"},
            {"id": "room001_02", "cutout": "c.png", "crop": "c.jpg"},
        ]})
        results = [{"group": f"g{i}", "score": 0.9 - i / 100, "title": f"T{i}", "image_url": f"u{i}", "unique_id": str(i)} for i in range(7)]
        write_json(self.run_path / "matches" / "room001.json", {
            "room001_00": {"results": results},
            "room001_01": {"results": results[:2]},
        })

    def test_only_searched_pieces_are_judged_with_the_top_five(self):
        items = {i["id"]: i for i in judge_items(self.run_path, self.manifest)}
        self.assertEqual(set(items), {"room001_00", "room001_01"})
        self.assertEqual({r["group"] for r in items["room001_00"]["results"]}, {f"g{i}" for i in range(5)})
        self.assertEqual(items["room001_01"]["cutout"], "b.jpg")

    def test_a_judged_item_shows_no_score_title_or_rank(self):
        for item in judge_items(self.run_path, self.manifest):
            self.assertEqual(set(item), {"id", "cutout", "results"})
            for result in item["results"]:
                self.assertEqual(set(result), {"group", "image_url"})

    def test_a_judgment_picks_one_result_with_a_grade_or_none(self):
        item = {"results": [{"group": "g0"}, {"group": "g1"}]}
        self.assertTrue(valid_judgment({"best": "g1", "grade": "similar"}, item))
        self.assertTrue(valid_judgment({"best": None, "grade": None}, item))
        self.assertFalse(valid_judgment({"best": "g9", "grade": "exact"}, item))
        self.assertFalse(valid_judgment({"best": "g0", "grade": None}, item))
        self.assertFalse(valid_judgment({"best": None, "grade": "exact"}, item))
        self.assertFalse(valid_judgment({"g0": "exact"}, item))

    def test_human_hits_ranks_and_agreement_with_the_answer(self):
        rooms = [{"detections": [
            {"id": "a", "group": "g0", "results": [result("g0", 0.9), result("g1", 0.8)]},
            {"id": "b", "group": "g9", "results": [result("g2", 0.7), result("g3", 0.6)]},
            {"id": "c", "group": "g5", "results": [result("g4", 0.6)]},
        ]}]
        judgments = {
            "a": {"best": "g0", "grade": "exact"},
            "b": {"best": "g3", "grade": "similar"},
            "c": {"best": None, "grade": None},
        }
        human = human_judgments(rooms, judgments, [0.5, 0.8])
        self.assertEqual(human["n"], 3)
        self.assertAlmostEqual(human["hit_useful"], 2 / 3)
        self.assertAlmostEqual(human["hit_exact"], 1 / 3)
        self.assertAlmostEqual(human["best_is_top1"], 1 / 3)
        self.assertAlmostEqual(human["best_is_answer"], 1 / 3)
        self.assertEqual((human["best_rank"], human["none"]), ({"1": 1, "2": 1}, 1))
        self.assertEqual(human["labels"]["high"], {"n": 1, "useful": 1.0, "exact": 1.0})
        self.assertEqual(human["labels"]["low"]["n"], 1)
        self.assertEqual([row["n"] for row in human["curve"]], [3, 1])

    def test_no_judgments_means_no_section(self):
        self.assertEqual(human_judgments([], {}, [0.5])["n"], 0)


if __name__ == "__main__":
    unittest.main()
