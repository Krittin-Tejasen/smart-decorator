"""Unit tests for feature_extraction.py / feature_vocab.py (colour + material features).

Run from backend/:  python -m unittest discover -s tests -v
No network: Gemini is replaced by a fake `post` callable.
"""

import base64
import io
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image, ImageDraw

import feature_extraction as fx
import feature_vocab as vocab
import segmentation as seg


def solid(size, rgb, alpha=255):
    return Image.new("RGBA", size, (*rgb, alpha))


class VocabTests(unittest.TestCase):
    def test_every_material_maps_to_a_catalog_group(self):
        catalog_groups = {"wood", "fabric", "metal", "glass", "plastic", "other"}
        for material in vocab.MATERIALS:
            self.assertIn(vocab.material_group(material), catalog_groups, material)

    def test_leather_and_rattan_fold_into_the_nearest_catalog_group(self):
        self.assertEqual(vocab.material_group("leather"), "fabric")
        self.assertEqual(vocab.material_group("rattan"), "wood")
        self.assertEqual(vocab.material_group("ceramic"), "other")

    def test_unknown_material_has_no_group(self):
        self.assertIsNone(vocab.material_group("unobtainium"))

    def test_catalog_colours_are_the_twelve_catalog_words(self):
        self.assertEqual(
            set(vocab.CATALOG_COLORS),
            {"white", "black", "gray", "brown", "beige", "green", "blue", "red",
             "yellow", "pink", "orange", "purple"},
        )


class ColourMathTests(unittest.TestCase):
    def test_white_and_black_in_lab(self):
        white, black = fx.srgb_to_lab([[255, 255, 255], [0, 0, 0]])
        self.assertAlmostEqual(white[0], 100, delta=0.1)
        self.assertAlmostEqual(white[1], 0, delta=0.1)
        self.assertAlmostEqual(black[0], 0, delta=0.1)

    def test_pure_red_in_lab_matches_published_value(self):
        L, a, b = fx.srgb_to_lab([[255, 0, 0]])[0]
        self.assertAlmostEqual(L, 53.24, delta=0.1)
        self.assertAlmostEqual(a, 80.09, delta=0.2)
        self.assertAlmostEqual(b, 67.20, delta=0.2)

    def test_lab_hex_round_trip(self):
        for rgb in [(195, 166, 134), (0, 0, 0), (255, 255, 255), (109, 66, 44), (30, 60, 200)]:
            hex_back = fx.lab_to_hex(fx.srgb_to_lab([rgb])[0])
            got = tuple(int(hex_back[i : i + 2], 16) for i in (1, 3, 5))
            for want, have in zip(rgb, got):
                self.assertLessEqual(abs(want - have), 1, (rgb, hex_back))

    def test_ciede2000_reference_pair(self):
        # Pair 1 from Sharma, Wu & Dalal (2005): expected ΔE00 = 2.0425
        d = fx.ciede2000([50.0, 2.6772, -79.7751], [50.0, 0.0, -82.7485])
        self.assertAlmostEqual(float(d), 2.0425, places=3)

    def test_ciede2000_of_identical_colours_is_zero(self):
        self.assertAlmostEqual(float(fx.ciede2000([60, 10, -5], [60, 10, -5])), 0.0, places=6)

    def test_every_shade_of_a_colour_word_names_that_word(self):
        for name, shades in vocab.CATALOG_COLOR_SHADES.items():
            for rgb in shades:
                self.assertEqual(fx.catalog_color_for_lab(fx.srgb_to_lab([rgb])[0]), name, rgb)

    def test_pale_sage_and_dark_green_are_green_not_gray_or_black(self):
        # regression: real catalog photos (sage armchair, deep-green sofa) used
        # to come out "gray" / "black" with a single prototype per word
        for rgb in [(160, 178, 150), (150, 175, 160), (45, 75, 55)]:
            self.assertEqual(fx.catalog_color_for_lab(fx.srgb_to_lab([rgb])[0]), "green", rgb)

    def test_dark_brown_is_not_black_and_anthracite_is_gray(self):
        self.assertEqual(fx.catalog_color_for_lab(fx.srgb_to_lab([(80, 55, 40)])[0]), "brown")
        self.assertEqual(fx.catalog_color_for_lab(fx.srgb_to_lab([(75, 75, 78)])[0]), "gray")

    def test_plausible_everyday_colours_get_the_expected_catalog_word(self):
        cases = {
            (200, 30, 30): "red",
            (30, 60, 200): "blue",
            (40, 140, 60): "green",
            (250, 250, 250): "white",
            (10, 10, 10): "black",
            (110, 70, 45): "brown",
        }
        for rgb, want in cases.items():
            self.assertEqual(fx.catalog_color_for_lab(fx.srgb_to_lab([rgb])[0]), want, rgb)


class SwatchTests(unittest.TestCase):
    def test_flat_item_gives_one_full_share_swatch_with_a_catalog_word(self):
        swatches = fx.extract_swatches(solid((80, 80), (200, 30, 30)))
        self.assertEqual(len(swatches), 1)
        self.assertEqual(swatches[0].catalog_color, "red")
        self.assertEqual(swatches[0].share, 1.0)
        self.assertEqual(len(swatches[0].lab), 3)

    def test_two_coloured_halves_give_two_swatches_by_share(self):
        img = solid((100, 100), (200, 30, 30))
        ImageDraw.Draw(img).rectangle((0, 0, 29, 99), fill=(30, 60, 200, 255))  # 30% blue
        swatches = fx.extract_swatches(img)
        self.assertEqual([s.catalog_color for s in swatches], ["red", "blue"])
        self.assertAlmostEqual(swatches[0].share, 0.7, delta=0.08)
        self.assertAlmostEqual(sum(s.share for s in swatches), 1.0, delta=0.02)

    def test_transparent_pixels_do_not_count(self):
        img = solid((100, 100), (30, 60, 200), alpha=0)               # blue, invisible
        ImageDraw.Draw(img).rectangle((20, 20, 79, 79), fill=(200, 30, 30, 255))
        swatches = fx.extract_swatches(img)
        self.assertEqual([s.catalog_color for s in swatches], ["red"])

    def test_nearly_identical_shades_are_merged_into_one_swatch(self):
        img = solid((100, 100), (195, 166, 134))
        ImageDraw.Draw(img).rectangle((0, 0, 99, 49), fill=(198, 169, 137, 255))
        self.assertEqual(len(fx.extract_swatches(img)), 1)

    def test_a_small_specular_highlight_is_not_reported_as_a_colour(self):
        img = solid((100, 100), (110, 70, 45))
        ImageDraw.Draw(img).rectangle((10, 10, 89, 11), fill=(255, 255, 255, 255))  # ~1.6% glare
        names = {s.catalog_color for s in fx.extract_swatches(img)}
        self.assertNotIn("white", names)

    def test_fully_transparent_image_has_no_swatches(self):
        self.assertEqual(fx.extract_swatches(solid((50, 50), (1, 2, 3), alpha=0)), [])

    def test_result_is_deterministic(self):
        rng = np.random.RandomState(1)
        noise = rng.randint(0, 255, (60, 60, 4)).astype("uint8")
        noise[..., 3] = 255
        img = Image.fromarray(noise, "RGBA")
        self.assertEqual(
            [s.model_dump() for s in fx.extract_swatches(img)],
            [s.model_dump() for s in fx.extract_swatches(img)],
        )


def fake_response(payload_text):
    return {"candidates": [{"content": {"parts": [{"text": payload_text}]}}]}


class ParseEnrichmentTests(unittest.TestCase):
    def test_parses_materials_finish_colour_and_usable(self):
        out = fx.parse_enrichment(
            [{"id": "1", "materials": [{"name": "Wood", "confidence": 0.9}, {"name": "fabric", "confidence": 0.4}],
              "finish": " light  oak, matte ", "color_name": "Terracotta", "usable": True}],
            {"1"},
        )
        e = out["1"]
        self.assertEqual([(m.name, m.confidence) for m in e.materials], [("wood", 0.9), ("fabric", 0.4)])
        self.assertEqual(e.finish, "light oak, matte")
        self.assertEqual(e.color_name, "Terracotta")
        self.assertTrue(e.usable)

    def test_words_outside_the_vocabulary_and_duplicates_are_dropped(self):
        out = fx.parse_enrichment(
            [{"id": "1", "materials": [{"name": "unobtainium", "confidence": 1}, {"name": "wood", "confidence": 0.8},
                                       {"name": "wood", "confidence": 0.7}, {"name": "metal", "confidence": 0.2},
                                       {"name": "glass", "confidence": 0.1}]}],
            {"1"},
        )
        self.assertEqual([m.name for m in out["1"].materials], ["wood", "metal"])   # also capped at 2

    def test_confidence_is_clamped_and_bad_values_get_a_default(self):
        out = fx.parse_enrichment(
            [{"id": "1", "materials": [{"name": "wood", "confidence": 7}, {"name": "metal", "confidence": "high"}]}],
            {"1"},
        )
        self.assertEqual([m.confidence for m in out["1"].materials], [1.0, 0.5])

    def test_unknown_ids_are_ignored(self):
        self.assertEqual(fx.parse_enrichment([{"id": "99", "materials": []}], {"1"}), {})

    def test_usable_false_is_kept_and_missing_means_usable(self):
        out = fx.parse_enrichment([{"id": "1", "usable": False}, {"id": "2"}], {"1", "2"})
        self.assertFalse(out["1"].usable)
        self.assertTrue(out["2"].usable)

    def test_garbage_shapes_do_not_raise(self):
        for junk in (None, 5, "text", {"nothing": 1}, [1, "a", None], [{"id": "1", "materials": "wood"}]):
            fx.parse_enrichment(junk, {"1"})

    def test_object_wrapped_list_is_accepted(self):
        out = fx.parse_enrichment({"items": [{"id": "1", "materials": [{"name": "wood", "confidence": 1}]}]}, {"1"})
        self.assertIn("1", out)


class EnrichFeaturesTests(unittest.IsolatedAsyncioTestCase):
    def inputs(self, n):
        return [fx.EnrichInput(id=str(i), label="sofa", image=solid((40, 40), (120, 90, 60))) for i in range(n)]

    async def test_provider_none_never_calls_gemini(self):
        post = AsyncMock()
        with patch.dict(os.environ, {"FEATURE_PROVIDER": "none"}):
            self.assertEqual(await fx.enrich_features(self.inputs(3), post=post), {})
        post.assert_not_called()

    async def test_one_request_for_a_room_and_results_are_keyed_by_id(self):
        async def post(body):
            ids = [p["text"].split()[0] for p in body["contents"][0]["parts"] if p.get("text", "").startswith("id=")]
            import json
            return fake_response("```json\n" + json.dumps(
                [{"id": i.removeprefix("id="), "materials": [{"name": "fabric", "confidence": 0.9}]} for i in ids]
            ) + "\n```")

        spy = AsyncMock(side_effect=post)
        with patch.dict(os.environ, {"FEATURE_PROVIDER": "gemini"}):
            out = await fx.enrich_features(self.inputs(5), post=spy)
        self.assertEqual(spy.await_count, 1)
        self.assertEqual(sorted(out), ["0", "1", "2", "3", "4"])
        self.assertEqual(out["3"].materials[0].name, "fabric")

    async def test_large_rooms_are_split_into_batches(self):
        spy = AsyncMock(return_value=fake_response("[]"))
        with patch.dict(os.environ, {"FEATURE_PROVIDER": "gemini"}):
            await fx.enrich_features(self.inputs(25), post=spy)
        self.assertEqual(spy.await_count, 3)     # 12 + 12 + 1

    async def test_failure_returns_empty_instead_of_raising(self):
        for post in (AsyncMock(side_effect=RuntimeError("quota")), AsyncMock(return_value=fake_response("not json"))):
            with patch.dict(os.environ, {"FEATURE_PROVIDER": "gemini"}):
                self.assertEqual(await fx.enrich_features(self.inputs(2), post=post), {})

    async def test_without_an_api_key_nothing_is_sent(self):
        env = {"FEATURE_PROVIDER": "gemini", "GEMINI_API_KEY": ""}
        with patch.dict(os.environ, env), patch.object(fx, "_post_to_gemini", new=AsyncMock()) as real:
            self.assertEqual(await fx.enrich_features(self.inputs(2)), {})
        real.assert_not_called()

    def test_request_has_one_labelled_jpeg_per_item(self):
        body = fx.build_request(self.inputs(3))
        parts = body["contents"][0]["parts"]
        self.assertEqual(sum("inline_data" in p for p in parts), 3)
        self.assertIn('id=2 type="sofa"', [p.get("text") for p in parts])
        jpeg = base64.b64decode(parts[2]["inline_data"]["data"])
        self.assertEqual(Image.open(io.BytesIO(jpeg)).format, "JPEG")
        self.assertEqual(body["generationConfig"]["temperature"], 0)

    def test_transparent_background_is_flattened_to_white_for_the_model(self):
        img = solid((20, 20), (255, 0, 0), alpha=0)
        flat = Image.open(io.BytesIO(fx.prepare_image(img))).convert("RGB")
        self.assertGreater(min(flat.getpixel((10, 10))), 240)


class SegmentationIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def build(self):
        room = Image.new("RGB", (400, 400), (30, 60, 200))
        ImageDraw.Draw(room).rectangle((100, 200, 300, 300), fill=(200, 30, 30))
        mask = Image.new("L", (400, 400), 0)
        ImageDraw.Draw(mask).rectangle((100, 200, 300, 300), fill=255)
        return seg._build_item(room, "sofa", 0.9, 0, 0.2, 0.4, 0.8, 0.85, mask)

    def test_build_item_measures_catalog_colours_from_the_masked_pixels(self):
        item = self.build()
        self.assertEqual([c.catalog_color for c in item.features.colors], ["red"])   # not the blue wall
        self.assertEqual(item.features.dominant_colors, [c.hex for c in item.features.colors])
        self.assertEqual(item.features.meta.source, "pixels_only")
        self.assertEqual(item.features.materials, [])

    def test_features_saved_before_this_change_still_parse(self):
        old = seg.FurnitureFeatures(**{"dominant_colors": ["#112233"], "area_pct": 1.0, "aspect_ratio": 1.5})
        self.assertEqual(old.colors, [])
        self.assertTrue(old.meta.usable)

    def test_default_lists_are_not_shared_between_items(self):
        a = seg.FurnitureFeatures(dominant_colors=[], area_pct=1, aspect_ratio=1)
        b = seg.FurnitureFeatures(dominant_colors=[], area_pct=1, aspect_ratio=1)
        a.materials.append(fx.MaterialGuess(name="wood", confidence=1))
        self.assertEqual(b.materials, [])

    async def test_enrichment_is_applied(self):
        item = self.build()
        enrichment = fx.Enrichment(
            materials=[fx.MaterialGuess(name="fabric", confidence=0.9)], finish="woven, matte",
            color_name="brick red", usable=False,
        )
        with patch.object(seg, "enrich_features", new=AsyncMock(return_value={item.id: enrichment})):
            await seg._enrich_items([item])
        f = item.features
        self.assertEqual([m.name for m in f.materials], ["fabric"])
        self.assertEqual(f.finish, "woven, matte")
        self.assertEqual(f.color_name, "brick red")                # display only…
        self.assertEqual(f.colors[0].catalog_color, "red")         # …the matching key stays measured
        self.assertEqual((f.meta.source, f.meta.usable), ("gemini", False))

    async def test_items_missing_from_the_answer_stay_pixels_only(self):
        item = self.build()
        with patch.object(seg, "enrich_features", new=AsyncMock(return_value={})):
            await seg._enrich_items([item])
        self.assertEqual(item.features.meta.source, "pixels_only")

    async def test_enrichment_crash_never_breaks_segmentation(self):
        item = self.build()
        with patch.object(seg, "enrich_features", new=AsyncMock(side_effect=RuntimeError("boom"))):
            await seg._enrich_items([item])
        self.assertEqual(item.features.meta.source, "pixels_only")

    async def test_no_items_means_no_call(self):
        spy = AsyncMock()
        with patch.object(seg, "enrich_features", new=spy):
            await seg._enrich_items([])
        spy.assert_not_called()


class ErrorLoggingTests(unittest.IsolatedAsyncioTestCase):
    """The Gemini API key is a ?key= query parameter and httpx quotes the whole
    URL in its errors: it must never reach the logs."""

    SECRET = "AQ.SECRET-KEY-123"

    def http_error(self):
        import httpx
        url = f"https://generativelanguage.googleapis.com/v1beta/models/m:generateContent?key={self.SECRET}"
        request = httpx.Request("POST", url)
        return httpx.HTTPStatusError(
            f"Client error '402 Payment Required' for url '{url}'",
            request=request, response=httpx.Response(402, request=request),
        )

    def test_safe_error_blanks_the_key_but_keeps_the_reason(self):
        text = fx.safe_error(self.http_error())
        self.assertNotIn(self.SECRET, text)
        self.assertIn("key=REDACTED", text)
        self.assertIn("402 Payment Required", text)
        self.assertTrue(text.startswith("HTTPStatusError:"))

    def test_safe_error_leaves_ordinary_errors_alone(self):
        self.assertEqual(fx.safe_error(RuntimeError("quota")), "RuntimeError: quota")

    async def test_a_failed_gemini_call_logs_without_the_key(self):
        import contextlib
        out = io.StringIO()
        inputs = [fx.EnrichInput(id="0", label="sofa", image=solid((20, 20), (1, 2, 3)))]
        with patch.dict(os.environ, {"FEATURE_PROVIDER": "gemini"}), contextlib.redirect_stdout(out):
            result = await fx.enrich_features(inputs, post=AsyncMock(side_effect=self.http_error()))
        self.assertEqual(result, {})
        self.assertNotIn(self.SECRET, out.getvalue())
        self.assertIn("402", out.getvalue())

    async def test_segmentation_logs_without_the_key_too(self):
        import contextlib
        item = SegmentationIntegrationTests().build()
        out = io.StringIO()
        with patch.object(seg, "enrich_features", new=AsyncMock(side_effect=self.http_error())), contextlib.redirect_stdout(out):
            await seg._enrich_items([item])
        self.assertNotIn(self.SECRET, out.getvalue())


if __name__ == "__main__":
    unittest.main()
