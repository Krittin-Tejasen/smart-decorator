"""Unit tests for furniture_hints.py (generation-time furniture list, closed vocabulary).

Run from backend/:  python -m unittest discover -s tests -v
No network: Gemini is replaced by a fake `post` callable.
"""

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import furniture_hints as fh
from app.services.furniture_types import FURNITURE_TYPES


def fake_response(payload_text):
    return {"candidates": [{"content": {"parts": [{"text": payload_text}]}}]}


class ParseHintsTests(unittest.TestCase):
    def test_counts_are_summed_per_type(self):
        payload = [{"type": "sofa", "count": 1}, {"type": "sofa", "count": 2}]
        self.assertEqual(fh.parse_hints(payload), {"sofa": 3})

    def test_type_is_case_insensitive(self):
        self.assertEqual(fh.parse_hints([{"type": "SOFA", "count": 1}]), {"sofa": 1})

    def test_other_and_unknown_types_are_dropped_not_mapped(self):
        payload = [{"type": "other", "count": 5}, {"type": "a synonym gemini made up", "count": 1}]
        self.assertEqual(fh.parse_hints(payload), {})

    def test_missing_count_defaults_to_one(self):
        self.assertEqual(fh.parse_hints([{"type": "vase"}]), {"vase": 1})

    def test_zero_or_negative_count_is_dropped(self):
        self.assertEqual(fh.parse_hints([{"type": "vase", "count": 0}]), {})

    def test_garbage_payload_shapes_return_empty(self):
        for payload in (None, "not a list", {"type": "sofa"}, [1, 2, 3], []):
            self.assertEqual(fh.parse_hints(payload), {})

    def test_every_furniture_type_is_a_valid_choice(self):
        # Sanity check that the prompt's own closed vocabulary round-trips.
        for t in FURNITURE_TYPES:
            self.assertEqual(fh.parse_hints([{"type": t, "count": 1}]), {t: 1})


class DetectAddedFurnitureTests(unittest.IsolatedAsyncioTestCase):
    async def test_without_an_api_key_nothing_is_sent(self):
        env = {"GEMINI_API_KEY": ""}
        with patch.dict(os.environ, env), patch.object(fh, "_post_to_gemini", new=AsyncMock()) as real:
            self.assertEqual(await fh.detect_added_furniture(b"img", "image/png"), {})
        real.assert_not_called()

    async def test_happy_path_returns_parsed_counts(self):
        post = AsyncMock(return_value=fake_response(
            json.dumps([{"type": "sofa", "count": 1}, {"type": "nightstand", "count": 2}])
        ))
        result = await fh.detect_added_furniture(b"img", "image/png", post=post)
        self.assertEqual(result, {"sofa": 1, "nightstand": 2})
        post.assert_awaited_once()

    async def test_markdown_fenced_json_is_accepted(self):
        post = AsyncMock(return_value=fake_response("```json\n" + json.dumps([{"type": "lamp"}]) + "\n```"))
        self.assertEqual(await fh.detect_added_furniture(b"img", "image/png", post=post), {"lamp": 1})

    async def test_network_failure_returns_empty_instead_of_raising(self):
        post = AsyncMock(side_effect=RuntimeError("quota"))
        self.assertEqual(await fh.detect_added_furniture(b"img", "image/png", post=post), {})

    async def test_malformed_json_returns_empty_instead_of_raising(self):
        post = AsyncMock(return_value=fake_response("not json"))
        self.assertEqual(await fh.detect_added_furniture(b"img", "image/png", post=post), {})

    async def test_request_sends_the_image_and_asks_for_json(self):
        captured = {}

        async def post(body):
            captured["body"] = body
            return fake_response("[]")

        await fh.detect_added_furniture(b"\x89PNG...", "image/png", post=post)
        parts = captured["body"]["contents"][0]["parts"]
        self.assertEqual(len(parts), 2)
        self.assertIn("inline_data", parts[1])
        self.assertEqual(parts[1]["inline_data"]["mime_type"], "image/png")
        self.assertEqual(captured["body"]["generationConfig"]["responseMimeType"], "application/json")


if __name__ == "__main__":
    unittest.main()
