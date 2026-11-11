"""Unit tests for label_verification.py (post-segmentation label double-check).

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

from PIL import Image

import label_verification as lv
from app.services.furniture_types import FURNITURE_TYPES


def fake_response(payload_text):
    return {"candidates": [{"content": {"parts": [{"text": payload_text}]}}]}


def solid(size=(20, 20), rgb=(120, 90, 60)):
    return Image.new("RGB", size, rgb)


class ParseVerificationsTests(unittest.TestCase):
    def test_a_recognised_id_and_type_is_kept(self):
        payload = [{"id": "0", "type": "coffee_table"}]
        self.assertEqual(lv.parse_verifications(payload, {"0"}), {"0": "coffee_table"})

    def test_an_id_outside_wanted_ids_is_dropped(self):
        payload = [{"id": "99", "type": "sofa"}]
        self.assertEqual(lv.parse_verifications(payload, {"0"}), {})

    def test_other_and_unknown_types_are_dropped(self):
        payload = [{"id": "0", "type": "other"}, {"id": "1", "type": "a made-up word"}]
        self.assertEqual(lv.parse_verifications(payload, {"0", "1"}), {})

    def test_type_is_case_insensitive(self):
        self.assertEqual(lv.parse_verifications([{"id": "0", "type": "SOFA"}], {"0"}), {"0": "sofa"})

    def test_garbage_payload_shapes_return_empty(self):
        for payload in (None, "not a list", {"id": "0"}, [1, 2, 3], []):
            self.assertEqual(lv.parse_verifications(payload, {"0"}), {})

    def test_every_furniture_type_is_a_valid_choice(self):
        for t in FURNITURE_TYPES:
            self.assertEqual(lv.parse_verifications([{"id": "0", "type": t}], {"0"}), {"0": t})


class VerifyLabelsTests(unittest.IsolatedAsyncioTestCase):
    def inputs(self, n):
        return [
            lv.VerifyInput(id=str(i), label="desk", cutout=solid(), context=solid((40, 40)))
            for i in range(n)
        ]

    async def test_without_an_api_key_nothing_is_sent(self):
        env = {"GEMINI_API_KEY": ""}
        with patch.dict(os.environ, env), patch.object(lv, "_post_to_gemini", new=AsyncMock()) as real:
            self.assertEqual(await lv.verify_labels(self.inputs(2)), {})
        real.assert_not_called()

    async def test_no_inputs_means_no_call(self):
        post = AsyncMock()
        self.assertEqual(await lv.verify_labels([], post=post), {})
        post.assert_not_called()

    async def test_happy_path_returns_parsed_types(self):
        post = AsyncMock(return_value=fake_response(
            json.dumps([{"id": "0", "type": "coffee_table"}, {"id": "1", "type": "desk"}])
        ))
        result = await lv.verify_labels(self.inputs(2), post=post)
        self.assertEqual(result, {"0": "coffee_table", "1": "desk"})
        post.assert_awaited_once()

    async def test_large_batches_are_split(self):
        post = AsyncMock(return_value=fake_response("[]"))
        await lv.verify_labels(self.inputs(25), post=post)
        self.assertEqual(post.await_count, 5)  # 6+6+6+6+1, batch size 6 (two images per item)

    async def test_network_failure_returns_empty_instead_of_raising(self):
        post = AsyncMock(side_effect=RuntimeError("quota"))
        self.assertEqual(await lv.verify_labels(self.inputs(2), post=post), {})

    async def test_malformed_json_returns_empty_instead_of_raising(self):
        post = AsyncMock(return_value=fake_response("not json"))
        self.assertEqual(await lv.verify_labels(self.inputs(2), post=post), {})

    def test_request_sends_the_current_guess_and_both_images_per_item(self):
        body = lv.build_request(self.inputs(2))
        parts = body["contents"][0]["parts"]
        # 2 items x (1 cutout + 1 context) = 4 images total.
        self.assertEqual(sum("inline_data" in p for p in parts), 4)
        texts = [p.get("text") for p in parts if "text" in p]
        self.assertIn('id=1 current_guess="desk" - ITEM:', texts)
        self.assertIn('id=1 - CONTEXT:', texts)
        self.assertEqual(body["generationConfig"]["responseMimeType"], "application/json")

    def test_item_image_comes_before_context_image_for_each_id(self):
        body = lv.build_request(self.inputs(1))
        parts = body["contents"][0]["parts"]
        item_text_idx = next(i for i, p in enumerate(parts) if p.get("text") == 'id=0 current_guess="desk" - ITEM:')
        context_text_idx = next(i for i, p in enumerate(parts) if p.get("text") == "id=0 - CONTEXT:")
        self.assertLess(item_text_idx, context_text_idx)
        self.assertIn("inline_data", parts[item_text_idx + 1])
        self.assertIn("inline_data", parts[context_text_idx + 1])


if __name__ == "__main__":
    unittest.main()
