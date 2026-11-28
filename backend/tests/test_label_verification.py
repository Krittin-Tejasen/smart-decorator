"""Tests for the Gemini naming step (label_verification.py and
segmentation.name_pieces). No network: Gemini is a fake `post` or patched."""

import asyncio
import json
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image

import label_verification as lv
import segmentation as seg


def _answer(entries) -> dict:
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(entries)}]}}]}


def _inputs(n: int) -> list[lv.VerifyInput]:
    img = Image.new("RGB", (40, 40), "white")
    return [lv.VerifyInput(str(i), "armchair", img, img) for i in range(n)]


class ParseTests(unittest.TestCase):
    def test_keeps_known_types_for_asked_ids_only(self):
        got = lv.parse_verifications(
            [{"id": "0", "type": "Stool"}, {"id": "1", "type": "other"},
             {"id": "2", "type": "spaceship"}, {"id": "9", "type": "sofa"}],
            {"0", "1", "2"},
        )
        self.assertEqual(got, {"0": "stool"})

    def test_not_a_list_gives_nothing(self):
        self.assertEqual(lv.parse_verifications({"id": "0"}, {"0"}), {})

    def test_strips_markdown_fences(self):
        response = {"candidates": [{"content": {"parts": [{"text": '```json\n[{"id": "0"}]\n```'}]}}]}
        self.assertEqual(lv._response_json(response), [{"id": "0"}])


class LabelForTypeTests(unittest.TestCase):
    def test_keeps_dinos_word_when_it_already_means_the_type(self):
        # product matching uses the lamp subtype, so "table lamp" must survive
        self.assertEqual(lv.label_for_type("table lamp", "lamp"), "table lamp")

    def test_moves_to_a_label_that_maps_back_to_the_type(self):
        from app.services.furniture_types import FURNITURE_TYPES, label_to_type

        for furniture_type in FURNITURE_TYPES:
            label = lv.label_for_type("decoration", furniture_type)
            self.assertEqual(label_to_type(label), furniture_type, label)


class VerifyLabelsTests(unittest.TestCase):
    def test_batches_of_six_and_merges(self):
        calls = []

        async def post(body):
            ids = [p["text"].split()[0][3:] for p in body["contents"][0]["parts"][1:] if "ITEM" in p.get("text", "")]
            calls.append(ids)
            return _answer([{"id": i, "type": "chair"} for i in ids])

        got = asyncio.run(lv.verify_labels(_inputs(8), post=post))
        self.assertEqual([len(c) for c in calls], [6, 2])
        self.assertEqual(got, {str(i): "chair" for i in range(8)})

    def test_a_failed_call_gives_nothing_not_an_error(self):
        async def post(body):
            raise RuntimeError("402 Payment Required ?key=SECRET")

        self.assertEqual(asyncio.run(lv.verify_labels(_inputs(2), post=post)), {})

    def test_no_key_means_no_call(self):
        with mock.patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
            self.assertEqual(asyncio.run(lv.verify_labels(_inputs(1))), {})

    def test_safe_error_hides_the_key(self):
        self.assertNotIn("SECRET", lv.safe_error(RuntimeError("https://x/?key=SECRET&a=1")))

    def test_request_sends_two_images_per_piece(self):
        body = lv.build_request(_inputs(3))
        images = [p for p in body["contents"][0]["parts"] if "inline_data" in p]
        self.assertEqual(len(images), 6)


def _detections():
    img = Image.new("RGB", (100, 100), "white")
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (10, 10, 50, 50))
    return img, [
        {"label": "armchair", "box": (0.1, 0.1, 0.5, 0.5), "mask": mask, "confidence": 0.5},
        {"label": "console table", "box": (0.5, 0.5, 0.9, 0.9), "mask": None, "confidence": 0.5},
    ]


class NamePiecesTests(unittest.TestCase):
    def test_gemini_renames_and_clip_only_gets_the_rest(self):
        img, dets = _detections()
        seen = {}

        async def fake_verify(inputs):
            seen["n"] = len(inputs)
            return {"0": "stool"}

        clip_got = []
        with mock.patch.dict(os.environ, {"SEGMENTATION_LABELER": "gemini"}), \
                mock.patch.object(lv, "verify_labels", fake_verify), \
                mock.patch.object(seg, "relabel_with_clip", lambda _img, rest: clip_got.extend(rest)):
            asyncio.run(seg.name_pieces(img, dets))
        self.assertEqual(seen["n"], 2)
        self.assertEqual(dets[0]["label"], "stool")
        self.assertEqual(dets[0]["dino_label"], "armchair")
        self.assertEqual(dets[0]["label_source"], "gemini")
        self.assertEqual(clip_got, [dets[1]])

    def test_confirmed_label_is_kept_but_marked(self):
        img, dets = _detections()

        async def fake_verify(inputs):
            return {"0": "armchair", "1": "desk"}

        with mock.patch.dict(os.environ, {"SEGMENTATION_LABELER": "gemini"}), \
                mock.patch.object(lv, "verify_labels", fake_verify), \
                mock.patch.object(seg, "relabel_with_clip", lambda *_: self.fail("CLIP not needed")):
            asyncio.run(seg.name_pieces(img, dets))
        self.assertEqual(dets[0]["label"], "armchair")
        self.assertNotIn("dino_label", dets[0])
        self.assertEqual(dets[1]["label"], "desk")

    def test_clip_labeler_never_calls_gemini(self):
        img, dets = _detections()

        async def fake_verify(inputs):
            raise AssertionError("Gemini called")

        clip_got = []
        with mock.patch.dict(os.environ, {"SEGMENTATION_LABELER": "clip"}), \
                mock.patch.object(lv, "verify_labels", fake_verify), \
                mock.patch.object(seg, "relabel_with_clip", lambda _img, rest: clip_got.extend(rest)):
            asyncio.run(seg.name_pieces(img, dets))
        self.assertEqual(len(clip_got), 2)

    def test_none_labeler_changes_nothing(self):
        img, dets = _detections()
        with mock.patch.dict(os.environ, {"SEGMENTATION_LABELER": "none"}), \
                mock.patch.object(seg, "relabel_with_clip", lambda *_: self.fail("CLIP called")):
            asyncio.run(seg.name_pieces(img, dets))
        self.assertEqual([d["label"] for d in dets], ["armchair", "console table"])


if __name__ == "__main__":
    unittest.main()
