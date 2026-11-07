"""
label_verification.py
───────────────────────
After segmentation produces its items, double-check each detected label
against its actual crop with a Gemini call, answering from the same closed
vocabulary used everywhere else (furniture_types.FURNITURE_TYPES).

Why this is needed even with a correct hint list: Grounding DINO's
vocabulary only has one phrase per type, and for an ambiguous object its
only real visual affinity can be a *different* word entirely - confirmed on
a real photo where a round coffee table was never once recognised as
"coffee table" under any vocabulary, but WAS found (mislabelled "desk")
whenever "desk" happened to be available (see
segmentation._fill_missing_from_static_vocab). That gap-fill recovers the
box; this corrects the label afterwards, from the crop alone.

One batched call per room (same pattern as feature_extraction.enrich_
features), not one per item. Best-effort: {} on any failure (no key, quota,
bad JSON, network) - callers must treat a missing id as "not verified",
never as "confirmed wrong."
"""

import asyncio
import base64
import json
import os
import re
from typing import Any, Awaitable, Callable, NamedTuple

import httpx
from PIL import Image

from app.services.furniture_types import FURNITURE_TYPES
from feature_extraction import prepare_image, safe_error

PostFn = Callable[[dict], Awaitable[dict]]

_BATCH_SIZE = 12
_ALLOWED_TYPES = tuple(FURNITURE_TYPES) + ("other",)


class VerifyInput(NamedTuple):
    id: str
    label: str
    image: Image.Image  # RGBA cut-out, or an RGB crop when there is no mask


_INSTRUCTIONS = (
    "Each image below is one piece of furniture/decor cut out of the SAME "
    "AI-generated interior photo (background removed, shown on white). Each "
    "is preceded by its id and what a detector currently guesses it is - "
    "that guess can be wrong, especially for an unusual shape.\n"
    "For EVERY id, look at the actual picture and answer with the single "
    "best match for what it really shows, choosing EXACTLY one word from "
    'this list (pick the closest even if the current guess looks close; '
    'use "other" only if truly none fit): ' + ", ".join(_ALLOWED_TYPES) + "\n"
    "Return ONLY a JSON array, no markdown, no extra text. Each element:\n"
    '  {"id": <the same id given with that image>, "type": <your answer>}'
)


def build_request(inputs: list[VerifyInput]) -> dict:
    parts: list[dict] = [{"text": _INSTRUCTIONS}]
    for item in inputs:
        parts.append({"text": f'id={item.id} current_guess="{item.label}"'})
        parts.append(
            {"inline_data": {"mime_type": "image/jpeg", "data": _b64(prepare_image(item.image))}}
        )
    return {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _response_json(response: dict) -> Any:
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def parse_verifications(payload: Any, wanted_ids: set[str]) -> dict[str, str]:
    """{item_id: verified_type} - "other" and anything outside FURNITURE_TYPES
    or outside `wanted_ids` is dropped, not guessed at."""
    if not isinstance(payload, list):
        return {}
    out: dict[str, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("id", ""))
        if item_id not in wanted_ids:
            continue
        verified_type = str(entry.get("type", "")).strip().lower()
        if verified_type in FURNITURE_TYPES:
            out[item_id] = verified_type
    return out


async def _post_to_gemini(body: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_FEATURE_MODEL") or os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json=body,
        )
        response.raise_for_status()
        return response.json()


async def verify_labels(inputs: list[VerifyInput], post: PostFn | None = None) -> dict[str, str]:
    """{item_id: verified_type}, closed vocabulary. {} - never raises - when
    there's nothing to check, no API key, or the call/parse fails; callers
    treat a missing id as "not verified", not as "confirmed correct"."""
    if not inputs:
        return {}
    if post is None:
        if not os.getenv("GEMINI_API_KEY"):
            return {}
        post = _post_to_gemini

    async def one_batch(batch: list[VerifyInput]) -> dict[str, str]:
        try:
            response = await post(build_request(batch))
            return parse_verifications(_response_json(response), {b.id for b in batch})
        except Exception as exc:
            print(f"[label_verification] Gemini call failed for {len(batch)} item(s): {safe_error(exc)}")
            return {}

    batches = [inputs[i : i + _BATCH_SIZE] for i in range(0, len(inputs), _BATCH_SIZE)]
    merged: dict[str, str] = {}
    for result in await asyncio.gather(*(one_batch(b) for b in batches)):
        merged.update(result)
    return merged
