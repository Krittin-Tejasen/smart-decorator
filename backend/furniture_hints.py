"""
furniture_hints.py
───────────────────
Right after a room image is generated, ask Gemini Vision what furniture is
now visible in it, constrained to the same closed vocabulary segmentation
results are matched against (`furniture_types.FURNITURE_TYPES`) instead of
free text. `run_segmentation` (segmentation.py) uses the result to build a
per-image Grounding DINO vocabulary instead of always scanning for every
known type, and callers learn up front what to expect and how many.

Why a closed vocabulary instead of free text + a synonym map afterwards:
letting Gemini answer in its own words and mapping with
`furniture_types.label_to_type` still drops anything that map doesn't
recognise (a real synonym mismatch problem); asking it to choose from the
same list used everywhere else avoids that at the source. "other" is kept as
an explicit escape hatch for a real item with no matching type - it is
reported for visibility but never used to build the detector vocabulary.
"""

import base64
import json
import os
import re
from typing import Any, Awaitable, Callable

import httpx

from app.services.furniture_types import FURNITURE_TYPES
from feature_extraction import safe_error

PostFn = Callable[[dict], Awaitable[dict]]

_ALLOWED_TYPES = tuple(FURNITURE_TYPES) + ("other",)

_PROMPT = (
    "Look at this interior design photo and list every distinct piece of "
    "furniture or decor visible in it.\n"
    "Return ONLY a JSON array, no markdown, no extra text. Each element:\n"
    '  "type": string - must be EXACTLY one of this list (choose the '
    'closest match, or "other" if truly none fit): '
    + ", ".join(_ALLOWED_TYPES) + "\n"
    '  "count": integer - how many separate instances of that type you see\n'
    "Merge separate instances of the same type into one element with the "
    "right count; never list the same type twice."
)


def _response_json(response: dict) -> Any:
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


async def _post_to_gemini(body: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": api_key},
            json=body,
        )
        response.raise_for_status()
        return response.json()


def parse_hints(payload: Any) -> dict[str, int]:
    """{"sofa": 1, "coffee_table": 1, ...} - "other" and anything outside
    FURNITURE_TYPES is dropped, not forced into the nearest word."""
    if not isinstance(payload, list):
        return {}
    counts: dict[str, int] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        furniture_type = str(entry.get("type", "")).strip().lower()
        if furniture_type not in FURNITURE_TYPES:
            continue
        try:
            n = int(entry.get("count", 1))
        except (TypeError, ValueError):
            n = 1
        if n > 0:
            counts[furniture_type] = counts.get(furniture_type, 0) + n
    return counts


async def detect_added_furniture(
    image_bytes: bytes, mime_type: str, post: PostFn | None = None
) -> dict[str, int]:
    """Best-effort: {} on any failure (no API key, quota, bad JSON, network).

    Callers must treat an empty result as "no hint available" and fall back
    to their default behaviour, not as "the room has no furniture"."""
    if post is None:
        if not os.getenv("GEMINI_API_KEY"):
            return {}
        post = _post_to_gemini

    body = {
        "contents": [
            {
                "parts": [
                    {"text": _PROMPT},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64.b64encode(image_bytes).decode(),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json"},
    }
    try:
        response = await post(body)
        return parse_hints(_response_json(response))
    except Exception as exc:
        print(f"[furniture_hints] Gemini call failed: {safe_error(exc)}")
        return {}
