"""
label_verification.py
─────────────────────
Second opinion on each piece's furniture type, from Gemini, after Grounding
DINO has found it and SAM 2 has cut it out. Replaces the CLIP relabel as the
main naming step; CLIP stays only as the offline fallback (no key, call failed).

Why: on 5 rooms generated from 30 IKEA products (2026-11-15), DINO + SAM 2 +
CLIP found every piece but named only 15/30 right. It mixed up look-alike types
(dining chairs and bar stools as "armchair", desks and dining tables as
"console table"). Gemini, shown the same cut-outs, named 30/30 right and turned
no correct name into a wrong one. The design (two images per piece, closed
vocabulary, one batched call per room) is the teammate's
feature/dynamic-furniture-labels `label_verification.py`, ported here without
its feature_extraction dependency.

Two images per piece:
  1. ITEM: the exact SAM 2 cut-out on white, so the model knows which pixels
     are in question.
  2. CONTEXT: the piece's box plus a margin of the room, to tell a coffee table
     from a bench. Context alone mis-named a chair as the table beside it.

Best-effort: {} on any failure (no key, quota, bad JSON, network). A missing id
means "not verified", never "confirmed".
"""

import asyncio
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Awaitable, Callable, NamedTuple

import httpx
from PIL import Image

from app.services.furniture_types import FURNITURE_TYPES

PostFn = Callable[[dict], Awaitable[dict]]

_BATCH_SIZE = 6       # two images per piece: keeps each request a sane size
_IMAGE_PX = 256       # longest side sent to Gemini
_ALLOWED = tuple(FURNITURE_TYPES) + ("other",)
_SECRET_QUERY = re.compile(r"((?:key|api_key)=)[^&\s]+")

# Type -> a segmentation label that maps back to it (furniture_types._LABEL_TO_TYPE),
# for when Gemini moves a piece to another type.
_TYPE_LABEL = {"tv_bench": "tv stand"}


class VerifyInput(NamedTuple):
    id: str
    label: str
    cutout: Image.Image   # the exact piece (RGBA cut-out or crop)
    context: Image.Image  # its box plus a margin of the room


_INSTRUCTIONS = (
    "For each id below you are shown TWO images, in this order:\n"
    "1. ITEM - the exact piece of furniture/decor in question, cut tight "
    "around it (on a plain white background).\n"
    "2. CONTEXT - a wider crop of the same interior photo around that same "
    "item, showing some of its surroundings (what it's next to, in front of, "
    "resting on).\n"
    "Identify ONLY the item shown in image 1. Use image 2 purely as supporting "
    "context to help tell it apart from a similar-looking type (e.g. a coffee "
    "table vs a bench, a dining chair vs an armchair vs a bar stool) - never "
    "answer with something that is only visible in image 2 and not image 1.\n"
    "Each id is preceded by what a detector currently guesses the item is - "
    "that guess can be wrong, especially for an unusual shape.\n"
    "For EVERY id, answer with the single best match for what image 1 really "
    "shows, choosing EXACTLY one word from this list (pick the closest even if "
    'the current guess looks close; use "other" only if truly none fit): '
    + ", ".join(_ALLOWED) + "\n"
    "Return ONLY a JSON array, no markdown, no extra text. Each element:\n"
    '  {"id": <the same id given with that item>, "type": <your answer>}'
)


def safe_error(exc: BaseException) -> str:
    """`ExcType: message` for logs, with any API key in a URL blanked out."""
    text = _SECRET_QUERY.sub(r"\1REDACTED", str(exc))
    return f"{type(exc).__name__}: {text}"


def prepare_image(image: Image.Image) -> bytes:
    """Flattened onto white, longest side _IMAGE_PX, JPEG."""
    rgba = image.convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.split()[-1])
    flat.thumbnail((_IMAGE_PX, _IMAGE_PX), Image.LANCZOS)
    buf = BytesIO()
    flat.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def build_request(inputs: list[VerifyInput]) -> dict:
    parts: list[dict] = [{"text": _INSTRUCTIONS}]
    for item in inputs:
        parts.append({"text": f'id={item.id} current_guess="{item.label}" - ITEM:'})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(prepare_image(item.cutout))}})
        parts.append({"text": f"id={item.id} - CONTEXT:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": _b64(prepare_image(item.context))}})
    return {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }


def _response_json(response: dict) -> Any:
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


def parse_verifications(payload: Any, wanted_ids: set[str]) -> dict[str, str]:
    """{item_id: type}. "other", unknown types and unknown ids are dropped."""
    if not isinstance(payload, list):
        return {}
    out: dict[str, str] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("id", ""))
        answer = str(entry.get("type", "")).strip().lower()
        if item_id in wanted_ids and answer in FURNITURE_TYPES:
            out[item_id] = answer
    return out


def label_for_type(current_label: str, furniture_type: str) -> str:
    """The label to show for a piece Gemini typed as `furniture_type`. Keeps
    DINO's own word when it already means that type, so "table lamp" stays a
    table lamp (product matching uses the lamp subtype) instead of "lamp"."""
    from app.services.furniture_types import label_to_type

    if label_to_type(current_label) == furniture_type:
        return current_label
    return _TYPE_LABEL.get(furniture_type, furniture_type.replace("_", " "))


async def _post_to_gemini(body: dict) -> dict:
    model = os.getenv("GEMINI_LABEL_MODEL") or os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            headers={"x-goog-api-key": os.getenv("GEMINI_API_KEY", "").strip()},
            json=body,
        )
        response.raise_for_status()
        return response.json()


async def verify_labels(inputs: list[VerifyInput], post: PostFn | None = None) -> dict[str, str]:
    """{item_id: type} from Gemini, closed vocabulary. Never raises: {} when
    there is nothing to check, no API key, or every call failed."""
    if not inputs:
        return {}
    if post is None:
        if not os.getenv("GEMINI_API_KEY", "").strip():
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
