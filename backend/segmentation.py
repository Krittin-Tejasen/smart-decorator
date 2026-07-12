"""
furniture_segmentation.py
─────────────────────────
POST /segment-furniture
  Accepts a room image (multipart), returns each detected furniture item with
  a labelled bounding box, a cropped image, dominant colours, and shape
  features — ready for downstream product matching.

Detection back-ends (chosen automatically via env vars):
  • Grounded SAM 2 on Replicate  — if REPLICATE_API_TOKEN + SAM_REPLICATE_MODEL
  • Gemini Vision fallback        — always available when GEMINI_API_KEY is set

Env vars
────────
  SAM_REPLICATE_MODEL   Replicate model slug  (default: adirik/grounded-sam-2)
  SAM_BOX_THRESHOLD     Confidence threshold for detection boxes (default: 0.30)
  SAM_TEXT_THRESHOLD    Text-matching threshold for Grounded SAM (default: 0.25)
  REPLICATE_API_TOKEN   Already used by /generate-room
  GEMINI_API_KEY        Already used by /generate-room
  GEMINI_VISION_MODEL   Gemini model for bbox detection (default: gemini-2.0-flash)
"""

import asyncio
import base64
import json
import os
from collections import Counter
from io import BytesIO
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

router = APIRouter(tags=["segmentation"])

# ── Furniture vocabulary used as the Grounded SAM text prompt ──────────────
_FURNITURE_PROMPT = (
    "sofa . couch . armchair . chair . dining chair . office chair . stool . bench . ottoman . "
    "coffee table . dining table . side table . end table . console table . desk . "
    "bed . headboard . nightstand . dresser . wardrobe . bookshelf . bookcase . "
    "cabinet . TV stand . media console . "
    "floor lamp . table lamp . pendant lamp . chandelier . "
    "rug . carpet . curtain . blinds . "
    "plant . artwork . painting . mirror . vase . decoration"
)


# ── Pydantic response models ────────────────────────────────────────────────

class BoundingBox(BaseModel):
    x_min: float   # normalised 0-1
    y_min: float
    x_max: float
    y_max: float


class FurnitureFeatures(BaseModel):
    dominant_colors: list[str]  # e.g. ["#8b4513", "#d2b48c", "#ffffff"]
    area_pct: float             # percentage of total image area
    aspect_ratio: float         # bbox width / height


class FurnitureItem(BaseModel):
    id: str
    label: str
    confidence: float
    bbox: BoundingBox
    crop_image: str             # data:image/png;base64,…
    features: FurnitureFeatures


class SegmentationResult(BaseModel):
    items: list[FurnitureItem]
    counts: dict[str, int]      # {"sofa": 1, "chair": 2, …}
    total: int
    method: str                 # "grounded_sam" | "gemini_vision"


# ── Image / colour helpers ──────────────────────────────────────────────────

def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"


def _dominant_hex_colors(crop: Image.Image, n: int = 3) -> list[str]:
    """Return n dominant hex colours via median-cut quantisation."""
    thumb = crop.resize((60, 60), Image.LANCZOS).convert("RGB")
    quantized = thumb.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    return [
        f"#{palette[i * 3]:02x}{palette[i * 3 + 1]:02x}{palette[i * 3 + 2]:02x}"
        for i in range(n)
    ]


def _crop_to_data_url(crop: Image.Image) -> str:
    buf = BytesIO()
    crop.save(buf, format="PNG")
    return _build_data_url(buf.getvalue(), "image/png")


def _build_item(
    img: Image.Image,
    label: str,
    confidence: float,
    idx: int,
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    bbox_padding: float = 0.0,
) -> FurnitureItem:
    W, H = img.size

    # Expand bbox by padding (fraction of image dimension) on each side
    x_min_pad = max(0.0, x_min - bbox_padding)
    y_min_pad = max(0.0, y_min - bbox_padding)
    x_max_pad = min(1.0, x_max + bbox_padding)
    y_max_pad = min(1.0, y_max + bbox_padding)

    px = (int(x_min_pad * W), int(y_min_pad * H), int(x_max_pad * W), int(y_max_pad * H))
    crop = img.crop(px)
    w_px = max(px[2] - px[0], 1)
    h_px = max(px[3] - px[1], 1)
    return FurnitureItem(
        id=str(idx),
        label=label,
        confidence=round(confidence, 3),
        bbox=BoundingBox(
            x_min=round(x_min, 4),
            y_min=round(y_min, 4),
            x_max=round(x_max, 4),
            y_max=round(y_max, 4),
        ),
        crop_image=_crop_to_data_url(crop),
        features=FurnitureFeatures(
            dominant_colors=_dominant_hex_colors(crop),
            area_pct=round(w_px * h_px / (W * H) * 100, 2),
            aspect_ratio=round(w_px / h_px, 3),
        ),
    )


# ── Replicate polling ───────────────────────────────────────────────────────

async def _poll_replicate(
    client: httpx.AsyncClient,
    prediction: dict,
    token: str,
    max_polls: int = 24,
) -> dict:
    for _ in range(max_polls):
        status = prediction.get("status")
        if status == "succeeded":
            return prediction
        if status in {"failed", "canceled"}:
            raise HTTPException(
                502,
                f"Replicate prediction {status}: {prediction.get('error')}",
            )
        get_url = prediction.get("urls", {}).get("get")
        if not get_url:
            return prediction
        await asyncio.sleep(5)
        poll = await client.get(get_url, headers={"Authorization": f"Bearer {token}"})
        poll.raise_for_status()
        prediction = poll.json()
    raise HTTPException(504, "Replicate SAM prediction timed out.")


# ── Grounded SAM 2 via Replicate ────────────────────────────────────────────

def _parse_grounded_sam_output(output: Any) -> list[dict]:
    """
    Normalise Grounded SAM 2 output into:
      [{"label": str, "confidence": float, "box": [x1,y1,x2,y2]}, …]

    Handles two common Replicate output shapes:
      Shape A — {"detections": [{"label":…, "score":…, "box":…}, …]}
      Shape B — {"boxes": […], "labels": […], "scores": […]}
    """
    if not isinstance(output, dict):
        return []

    if "detections" in output:
        return [
            {
                "label": d.get("label", "furniture"),
                "confidence": float(d.get("score", d.get("confidence", 0.8))),
                "box": d.get("box", d.get("bbox", [])),
            }
            for d in output["detections"]
        ]

    boxes = output.get("boxes", [])
    labels = output.get("labels", [])
    scores = output.get("scores", [])
    return [
        {
            "label": labels[i] if i < len(labels) else "furniture",
            "confidence": float(scores[i]) if i < len(scores) else 0.8,
            "box": box,
        }
        for i, box in enumerate(boxes)
    ]


async def _segment_with_grounded_sam(
    image_bytes: bytes, mime_type: str
) -> list[dict]:
    token = os.getenv("REPLICATE_API_TOKEN", "")
    model = os.getenv("SAM_REPLICATE_MODEL", "adirik/grounded-sam-2")

    if not token:
        raise HTTPException(500, "REPLICATE_API_TOKEN is not configured.")

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            resp = await client.post(
                f"https://api.replicate.com/v1/models/{model}/predictions",
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Prefer": "wait=60",
                },
                json={
                    "input": {
                        "image": _build_data_url(image_bytes, mime_type),
                        "text_input": _FURNITURE_PROMPT,
                        "box_threshold": float(os.getenv("SAM_BOX_THRESHOLD", "0.30")),
                        "text_threshold": float(os.getenv("SAM_TEXT_THRESHOLD", "0.25")),
                    }
                },
            )
            resp.raise_for_status()
            prediction = await _poll_replicate(client, resp.json(), token)
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Grounded SAM failed: {exc.response.text}") from exc
    except Exception as exc:
        raise HTTPException(502, f"Grounded SAM failed: {exc}") from exc

    return _parse_grounded_sam_output(prediction.get("output") or {})


# ── Gemini Vision fallback ──────────────────────────────────────────────────

async def _detect_with_gemini(image_bytes: bytes, mime_type: str) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")

    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY is not configured.")

    prompt = (
        "Analyze this interior design image. Identify every distinct furniture or decor item.\n"
        "Return ONLY a JSON array — no markdown, no extra text. Each element:\n"
        '  "label": string  (e.g. "sofa", "coffee table", "floor lamp")\n'
        '  "bbox": [x_min, y_min, x_max, y_max]  (floats 0–1 relative to image)\n'
        '  "confidence": float 0–1\n'
        "If nothing is found, return []."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
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
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Gemini detection failed: {exc.response.text}") from exc

    text = "".join(
        part.get("text", "")
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    )

    try:
        items = json.loads(text)
        if not isinstance(items, list):
            items = []
    except Exception:
        items = []

    return [
        {
            "label": str(item.get("label", "furniture")),
            "confidence": float(item.get("confidence", 0.8)),
            "box": item["bbox"],
        }
        for item in items
        if isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4
    ]


# ── Orchestrator ────────────────────────────────────────────────────────────

async def run_segmentation(
    image_bytes: bytes,
    mime_type: str,
    bbox_padding: float | None = None,
) -> SegmentationResult:
    """
    Public helper — also called from main.py's /generate-room when segment=true.

    bbox_padding: expand each detected box by this fraction of image size on
    every side before cropping (e.g. 0.05 = 5%).  Defaults to BBOX_PADDING
    env var, then 0.02.
    """
    if bbox_padding is None:
        bbox_padding = float(os.getenv("BBOX_PADDING", "0.02"))

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    W, H = img.size

    use_sam = bool(
        os.getenv("REPLICATE_API_TOKEN")
        and os.getenv("SAM_REPLICATE_MODEL", "adirik/grounded-sam-2")
    )
    method = "grounded_sam" if use_sam else "gemini_vision"

    try:
        raw = (
            await _segment_with_grounded_sam(image_bytes, mime_type)
            if use_sam
            else await _detect_with_gemini(image_bytes, mime_type)
        )
    except HTTPException:
        raw = await _detect_with_gemini(image_bytes, mime_type)
        method = "gemini_vision"

    items: list[FurnitureItem] = []
    for idx, det in enumerate(raw):
        box = det.get("box", [])
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = (float(v) for v in box)

        # Normalise pixel coords if needed (SAM often returns pixels)
        if x2 > 1.5:
            x1, y1, x2, y2 = x1 / W, y1 / H, x2 / W, y2 / H

        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            continue

        items.append(
            _build_item(img, det["label"], det["confidence"], idx, x1, y1, x2, y2, bbox_padding)
        )

    counts = dict(Counter(item.label for item in items))
    return SegmentationResult(items=items, counts=counts, total=len(items), method=method)


# ── Route ───────────────────────────────────────────────────────────────────

@router.post("/segment-furniture", response_model=SegmentationResult)
async def segment_furniture(
    image: UploadFile = File(...),
    bbox_padding: float = Form(default=None, description="Expand each bbox by this fraction (0–0.2). Overrides BBOX_PADDING env var."),
):
    """
    Segment furniture items from an interior design image.

    Returns each detected item with:
    - `label` & `confidence`
    - Normalised `bbox` (0-1)
    - `crop_image` — base64 PNG of the cropped item
    - `features` — dominant colours, area %, aspect ratio

    The `counts` summary e.g. {"sofa": 1, "chair": 3} is useful for
    determining how many of each furniture type to look up in the product
    catalogue for matching.
    """
    raw_bytes = await image.read()
    mime_type = image.content_type or "image/png"

    try:
        Image.open(BytesIO(raw_bytes)).load()
    except Exception as exc:
        raise HTTPException(400, "Uploaded file is not a valid image.") from exc

    return await run_segmentation(raw_bytes, mime_type, bbox_padding=bbox_padding)
