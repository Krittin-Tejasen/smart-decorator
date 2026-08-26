"""
furniture_segmentation.py
─────────────────────────
POST /segment-furniture
  Accepts a room image (multipart), returns each detected furniture item with
  a labelled bounding box, a cropped image, a per-item segmentation mask,
  dominant colours, and shape features — ready for downstream product matching.

Pipeline (when REPLICATE_API_TOKEN is configured):

    Generated image
          │
          ▼
    Grounding DINO   (open-vocabulary text-prompted object detection)
          │
          │ "bed", "desk", "chair", "lamp", ...
          ▼
    2D bounding boxes
          │
          ▼
    SAM 2            (boxes used as prompts for mask prediction)
          │
          ▼
    Individual furniture masks   (one binary mask per detected item)

Each result is a `FurnitureItem` with its own `mask_image` (base64 PNG, same
size as the source image, white = furniture pixel / black = background).
The full result is returned in the HTTP response AND written to disk as JSON
under `SEGMENTATION_RESULTS_DIR` for later reuse (e.g. product matching).

Detection back-ends (chosen automatically via env vars):
  • Grounding DINO + SAM 2 on Replicate  — used when REPLICATE_API_TOKEN is set
  • Gemini Vision fallback                — always available when GEMINI_API_KEY is set,
                                             also used if the Replicate stages fail.
    The Gemini fallback can only produce bounding boxes, so its "masks" are a
    rectangular approximation of the box — each item's `mask_precise` flag
    tells you which kind of mask you got.

Env vars
────────
  GROUNDING_DINO_MODEL   Replicate model slug for stage 1 (default: adirik/grounding-dino)
  SAM2_MODEL             Replicate model slug for stage 2 (default: meta/sam-2)
  SAM_BOX_THRESHOLD      Confidence threshold for detection boxes (default: 0.30)
  SAM_TEXT_THRESHOLD     Text-matching threshold for Grounding DINO (default: 0.25)
  SEGMENTATION_RESULTS_DIR  Folder to persist each result as JSON (default: segmentation_results)
  REPLICATE_API_TOKEN    Already used by /generate-room
  GEMINI_API_KEY         Already used by /generate-room
  GEMINI_VISION_MODEL    Gemini model for bbox detection fallback (default: gemini-2.0-flash)
"""

import asyncio
import base64
import json
import os
import uuid
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image, ImageDraw
from pydantic import BaseModel

router = APIRouter(tags=["segmentation"])

# ── Furniture vocabulary used as the Grounding DINO text prompt ────────────
_FURNITURE_PROMPT = (
    "sofa . couch . armchair . chair . dining chair . office chair . stool . bench . ottoman . "
    "coffee table . dining table . side table . end table . console table . desk . "
    "bed . headboard . nightstand . dresser . wardrobe . bookshelf . bookcase . "
    "cabinet . TV stand . media console . "
    "floor lamp . table lamp . pendant lamp . chandelier . "
    "rug . carpet . curtain . blinds . "
    "plant . artwork . painting . mirror . vase . decoration"
)

_RESULTS_DIR = Path(os.getenv("SEGMENTATION_RESULTS_DIR", "segmentation_results"))


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
    crop_image: str             # data:image/png;base64,… (cropped to bbox)
    mask_image: str             # data:image/png;base64,… (full-size binary mask, L mode)
    mask_precise: bool          # True = real SAM 2 mask, False = bbox rectangle approximation
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


def _pil_to_data_url(image: Image.Image, mode: str | None = None) -> str:
    buf = BytesIO()
    (image.convert(mode) if mode else image).save(buf, format="PNG")
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
    mask_img: Image.Image | None = None,
) -> FurnitureItem:
    W, H = img.size
    px = (int(x_min * W), int(y_min * H), int(x_max * W), int(y_max * H))
    crop = img.crop(px)
    w_px = max(px[2] - px[0], 1)
    h_px = max(px[3] - px[1], 1)

    if mask_img is not None:
        mask = mask_img.convert("L").resize((W, H))
        mask_precise = True
    else:
        # No real SAM 2 mask available — approximate with a filled bbox rectangle.
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rectangle(px, fill=255)
        mask_precise = False

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
        crop_image=_pil_to_data_url(crop),
        mask_image=_pil_to_data_url(mask, mode="L"),
        mask_precise=mask_precise,
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
    raise HTTPException(504, "Replicate prediction timed out.")


async def _run_replicate_model(
    client: httpx.AsyncClient, model: str, token: str, model_input: dict
) -> Any:
    resp = await client.post(
        f"https://api.replicate.com/v1/models/{model}/predictions",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Prefer": "wait=60",
        },
        json={"input": model_input},
    )
    resp.raise_for_status()
    prediction = await _poll_replicate(client, resp.json(), token)
    return prediction.get("output")


# ── Stage 1: Grounding DINO (image + text prompt → boxes) ──────────────────

def _parse_detection_output(output: Any) -> list[dict]:
    """
    Normalise a Grounding DINO-style output into:
      [{"label": str, "confidence": float, "box": [x1,y1,x2,y2]}, …]

    Handles the common Replicate output shapes:
      Shape A — {"detections": [{"label":…, "score":…, "box":…}, …]}
      Shape B — {"boxes": […], "labels": […], "scores": […]}
      Shape C — {"json_data": "<json string with a list or {'annotations': […]}>"}
    """
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except Exception:
            return []

    if isinstance(output, list):
        return [
            {
                "label": str(d.get("label", d.get("class_name", "furniture"))),
                "confidence": float(d.get("score", d.get("confidence", 0.8))),
                "box": d.get("box", d.get("bbox", [])),
            }
            for d in output
            if isinstance(d, dict)
        ]

    if not isinstance(output, dict):
        return []

    if "detections" in output:
        return [
            {
                "label": str(d.get("label", "furniture")),
                "confidence": float(d.get("score", d.get("confidence", 0.8))),
                "box": d.get("box", d.get("bbox", [])),
            }
            for d in output["detections"]
        ]

    if "json_data" in output:
        try:
            parsed = json.loads(output["json_data"])
            annotations = parsed.get("annotations", parsed) if isinstance(parsed, dict) else parsed
            return _parse_detection_output(annotations)
        except Exception:
            return []

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


async def _detect_with_grounding_dino(image_bytes: bytes, mime_type: str) -> list[dict]:
    token = os.getenv("REPLICATE_API_TOKEN", "")
    model = os.getenv("GROUNDING_DINO_MODEL", "adirik/grounding-dino")

    if not token:
        raise HTTPException(500, "REPLICATE_API_TOKEN is not configured.")

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            output = await _run_replicate_model(
                client,
                model,
                token,
                {
                    "image": _build_data_url(image_bytes, mime_type),
                    "query": _FURNITURE_PROMPT,
                    "box_threshold": float(os.getenv("SAM_BOX_THRESHOLD", "0.30")),
                    "text_threshold": float(os.getenv("SAM_TEXT_THRESHOLD", "0.25")),
                },
            )
    except HTTPException:
        raise
    except httpx.HTTPStatusError as exc:
        raise HTTPException(502, f"Grounding DINO failed: {exc.response.text}") from exc
    except Exception as exc:
        raise HTTPException(502, f"Grounding DINO failed: {exc}") from exc

    return _parse_detection_output(output or {})


# ── Stage 2: SAM 2 (image + boxes → one mask per box) ───────────────────────

async def _resolve_mask_image(
    client: httpx.AsyncClient, ref: Any, token: str
) -> Image.Image | None:
    if isinstance(ref, dict):
        ref = ref.get("url") or ref.get("mask") or ref.get("image")
    if not isinstance(ref, str) or not ref:
        return None

    try:
        if ref.startswith("http://") or ref.startswith("https://"):
            resp = await client.get(ref, headers={"Authorization": f"Bearer {token}"})
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("L")
        if ref.startswith("data:"):
            _, encoded = ref.split(",", 1)
            return Image.open(BytesIO(base64.b64decode(encoded))).convert("L")
        return Image.open(BytesIO(base64.b64decode(ref))).convert("L")
    except Exception:
        return None


async def _segment_boxes_with_sam2(
    image_bytes: bytes, mime_type: str, boxes_px: list[list[float]]
) -> list[Image.Image | None]:
    """
    Prompts SAM 2 with the Grounding DINO boxes and returns one mask per box,
    in the same order as `boxes_px`. Returns an all-None list (same length)
    if the model output can't be reliably matched 1:1 to the input boxes —
    callers then fall back to a bbox-rectangle mask for that request.
    """
    if not boxes_px:
        return []

    token = os.getenv("REPLICATE_API_TOKEN", "")
    model = os.getenv("SAM2_MODEL", "meta/sam-2")

    try:
        async with httpx.AsyncClient(timeout=180) as client:
            output = await _run_replicate_model(
                client,
                model,
                token,
                {
                    "image": _build_data_url(image_bytes, mime_type),
                    "input_boxes": json.dumps(boxes_px),
                },
            )
            if not isinstance(output, dict):
                return [None] * len(boxes_px)

            mask_refs = output.get("individual_masks") or output.get("masks") or []
            if len(mask_refs) != len(boxes_px):
                # Model didn't honour the box prompts 1:1 — don't risk mismatched
                # mask/label pairing, let the caller fall back to bbox masks.
                return [None] * len(boxes_px)

            return [await _resolve_mask_image(client, ref, token) for ref in mask_refs]
    except Exception:
        return [None] * len(boxes_px)


# ── Gemini Vision fallback (boxes only, no true mask) ───────────────────────

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

def _persist_result(result: SegmentationResult) -> str | None:
    """Write the full segmentation result (incl. per-item masks) to disk as JSON."""
    try:
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = _RESULTS_DIR / f"{timestamp}_{uuid.uuid4().hex[:8]}.json"
        path.write_text(json.dumps(result.model_dump(), indent=2))
        return str(path)
    except Exception as exc:
        print(f"[segmentation] failed to persist result: {exc}")
        return None


async def run_segmentation(image_bytes: bytes, mime_type: str) -> SegmentationResult:
    """
    Public helper — also called from main.py's /generate-room when segment=true.

    Runs Grounding DINO (boxes) → SAM 2 (masks) when REPLICATE_API_TOKEN is
    configured, falling back to Gemini Vision (boxes only) otherwise or if
    either Replicate stage fails.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    W, H = img.size

    use_sam = bool(os.getenv("REPLICATE_API_TOKEN"))
    method = "grounded_sam" if use_sam else "gemini_vision"
    mask_imgs: list[Image.Image | None] = []

    try:
        if use_sam:
            raw = await _detect_with_grounding_dino(image_bytes, mime_type)
            boxes_px = []
            for det in raw:
                box = det.get("box", [])
                if len(box) != 4:
                    boxes_px.append([0, 0, 0, 0])
                    continue
                x1, y1, x2, y2 = (float(v) for v in box)
                # SAM 2 expects pixel coordinates; normalise-looking boxes (<=1) are scaled up.
                if x2 <= 1.5 and y2 <= 1.5:
                    x1, y1, x2, y2 = x1 * W, y1 * H, x2 * W, y2 * H
                boxes_px.append([x1, y1, x2, y2])
            mask_imgs = await _segment_boxes_with_sam2(image_bytes, mime_type, boxes_px) if raw else []
        else:
            raw = await _detect_with_gemini(image_bytes, mime_type)
    except HTTPException:
        raw = await _detect_with_gemini(image_bytes, mime_type)
        method = "gemini_vision"

    if len(mask_imgs) != len(raw):
        mask_imgs = [None] * len(raw)

    items: list[FurnitureItem] = []
    for idx, (det, mask_img) in enumerate(zip(raw, mask_imgs)):
        box = det.get("box", [])
        if len(box) != 4:
            continue

        x1, y1, x2, y2 = (float(v) for v in box)

        # Normalise pixel coords if needed (SAM/DINO often return pixels)
        if x2 > 1.5:
            x1, y1, x2, y2 = x1 / W, y1 / H, x2 / W, y2 / H

        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            continue

        items.append(
            _build_item(img, det["label"], det["confidence"], idx, x1, y1, x2, y2, mask_img)
        )

    counts = dict(Counter(item.label for item in items))
    result = SegmentationResult(items=items, counts=counts, total=len(items), method=method)
    _persist_result(result)
    return result


# ── Route ───────────────────────────────────────────────────────────────────

@router.post("/segment-furniture", response_model=SegmentationResult)
async def segment_furniture(image: UploadFile = File(...)):
    """
    Segment furniture items from an interior design image.

    Pipeline: Grounding DINO detects each furniture item as a labelled box,
    then SAM 2 turns each box into a precise pixel mask — one mask per item.

    Returns each detected item with:
    - `label` & `confidence`
    - Normalised `bbox` (0-1)
    - `crop_image` — base64 PNG of the cropped item
    - `mask_image` — base64 PNG binary mask (full image size) for that one item
    - `mask_precise` — True if the mask came from SAM 2, False if it's a bbox approximation
    - `features` — dominant colours, area %, aspect ratio

    The full result is also written to disk as JSON under
    `SEGMENTATION_RESULTS_DIR` (default: `segmentation_results/`).

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

    return await run_segmentation(raw_bytes, mime_type)
