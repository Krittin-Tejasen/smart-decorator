"""
furniture_segmentation.py
─────────────────────────
POST /segment-furniture
  Accepts a room image (multipart), returns each detected furniture item with
  a labelled bounding box, a cropped image, a per-item segmentation mask,
  dominant colours, and shape features — ready for downstream product matching.

Pipeline (runs locally, no external API — when torch/transformers/sam2 + a
SAM 2 checkpoint are installed):

    Generated image
          │
          ▼
    Grounding DINO   (open-vocabulary text-prompted object detection, via
                       transformers' AutoModelForZeroShotObjectDetection)
          │
          │ "bed", "desk", "chair", "lamp", ...
          ▼
    2D bounding boxes
          │
          ▼
    SAM 2            (boxes used as prompts for mask prediction, via the
                       `sam2` package's SAM2ImagePredictor)
          │
          ▼
    Individual furniture masks   (one binary mask per detected item)

Each result is a `FurnitureItem` with its own `mask_image` (base64 PNG, same
size as the source image, white = furniture pixel / black = background).
The full result is returned in the HTTP response AND written to disk as JSON
under `SEGMENTATION_RESULTS_DIR` for later reuse (e.g. product matching).

Everything runs on whatever machine hosts this FastAPI process — the phone
client only uploads a photo and downloads the JSON result, so its hardware
is irrelevant. Device selection is automatic (`cuda` if available, else
`cpu`); CPU inference works but is much slower.

Detection back-ends (chosen automatically):
  • Grounding DINO + SAM 2, run locally — used whenever the dependencies
    (torch, transformers, sam2) are installed and a SAM 2 checkpoint is
    present on disk.
  • Gemini Vision fallback — always available when GEMINI_API_KEY is set,
    also used if the local pipeline isn't set up yet or fails at runtime.
    The Gemini fallback can only produce bounding boxes, so its "masks" are
    a rectangular approximation of the box — each item's `mask_precise`
    flag tells you which kind of mask you got.

Setup (one-time, per machine that runs this backend)
──────────────────────────────────────────────────────
  pip install torch torchvision transformers sam2
  # (On a CPU-only machine, prefer the smaller CPU wheel instead:
  #  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu)

  Download a SAM 2.1 checkpoint into backend/checkpoints/, e.g. the "small"
  variant (best CPU/GPU tradeoff, ~180MB):
      curl -L -o checkpoints/sam2.1_hiera_small.pt \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

  Grounding DINO's weights (IDEA-Research/grounding-dino-tiny, ~660MB) are
  downloaded automatically from the Hugging Face Hub on first use — no
  manual step needed for that half of the pipeline.

Env vars
────────
  GROUNDING_DINO_MODEL      Hugging Face model id (default: IDEA-Research/grounding-dino-tiny)
  SAM2_CHECKPOINT           Path to a local SAM 2.1 checkpoint (default: checkpoints/sam2.1_hiera_small.pt)
  SAM2_MODEL_CFG            Matching hydra config name (default: configs/sam2.1/sam2.1_hiera_s.yaml)
  SEGMENTATION_DEVICE       Force "cuda" or "cpu" (default: auto-detect via torch.cuda.is_available())
  SAM_BOX_THRESHOLD         Confidence threshold for detection boxes (default: 0.30)
  SAM_TEXT_THRESHOLD        Text-matching threshold for Grounding DINO (default: 0.25)
  SEGMENTATION_RESULTS_DIR  Folder to persist each result as JSON (default: segmentation_results)
  GEMINI_API_KEY            Already used by /generate-room
  GEMINI_VISION_MODEL       Gemini model for bbox detection fallback (default: gemini-2.0-flash)
"""

import asyncio
import base64
import contextlib
import json
import os
import threading
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
# Grounding DINO requires lowercase phrases separated by " . " and a
# trailing period.
_FURNITURE_PROMPT = (
    "sofa . couch . armchair . chair . dining chair . office chair . stool . bench . ottoman . "
    "coffee table . dining table . side table . end table . console table . desk . "
    "bed . headboard . nightstand . dresser . wardrobe . bookshelf . bookcase . "
    "cabinet . tv stand . media console . "
    "floor lamp . table lamp . pendant lamp . chandelier . "
    "rug . carpet . curtain . blinds . "
    "plant . artwork . painting . mirror . vase . decoration ."
)

_RESULTS_DIR = Path(os.getenv("SEGMENTATION_RESULTS_DIR", "segmentation_results"))

_LOCAL_DEPS_HINT = (
    "Local segmentation dependencies are missing or not set up. "
    "Run: pip install torch torchvision transformers sam2, and download a SAM 2 "
    "checkpoint — see the Furniture Segmentation section of the README."
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
    crop_image: str             # data:image/png;base64,… (cropped to bbox)
    mask_image: str             # data:image/png;base64,… (full-size binary mask, L mode)
    mask_precise: bool          # True = real SAM 2 mask, False = bbox rectangle approximation
    features: FurnitureFeatures


class SegmentationResult(BaseModel):
    items: list[FurnitureItem]
    counts: dict[str, int]      # {"sofa": 1, "chair": 2, …}
    total: int
    method: str                 # "grounding_dino_sam2" | "gemini_vision"


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


# ── Local model loading (lazy singletons, loaded once per process) ─────────

_model_lock = threading.Lock()
_grounding_dino_cache: tuple[Any, Any, str] | None = None
_sam2_predictor_cache: tuple[Any, str] | None = None


def _get_device() -> str:
    override = os.getenv("SEGMENTATION_DEVICE")
    if override:
        return override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_grounding_dino() -> tuple[Any, Any, str]:
    """Returns (processor, model, device), loading the model on first call."""
    global _grounding_dino_cache
    if _grounding_dino_cache is not None:
        return _grounding_dino_cache

    with _model_lock:
        if _grounding_dino_cache is None:
            try:
                from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
            except ImportError as exc:
                raise HTTPException(500, _LOCAL_DEPS_HINT) from exc

            model_id = os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-tiny")
            device = _get_device()
            processor = AutoProcessor.from_pretrained(model_id)
            model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)
            model.eval()
            _grounding_dino_cache = (processor, model, device)

    return _grounding_dino_cache


def _get_sam2_predictor() -> tuple[Any, str]:
    """Returns (SAM2ImagePredictor, device), loading the model on first call."""
    global _sam2_predictor_cache
    if _sam2_predictor_cache is not None:
        return _sam2_predictor_cache

    with _model_lock:
        if _sam2_predictor_cache is None:
            try:
                from sam2.build_sam import build_sam2
                from sam2.sam2_image_predictor import SAM2ImagePredictor
            except ImportError as exc:
                raise HTTPException(500, _LOCAL_DEPS_HINT) from exc

            checkpoint = os.getenv("SAM2_CHECKPOINT", "checkpoints/sam2.1_hiera_small.pt")
            model_cfg = os.getenv("SAM2_MODEL_CFG", "configs/sam2.1/sam2.1_hiera_s.yaml")
            if not Path(checkpoint).exists():
                raise HTTPException(
                    500,
                    f"SAM 2 checkpoint not found at '{checkpoint}'. Download it first — "
                    "see the Furniture Segmentation section of the README.",
                )

            device = _get_device()
            sam2_model = build_sam2(model_cfg, checkpoint, device=device)
            _sam2_predictor_cache = (SAM2ImagePredictor(sam2_model), device)

    return _sam2_predictor_cache


# ── Stage 1: Grounding DINO (image + text prompt → boxes), run locally ─────

def _detect_with_grounding_dino_local(img: Image.Image) -> list[dict]:
    import torch

    processor, model, device = _get_grounding_dino()
    box_threshold = float(os.getenv("SAM_BOX_THRESHOLD", "0.30"))
    text_threshold = float(os.getenv("SAM_TEXT_THRESHOLD", "0.25"))

    inputs = processor(images=img, text=_FURNITURE_PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=[img.size[::-1]],  # (height, width)
    )[0]

    labels = results.get("text_labels") or results.get("labels") or []
    boxes = results["boxes"].tolist()   # pixel xyxy, already scaled to img size
    scores = results["scores"].tolist()

    return [
        {"label": str(label) or "furniture", "confidence": float(score), "box": [float(v) for v in box]}
        for box, score, label in zip(boxes, scores, labels)
    ]


# ── Stage 2: SAM 2 (image + boxes → one mask per box), run locally ─────────

def _segment_boxes_with_sam2_local(
    img: Image.Image, boxes_px: list[list[float]]
) -> list[Image.Image | None]:
    if not boxes_px:
        return []

    import numpy as np
    import torch

    predictor, device = _get_sam2_predictor()
    image_np = np.array(img.convert("RGB"))

    autocast_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device == "cuda"
        else contextlib.nullcontext()
    )

    masks_out: list[Image.Image | None] = []
    with torch.inference_mode(), autocast_ctx:
        predictor.set_image(image_np)
        for box in boxes_px:
            try:
                masks, _scores, _logits = predictor.predict(
                    box=np.array(box), multimask_output=False
                )
                mask_arr = (masks[0] > 0).astype("uint8") * 255
                masks_out.append(Image.fromarray(mask_arr, mode="L"))
            except Exception as exc:
                print(f"[segmentation] SAM 2 failed on box {box}: {exc}")
                masks_out.append(None)

    return masks_out


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

    Runs Grounding DINO (boxes) → SAM 2 (masks) locally, falling back to
    Gemini Vision (boxes only) if the local dependencies/checkpoint aren't
    set up yet, or if either local stage fails at runtime.
    """
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    W, H = img.size

    method = "grounding_dino_sam2"
    mask_imgs: list[Image.Image | None] = []

    try:
        raw = await asyncio.to_thread(_detect_with_grounding_dino_local, img)
        boxes_px = [det["box"] for det in raw]
        mask_imgs = (
            await asyncio.to_thread(_segment_boxes_with_sam2_local, img, boxes_px) if raw else []
        )
    except HTTPException:
        raw = await _detect_with_gemini(image_bytes, mime_type)
        method = "gemini_vision"
    except Exception as exc:
        print(f"[segmentation] local Grounding DINO / SAM 2 pipeline failed: {exc}")
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

        # Normalise pixel coords if needed (Grounding DINO/Gemini return pixels or 0-1).
        if x2 > 1.5 or y2 > 1.5:
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
    Both models run locally on whatever machine hosts this backend.

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
