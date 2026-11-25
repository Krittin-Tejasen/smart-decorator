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
size as the source image, white = furniture pixel / black = background), a
small `crop_image` (bbox crop, JPEG thumbnail) and — when the mask is a real
SAM 2 mask — a `cutout_image` (transparent-background PNG trimmed to the
mask, ready to drop onto a product-style card).
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

  Grounding DINO's weights (IDEA-Research/grounding-dino-base, ~1GB) are
  downloaded automatically from the Hugging Face Hub on first use — no
  manual step needed for that half of the pipeline.

Env vars
────────
  GROUNDING_DINO_MODEL      Hugging Face model id (default: IDEA-Research/grounding-dino-base;
                            on the 24-image answer key it found 40/52 curtains where tiny
                            found 23, at detection F1 0.843 vs 0.800)
  SAM2_CHECKPOINT           Path to a local SAM 2.1 checkpoint (default: checkpoints/sam2.1_hiera_small.pt)
  SAM2_MODEL_CFG            Matching hydra config name (default: configs/sam2.1/sam2.1_hiera_s.yaml)
  SEGMENTATION_DEVICE       Force "cuda" or "cpu" (default: auto-detect via torch.cuda.is_available())
  SAM_BOX_THRESHOLD         Confidence threshold for detection boxes (default: 0.25, tuned for
                            grounding-dino-base: 0.30 lost recall, 0.20 let in too much junk)
  SAM_TEXT_THRESHOLD        Text-matching threshold for Grounding DINO (default: 0.25)
  SAM_BOX_PAD               Grow each DINO box by this share of its size before SAM 2 (default: 0,
                            DINO's own box; padding mostly pulled in wall and floor)
  SAM_MULTIMASK             1 = ask SAM 2 for 3 candidate masks and pick the largest likely one,
                            0 = SAM 2's single default mask (default: 0; the largest candidate
                            tended to add pillows, plants behind a lamp, and floor)
  SAM_MASK_SCORE_MARGIN     With SAM_MULTIMASK=1: how far below the best candidate's score a
                            larger mask may be and still be picked (default: 0.15)
  SEGMENTATION_LABELER      Who names each piece after SAM 2: "gemini" (default: one batched
                            Gemini call per room, label_verification.py; CLIP for pieces it
                            doesn't answer), "clip" (offline rule only) or "none" (DINO's word)
  GEMINI_LABEL_MODEL        Gemini model for that (default: GEMINI_VISION_MODEL)
  SEGMENTATION_CLIP_RELABEL 0 = skip the CLIP second opinion on labels (default: 1; also
                            skipped when sentence-transformers isn't installed)
  CLIP_MODEL                sentence-transformers CLIP model for that (default: clip-ViT-B-32)
  SEGMENTATION_RESULTS_DIR  Folder to persist each result as JSON (default: segmentation_results)
  GEMINI_API_KEY            Already used by /generate-room
  GEMINI_VISION_MODEL       Gemini model for bbox detection fallback (default: gemini-3.6-flash)
"""

import asyncio
import base64
import contextlib
import inspect
import json
import os
import re
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
    "sofa . couch . armchair . chair . dining chair . office chair . bench . ottoman . stool . "
    "coffee table . dining table . side table . console table . desk . "
    "bed . headboard . nightstand . dresser . wardrobe . bookshelf . tv_bench . "
    # Not "tv stand": DINO put it on nightstands, desks and consoles (47 cards
    # in 56 test rooms, 2 before), and matching sends it to TV benches.
    "cabinet . "
    "floor lamp . table lamp . pendant lamp . chandelier . "
    "rug . carpet . curtain . blinds . "
    "plant pot . artwork . mirror . vase . decoration . "
    # Soft furnishings: folded into the sofa, bed or chair they lie on, never a
    # card. Named so DINO doesn't call them a stool or a sofa of their own.
    "pillow . cushion . throw pillow . "
    # Named so DINO doesn't call the duvet a "cushion".
    "blanket . "
    # Decoys, not furniture: without them Grounding DINO forces a door onto the
    # nearest furniture word ("mirror") and a window onto "curtain". Giving it
    # the truthful label lets us drop those detections below. They're what the
    # generator must preserve, not products to shop for.
    # "doorway" is the opening itself: what stands in it, no lower than its
    # sill, is outside the room (the garden chair seen through an open door).
    "doorway . door . window . "
    # Appliance decoys, same idea: named so a wall-mounted TV isn't forced onto
    # "artwork"/"mirror" or a fridge onto "wardrobe", then dropped below.
    # Appliances are never shoppable cards, in the original photo or the
    # generated one.
    "tv . television . computer . refrigerator . microwave . oven . washing machine . "
    "air conditioner ."
)

_VOCAB: tuple[str, ...] = tuple(p.strip() for p in _FURNITURE_PROMPT.split(".") if p.strip())
_VOCAB_SET = frozenset(_VOCAB)
_MAX_PHRASE_WORDS = max(len(p.split()) for p in _VOCAB)

# Architecture / fixtures that must never show up as a shoppable card — whether
# they came from the DINO decoys above or from Gemini's "every item" answer.
_NON_FURNITURE_LABELS = frozenset({"doorway", "door", "window", "ceiling light", "air vent", "vent"})
# Appliances and electronics, matched anywhere in a label so Gemini's free
# text ("flat screen tv", "stainless steel fridge") is caught too. Never a card.
_APPLIANCE_RE = re.compile(
    r"\b(tvs?|television|flat ?screen|tv screen|monitor|computer|laptop|printer|projector|"
    r"refrigerator|fridge|freezer|microwave|oven|stove|cooktop|hob|range hood|cooker|dishwasher|"
    r"washing machine|washer|dryer|air conditioner|air conditioning|ac unit|air purifier|"
    r"humidifier|dehumidifier|heater|radiator|fan|ceiling fan|speakers?|soundbar|subwoofer|"
    r"stereo|game console|kettle|toaster|coffee (machine|maker)|blender|vacuum|appliances?)\b"
)
# TV furniture is furniture, not an appliance.
_TV_FURNITURE_RE = re.compile(r"\btv (stand|bench|unit|cabinet|console|table)\b|\bmedia (console|unit|cabinet)\b")
# Openings something outside the room can be seen through.
_OPENING_LABELS = frozenset({"doorway", "door"})

# Never a card of their own: they join the mask of the sofa, bed or chair they
# lie on (merge_parts_into_hosts), or are dropped.
_MASK_ONLY_LABELS = frozenset({"pillow", "cushion", "throw pillow"})
_PART_ONLY_LABELS = frozenset({"blanket", "headboard"})
# Parts only a bed takes (a sofa or chair has no headboard).
_BED_PART_LABELS = frozenset({"headboard"})
# What pillows are cut out of. Not chairs: DINO calls a seat "cushion" too.
_PILLOW_HOSTS = frozenset({"sofa", "couch", "bed"})

# Labels of the same "family" are treated as the same physical object when
# their boxes overlap heavily (Grounding DINO often reports one chair twice as
# "armchair" and "office chair"). Tables and storage are one family on purpose:
# DINO regularly splits a console/cabinet into a "table" box and a "cabinet" box.
_LABEL_FAMILY: dict[str, str] = {
    **dict.fromkeys(["armchair", "chair", "dining chair", "office chair", "stool"], "seat"),
    **dict.fromkeys(["sofa", "couch"], "sofa"),
    **dict.fromkeys(
        [
            "coffee table", "dining table", "side table", "end table", "console table",
            "desk", "nightstand", "dresser", "wardrobe", "bookshelf", "bookcase",
            "cabinet", "tv stand", "media console",
        ],
        "table_or_storage",
    ),
    **dict.fromkeys(["floor lamp", "table lamp", "pendant lamp", "chandelier"], "light"),
    **dict.fromkeys(["rug", "carpet"], "floor_covering"),
    **dict.fromkeys(["curtain", "blinds"], "window_covering"),
    **dict.fromkeys(["artwork", "painting", "mirror"], "wall_decor"),
    # Pampas in a vase comes back as a "plant pot" box around a "vase" box.
    **dict.fromkeys(["vase", "plant pot"], "vessel"),
}

# A "sofa" covering 0.15% of the photo is a mis-detection (a cushion, a fold in
# a blanket), not a sofa. Only for pieces that can't plausibly be tiny — small
# decor (vases, lamps) legitimately covers well under 1%.
_MIN_AREA_PCT_BY_LABEL: dict[str, float] = {
    "sofa": 1.0,
    "couch": 1.0,
    "bed": 2.0,
    "wardrobe": 1.0,
    "dining table": 1.0,
    "bookshelf": 0.5,
    "bookcase": 0.5,
}

# Thumbnails keep the response (and the phone) light: the raw crops used to be
# full resolution, up to ~285KB of base64 each.
_THUMB_PX = 320

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
    crop_image: str             # data:image/jpeg;base64,… (bbox crop, thumbnail ≤ _THUMB_PX)
    # data:image/png;base64,… transparent-background cut-out trimmed to the mask
    # (thumbnail ≤ _THUMB_PX). None when there is no real mask — clients should
    # fall back to `crop_image`.
    cutout_image: str | None = None
    mask_image: str             # data:image/png;base64,… (full-size binary mask, L mode)
    mask_precise: bool          # True = real SAM 2 mask, False = bbox rectangle approximation
    features: FurnitureFeatures
    # Who named the piece: "gemini" (label_verification), "clip" (offline
    # fallback) or "dino" (unchanged). dino_label is DINO's word when changed.
    label_source: str = "dino"
    dino_label: str | None = None


class SegmentationResult(BaseModel):
    items: list[FurnitureItem]
    counts: dict[str, int]      # {"sofa": 1, "chair": 2, …}
    total: int
    method: str                 # "grounding_dino_sam2" | "gemini_vision" | "none"
    # Why the local pipeline wasn't used (None when it was). The local stage
    # used to fail silently and fall back to a weaker detector, which hid a
    # real bug for weeks — surface it instead.
    fallback_reason: str | None = None


# ── Image / colour helpers ──────────────────────────────────────────────────

def _build_data_url(image_bytes: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode()}"


def _dominant_hex_colors(
    crop: Image.Image, n: int = 3, mask: Image.Image | None = None
) -> list[str]:
    """Return n dominant hex colours via median-cut quantisation.

    With a real mask (same size as `crop`) only the furniture's own pixels
    count — otherwise a sofa's "dominant colour" is mostly the wall behind it.
    """
    if mask is not None:
        import numpy as np

        selected = np.asarray(crop.convert("RGB")).reshape(-1, 3)[
            np.asarray(mask.convert("L")).reshape(-1) > 127
        ]
        if len(selected) >= 64:
            if len(selected) > 4000:
                selected = selected[:: len(selected) // 4000 + 1]
            thumb = Image.fromarray(selected.reshape(-1, 1, 3).astype("uint8"), "RGB")
            return _quantize_to_hex(thumb, n)

    return _quantize_to_hex(crop.resize((60, 60), Image.LANCZOS).convert("RGB"), n)


def _quantize_to_hex(rgb: Image.Image, n: int) -> list[str]:
    """Up to n hex colours, most common first. May return fewer than n for
    images that simply don't have that many distinct colours (a flat crop used
    to raise IndexError here)."""
    quantized = rgb.quantize(colors=n, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette() or []
    by_frequency = sorted(quantized.getcolors() or [], reverse=True)  # (pixel count, palette index)
    colors = [
        f"#{palette[i * 3]:02x}{palette[i * 3 + 1]:02x}{palette[i * 3 + 2]:02x}"
        for _count, i in by_frequency[:n]
    ]
    return colors or ["#000000"]


def _pil_to_data_url(image: Image.Image, mode: str | None = None) -> str:
    buf = BytesIO()
    (image.convert(mode) if mode else image).save(buf, format="PNG")
    return _build_data_url(buf.getvalue(), "image/png")


def _thumbnail(image: Image.Image, max_px: int = _THUMB_PX) -> Image.Image:
    thumb = image.copy()
    thumb.thumbnail((max_px, max_px), Image.LANCZOS)
    return thumb


def _jpeg_data_url(image: Image.Image) -> str:
    buf = BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=88)
    return _build_data_url(buf.getvalue(), "image/jpeg")


# A SAM 2 mask covering less than this share of its own detection box is almost
# always a mis-segmentation (a handle, a shadow) — treat it as "no real mask"
# rather than hand the client an empty-looking cut-out.
_MIN_MASK_COVERAGE = 0.05


def _cutout_from_mask(
    img: Image.Image, mask: Image.Image, box_px: tuple[int, int, int, int]
) -> Image.Image | None:
    """Transparent-background cut-out of `img` under `mask`, trimmed to the
    mask's own extent inside `box_px`. None if the mask is empty/too small."""
    region = mask.crop(box_px).point(lambda v: 255 if v > 127 else 0)
    tight = region.getbbox()
    if tight is None:
        return None

    box_area = max((box_px[2] - box_px[0]) * (box_px[3] - box_px[1]), 1)
    covered = sum(region.histogram()[255:])
    if covered / box_area < _MIN_MASK_COVERAGE:
        return None

    x0, y0 = box_px[0], box_px[1]
    trimmed = (x0 + tight[0], y0 + tight[1], x0 + tight[2], y0 + tight[3])
    rgba = img.convert("RGBA")
    rgba.putalpha(mask)
    return rgba.crop(trimmed)


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

    cutout: Image.Image | None = None
    mask: Image.Image | None = None
    if mask_img is not None:
        candidate = mask_img.convert("L").resize((W, H))
        cutout = _cutout_from_mask(img, candidate, px)
        if cutout is not None:
            mask = candidate

    if mask is not None:
        mask_precise = True
        colors = _dominant_hex_colors(crop, mask=mask.crop(px))
    else:
        # No usable SAM 2 mask — approximate with a filled bbox rectangle.
        mask = Image.new("L", (W, H), 0)
        ImageDraw.Draw(mask).rectangle(px, fill=255)
        mask_precise = False
        colors = _dominant_hex_colors(crop)

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
        crop_image=_jpeg_data_url(_thumbnail(crop)),
        cutout_image=_pil_to_data_url(_thumbnail(cutout)) if cutout is not None else None,
        mask_image=_pil_to_data_url(mask, mode="L"),
        mask_precise=mask_precise,
        features=FurnitureFeatures(
            dominant_colors=colors,
            area_pct=round(w_px * h_px / (W * H) * 100, 2),
            aspect_ratio=round(w_px / h_px, 3),
        ),
    )


# ── Label / box post-processing (pure functions — unit-tested) ─────────────

def canonicalize_label(raw: str) -> str:
    """Grounding DINO glues every prompt phrase that matched an object into one
    string ("armchair chair chair office chair", "rug carpet"). Keep only the
    first vocabulary phrase (longest match at that position) so a card reads
    "armchair", not the whole pile.

    Architecture wins: DINO's label is every phrase that matched, so a
    "blinds window" is a window and a "wardrobe door" is a door — those come
    back as "window"/"door" and get dropped by is_furniture_label. (Trade-off:
    a real wardrobe or curtain whose label happens to include one of those two
    words is dropped too; better than a door shown as a product card.)

    Only for Grounding DINO output: Gemini's labels are already clean, and
    running this on e.g. "desk lamp" would wrongly return "desk".
    """
    words = raw.lower().split()
    # DINO decodes word pieces: "head ##board" is "headboard", and a lone
    # "##board" lost its first half.
    joined: list[str] = []
    for w in words:
        if w.startswith("##") and joined:
            joined[-1] += w[2:]
        else:
            joined.append(w)
    words = joined
    for architecture in ("doorway", "door", "window"):
        if architecture in words:
            return architecture
    for i in range(len(words)):
        for n in range(min(_MAX_PHRASE_WORDS, len(words) - i), 0, -1):
            phrase = " ".join(words[i : i + n])
            if phrase in _VOCAB_SET:
                return phrase
    # Half a word ("plant" of "plant pot", "book" of "bookshelf", "##board"
    # of "headboard"): the vocabulary phrase it starts or ends.
    for w in words:
        piece = w.lstrip("#")
        if not piece:
            continue
        for phrase in _VOCAB:
            if w.startswith("##") and phrase.endswith(piece):
                return phrase
            if not w.startswith("##") and phrase.startswith(piece):
                return phrase
    return " ".join(words) or "furniture"


def is_appliance_label(label: str) -> bool:
    label = label.strip().lower()
    return bool(_APPLIANCE_RE.search(label)) and not _TV_FURNITURE_RE.search(label)


def is_furniture_label(label: str) -> bool:
    return is_candidate_label(label) and label.strip().lower() not in _MASK_ONLY_LABELS | _PART_ONLY_LABELS


def is_mask_only_label(label: str) -> bool:
    return label.strip().lower() in _MASK_ONLY_LABELS


def is_candidate_label(label: str) -> bool:
    """Worth a SAM 2 mask: furniture, or something cut out of / folded into it.
    Not architecture, not appliances."""
    return label.strip().lower() not in _NON_FURNITURE_LABELS and not is_appliance_label(label)


def _box_iou_and_containment(a: tuple, b: tuple) -> tuple[float, float]:
    """IoU and intersection/smaller-area for two normalised (x1, y1, x2, y2) boxes."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union else 0.0), (inter / smaller if smaller else 0.0)


def _mask_iou_and_containment(a, b) -> tuple[float, float]:
    """IoU and intersection/smaller-area for two bool masks."""
    inter = int((a & b).sum())
    area_a, area_b = int(a.sum()), int(b.sum())
    union = area_a + area_b - inter
    smaller = min(area_a, area_b)
    return (inter / union if union else 0.0), (inter / smaller if smaller else 0.0)


# A box holding this much of each of two or more separate pieces of its own
# kind is a box on the group (a gallery wall, a pair of curtains, two vases on
# a console), not a piece: as a card it hid the real ones.
_GROUP_MEMBER_CONTAINMENT = 0.8
_GROUP_MEMBERS_MAX_IOU = 0.3
# Only kinds that hang or stand in sets. Not storage (a bookshelf holding the
# cabinets DINO boxed inside it) and not lights (a floor lamp's two shades).
_GROUP_FAMILIES = frozenset({"wall_decor", "window_covering", "vessel"})


def drop_group_boxes(dets: list[dict], masks: dict) -> list[dict]:
    family = lambda d: _LABEL_FAMILY.get(d["label"], d["label"])
    groups = set()
    for g in dets:
        mg = masks[id(g)]
        if mg is None or family(g) not in _GROUP_FAMILIES:
            continue
        members = []
        for d in dets:
            md = masks[id(d)]
            if d is g or md is None or family(d) != family(g) or md.sum() >= 0.7 * mg.sum():
                continue
            if (md & mg).sum() / max(md.sum(), 1) >= _GROUP_MEMBER_CONTAINMENT:
                members.append(md)
        separate = any(
            _mask_iou_and_containment(a, b)[0] < _GROUP_MEMBERS_MAX_IOU
            and (a & b).sum() < 0.5 * min(a.sum(), b.sum())
            for i, a in enumerate(members) for b in members[i + 1:]
        )
        if separate:
            groups.add(id(g))
    return [d for d in dets if id(d) not in groups]


def dedupe_detections(dets: list[dict]) -> list[dict]:
    """Greedy NMS over detections shaped {"label", "confidence", "box": (x1,y1,x2,y2)},
    optionally with a SAM 2 "mask".

    Two detections are the same object when they overlap heavily AND look like
    the same kind of thing (same label family) — a pillow sitting inside a
    sofa's box must survive, a chair reported twice must not. With masks on
    both, the overlap is measured on the masks: dining chairs set one behind
    another have overlapping boxes but separate masks. Without masks (Gemini
    fallback), on the boxes.

    Highest confidence goes first; when a duplicate has the larger mask it
    replaces the one kept, so the card shows the most complete view.
    """
    import numpy as np

    def as_bool(mask):
        return None if mask is None else np.asarray(mask.convert("L")) > 127

    masks = {id(d): as_bool(d.get("mask")) for d in dets}
    dets = drop_group_boxes(dets, masks)
    kept: list[dict] = []
    for det in sorted(dets, key=lambda d: d["confidence"], reverse=True):
        family = _LABEL_FAMILY.get(det["label"], det["label"])
        duplicate = None
        for i, other in enumerate(kept):
            m_det, m_other = masks[id(det)], masks[id(other)]
            if m_det is not None and m_other is not None:
                iou, containment = _mask_iou_and_containment(m_det, m_other)
            else:
                iou, containment = _box_iou_and_containment(det["box"], other["box"])
            same_family = family == _LABEL_FAMILY.get(other["label"], other["label"])
            if (same_family and (iou > 0.4 or containment > 0.8)) or iou > 0.8:
                duplicate = i
                break
        if duplicate is None:
            kept.append(det)
        else:
            m_det, m_other = masks[id(det)], masks[id(kept[duplicate])]
            if family == "vessel":
                # A vase gets one box on the vessel and one on the vessel with
                # its flowers or plant; the card is the container only, so the
                # smaller mask (and its box) wins.
                if m_det is not None and m_other is not None and m_det.sum() < m_other.sum():
                    kept[duplicate] = {**kept[duplicate], "box": det["box"], "mask": det["mask"]}
                    masks[id(kept[duplicate])] = m_det
            elif m_det is not None and m_other is not None and m_det.sum() > m_other.sum():
                # The more complete view, under the more confident name.
                # Keep the union box: a bed deduped against its headboard must
                # keep the bed's box, or the duvet no longer sits inside it.
                a, b = kept[duplicate]["box"], det["box"]
                union = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                kept[duplicate] = {**kept[duplicate], "box": union, "mask": det["mask"]}
                masks[id(kept[duplicate])] = m_det
    return kept


def gemini_box_to_xyxy(box: list) -> tuple[float, float, float, float] | None:
    """Convert a Gemini `box_2d` to normalised (x_min, y_min, x_max, y_max).

    Gemini's native order is [ymin, xmin, ymax, xmax] on a 0-1000 grid. The old
    prompt asked for [x_min, y_min, x_max, y_max] but Gemini kept answering in
    its own order, so every fallback crop landed on the wrong region.
    """
    try:
        ymin, xmin, ymax, xmax = (float(v) for v in box)
    except (TypeError, ValueError):
        return None

    scale = 1000.0 if max(ymin, xmin, ymax, xmax) > 1.5 else 1.0
    x1, x2 = sorted((xmin / scale, xmax / scale))
    y1, y2 = sorted((ymin / scale, ymax / scale))
    return x1, y1, x2, y2


def parse_gemini_items(items: list) -> list[dict]:
    """Turn Gemini's JSON array into detection dicts with a normalised xyxy box."""
    detections: list[dict] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_box = item.get("box_2d") or item.get("bbox")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        box = gemini_box_to_xyxy(raw_box)
        if box is None:
            continue
        detections.append(
            {
                "label": str(item.get("label", "furniture")).strip().lower() or "furniture",
                "confidence": float(item.get("confidence", 0.8)),
                "box": list(box),
            }
        )
    return detections


# ── SAM 2 mask selection / clean-up (pure functions — unit-tested) ─────────
# A box-only prompt with multimask_output=False often came back with one part
# of the piece (a sofa's cushions, a bed's frame) and DINO's tight box clipped
# legs and arms; the raw mask also carried stray blobs of wall/floor and the
# lamp standing on a nightstand. These helpers fix that between SAM 2 and
# _build_item. Masks here are numpy bool arrays, (H, W), full image size.

# Below this share of the largest blob, a connected region is a stray blob
# (a patch of wall, a shadow), not part of the piece.
_MIN_COMPONENT_SHARE = 0.25
# Enclosed holes smaller than this share of the mask are SAM 2 noise and get
# filled; bigger ones (the gap in a chair back) are real and stay. At 2% every
# opening of an open shelf counted as noise and the wall behind was filled in.
# The pipeline doesn't fill holes: a hole is usually wall seen through the piece.
_MAX_HOLE_SHARE = 0.005
# A mask with more than this share of its pixels outside the padded box has
# grabbed the wall or floor around the piece.
_MAX_MASK_LEAK = 0.2
# Flat on the floor: a rug blends into the floor around it, so padding its
# (already large) box let the mask spill onto bare floor. Rugs are also always
# behind whatever stands on them.
_FLOOR_LABELS = {"rug", "carpet"}
# A box must be smaller than this share of another's area to count as sitting
# in front of it (lamp on a nightstand). Two similar-sized boxes are left to
# split_contested: "smaller is in front" handed a dining table's pixels to the
# sideboard box.
_MAX_FRONT_AREA_RATIO = 0.4
# What two similar-sized pieces still share after resolve_overlaps (the office
# chair behind a desk, its box over the desk's front) is asked of SAM 2 again,
# per connected patch. Patches smaller than this follow the pair's largest one.
_MIN_CONTESTED_PX = 50


def pad_box(box: list[float], W: int, H: int, pad: float) -> list[float]:
    """Grow a pixel (x1, y1, x2, y2) box by `pad` × its size on every side,
    clipped to the image. DINO's boxes are tight and SAM 2 rarely paints outside
    the box it's given, so a chair leg past the box edge was cut off."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(W), x2 + dx), min(float(H), y2 + dy)]


def _box_slices(box: list[float]) -> tuple[slice, slice]:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    return slice(y1, y2), slice(x1, x2)


def select_best_mask(masks, scores, box: list[float], margin: float = 0.15) -> int:
    """Index of the candidate that is the WHOLE piece.

    SAM 2's three candidates for a box are roughly subpart / part / whole.
    Drop the ones that spill well outside the box (they grabbed the wall), then
    among those scored within `margin` of the best, take the largest: when SAM
    is nearly as sure of the whole sofa as of its cushions, we want the sofa.
    """
    rows, cols = _box_slices(box)
    candidates = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        total = int(mask.sum())
        if total == 0:
            continue
        inside = int(mask[rows, cols].sum())
        if 1 - inside / total > _MAX_MASK_LEAK:
            continue
        candidates.append((i, float(score), inside))

    if not candidates:
        return max(range(len(scores)), key=lambda i: float(scores[i]))

    top = max(score for _i, score, _area in candidates)
    return max(
        (c for c in candidates if c[1] >= top - margin), key=lambda c: c[2]
    )[0]


def clean_mask(mask, box: list[float], fill_holes: bool = True):
    """Clip to the (padded) box, drop stray blobs, fill pin-holes."""
    import cv2
    import numpy as np

    rows, cols = _box_slices(box)
    clipped = np.zeros_like(mask, dtype=bool)
    clipped[rows, cols] = mask[rows, cols]
    if not clipped.any():
        return clipped

    n, labels, stats, _ = cv2.connectedComponentsWithStats(clipped.astype("uint8"), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = [i + 1 for i, area in enumerate(areas) if area >= _MIN_COMPONENT_SHARE * areas.max()]
    cleaned = np.isin(labels, keep)
    if not fill_holes:
        return cleaned

    # Holes = background regions that don't touch the border of the box.
    region = ~cleaned[rows, cols]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(region.astype("uint8"), connectivity=4)
    h, w = region.shape
    max_hole = _MAX_HOLE_SHARE * cleaned.sum()
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if not touches_border and area <= max_hole:
            cleaned[rows, cols][labels == i] = True
    return cleaned


def resolve_overlaps(dets: list[dict]) -> None:
    """Give pixels claimed by two masks to the piece in front, in place.

    `dets` are {"box": normalised xyxy, "mask": bool array or None, "label"
    optional}. Only ever removes pixels from the piece behind; nothing gains
    any, so each cut-out holds one piece for product matching. A rug is always
    behind: it gives shared pixels to anything and takes none. Otherwise a box
    well under the other's size is in front (the lamp on the nightstand, the
    vase at the foot of a floor lamp) and the larger one
    gives those pixels up. Similar-sized boxes keep their own masks here;
    split_contested settles them.
    """
    for a in dets:
        for b in dets:
            if a is b or a["mask"] is None or b["mask"] is None:
                continue
            a_floor = a.get("label") in _FLOOR_LABELS
            b_floor = b.get("label") in _FLOOR_LABELS
            if b_floor:
                continue  # a rug never takes pixels
            if b.get("label") in _MASK_ONLY_LABELS | _PART_ONLY_LABELS:
                continue  # a pillow or throw is part of what it lies on
            area_a = (a["box"][2] - a["box"][0]) * (a["box"][3] - a["box"][1])
            area_b = (b["box"][2] - b["box"][0]) * (b["box"][3] - b["box"][1])
            # A bed's mask (the union of SAM 2's candidates) reaches over what
            # stands at its foot and sides; any smaller piece keeps its pixels.
            if not a_floor and a.get("label") not in _UNION_LABELS:
                if area_b >= area_a * _MAX_FRONT_AREA_RATIO:
                    continue  # not clearly in front
            a["mask"] = a["mask"] & ~b["mask"]


def pick_owner(candidates, mask_a, mask_b) -> int:
    """0 if SAM 2's whole-object answer for a shared patch (the largest of its
    candidates) matches mask_a at least as well (IoU) as mask_b, else 1.
    Asked at the desk's front, the whole object is the desk, not the chair
    behind it. The best match over all candidates was fooled by part answers:
    the flowers alone matched a "chandelier" box on them better than the vase
    with its flowers."""
    import numpy as np

    whole = max(candidates, key=lambda c: int(np.count_nonzero(c)))

    def iou(mask) -> float:
        return float(np.logical_and(whole, mask).sum()) / max(int(np.logical_or(whole, mask).sum()), 1)

    return 0 if iou(mask_a) >= iou(mask_b) else 1


def split_contested(dets: list[dict], ask=None) -> None:
    """Leave no pixel in two masks, in place. Run after resolve_overlaps.

    Each connected patch two masks still share goes to one of them: `ask(x, y)`
    returns SAM 2's candidate masks for a point at the patch's most interior
    pixel, and pick_owner says whose mask they match. Patches under
    _MIN_CONTESTED_PX follow the pair's largest patch. Without `ask`, or when
    it fails, the higher-confidence piece keeps the patch. Pillows and throws
    are left in their host (they are part of it, never a card). A piece that
    lost a patch also drops the slivers left around it (the edge of the frame
    next door), by clean_mask's stray-blob rule, so its cut-out isn't stretched.
    """
    import cv2
    import numpy as np

    losers: list[list] = []  # [piece, pixels it lost]
    for i, a in enumerate(dets):
        for b in dets[i + 1:]:
            if a["mask"] is None or b["mask"] is None:
                continue
            if {a.get("label"), b.get("label")} & (_MASK_ONLY_LABELS | _PART_ONLY_LABELS):
                continue
            shared = a["mask"] & b["mask"]
            if not shared.any():
                continue
            mask_a, mask_b = a["mask"], b["mask"]
            n, labels, stats, _ = cv2.connectedComponentsWithStats(shared.astype("uint8"), connectivity=8)
            patches = sorted(range(1, n), key=lambda k: -stats[k, cv2.CC_STAT_AREA])
            owner_by_default = 0 if a.get("confidence", 0) >= b.get("confidence", 0) else 1
            for k in patches:
                patch = labels == k
                owner = owner_by_default
                if ask is not None and (k == patches[0] or stats[k, cv2.CC_STAT_AREA] >= _MIN_CONTESTED_PX):
                    try:
                        depth = cv2.distanceTransform(patch.astype("uint8"), cv2.DIST_L2, 5)
                        y, x = np.unravel_index(int(depth.argmax()), depth.shape)
                        owner = pick_owner(ask(int(x), int(y)), mask_a, mask_b)
                    except Exception as exc:
                        print(f"[segmentation] SAM 2 point prompt failed: {exc}")
                    if k == patches[0]:
                        owner_by_default = owner
                loser = b if owner == 0 else a
                loser["mask"] = loser["mask"] & ~patch
                for entry in losers:
                    if entry[0] is loser:
                        entry[1] = entry[1] | patch
                        break
                else:
                    losers.append([loser, patch])

    # Only blobs next to what was taken: a chair's seat that SAM 2 already
    # painted apart from its back stays.
    for d, taken in losers:
        if not d["mask"].any():
            continue
        near = cv2.dilate(taken.astype("uint8"), np.ones((5, 5), np.uint8)) > 0
        n, labels, stats, _ = cv2.connectedComponentsWithStats(d["mask"].astype("uint8"), connectivity=8)
        biggest = stats[1:, cv2.CC_STAT_AREA].max()
        drop = [
            k for k in range(1, n)
            if stats[k, cv2.CC_STAT_AREA] < _MIN_COMPONENT_SHARE * biggest and near[labels == k].any()
        ]
        if drop:
            d["mask"] = d["mask"] & ~np.isin(labels, drop)


def mask_bbox(mask, W: int, H: int) -> tuple[float, float, float, float] | None:
    """Normalised (x1, y1, x2, y2) extent of a bool mask, None if empty."""
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return float(xs.min() / W), float(ys.min() / H), float((xs.max() + 1) / W), float((ys.max() + 1) / H)


# ── Local model loading (lazy singletons, loaded once per process) ─────────

_model_lock = threading.Lock()
_inference_lock = threading.Lock()
_grounding_dino_cache: tuple[Any, Any, str] | None = None
_sam2_predictor_cache: tuple[Any, str] | None = None
_clip_cache: Any = None


def _get_device() -> str:
    override = os.getenv("SEGMENTATION_DEVICE")
    if override:
        return override
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _get_clip() -> Any:
    """clip-ViT-B-32 (sentence-transformers), the model product matching uses;
    None when sentence-transformers isn't installed or SEGMENTATION_CLIP_RELABEL=0."""
    global _clip_cache
    if os.getenv("SEGMENTATION_CLIP_RELABEL", "1") == "0":
        return None
    if _clip_cache is None:
        with _model_lock:
            if _clip_cache is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError:
                    return None
                _clip_cache = SentenceTransformer(os.getenv("CLIP_MODEL", "clip-ViT-B-32"), device=_get_device())
    return _clip_cache


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

            model_id = os.getenv("GROUNDING_DINO_MODEL", "IDEA-Research/grounding-dino-base")
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

# A second, doorway-only pass: in the full prompt DINO often boxes just the
# door leaf and misses the opening beside it. Labelled "doorway" either way.
_OPENING_PROMPT = "doorway . open door ."
# Doorways score lower than furniture; the opening filters (real_openings and
# the sill check) catch what a lower bar lets in.
_OPENING_BOX_THRESHOLD = 0.25

# No separate curtain pass: a "curtain . blinds ." prompt has no "window"
# decoy, so it boxed bare windows as curtains (23 wrong cards on the 24-image
# answer key, F1 unchanged). grounding-dino-base finds the curtains from the
# main prompt instead. See docs/curtain-dino-handoff.md.

# A third pass for what hangs on or stands against the wall. grounding-dino-base
# spreads its scores over the ~50 phrases of the main prompt, and artwork falls
# under the bar there (24-image answer key: 19/27 artworks scored >= 0.25 in
# the main prompt, 26/27 in this one); this pass took artwork back from 15 to
# 19/27 at 3 wrong cards. Keep "table lamp"-like look-alikes in mind before
# shortening it: "artwork . mirror . vase ." alone called 20 lamp bases vases.
_WALL_PROMPT = "artwork . painting . picture frame . mirror . vase ."
_WALL_BOX_THRESHOLD = 0.30
_WALL_ARTWORK_WORDS = frozenset({"painting", "picture", "frame"})


def wall_pass_label(label: str) -> str | None:
    """The main-vocabulary name for a wall-pass box: "painting" / "picture
    frame" are artwork. None for a box DINO gave no text ("furniture")."""
    if label in _VOCAB_SET:
        return label
    if _WALL_ARTWORK_WORDS & set(label.split()):
        return "artwork"
    return None


def _detect_with_grounding_dino_local(img: Image.Image) -> list[dict]:
    openings = [
        {**d, "label": "doorway"}
        for d in _run_grounding_dino(img, _OPENING_PROMPT, _OPENING_BOX_THRESHOLD)
    ]
    wall = [
        {**d, "label": label}
        for d in _run_grounding_dino(img, _WALL_PROMPT, _WALL_BOX_THRESHOLD)
        if (label := wall_pass_label(d["label"])) is not None
    ]
    # Same piece from two passes: dedupe_detections merges it (same family).
    return _run_grounding_dino(img, _FURNITURE_PROMPT) + openings + wall


def _run_grounding_dino(img: Image.Image, prompt: str, box_threshold: float | None = None) -> list[dict]:
    import torch

    processor, model, device = _get_grounding_dino()
    if box_threshold is None:
        box_threshold = float(os.getenv("SAM_BOX_THRESHOLD", "0.25"))
    text_threshold = float(os.getenv("SAM_TEXT_THRESHOLD", "0.25"))

    # The predictor/model objects are shared singletons — one inference at a
    # time (two overlapping requests would otherwise race on SAM 2's
    # set_image state and mix up each other's masks).
    with _inference_lock:
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            **_post_process_threshold_kwargs(processor, box_threshold, text_threshold),
            target_sizes=[img.size[::-1]],  # (height, width)
        )[0]

    labels = results.get("text_labels") or results.get("labels") or []
    boxes = results["boxes"].tolist()   # pixel xyxy, already scaled to img size
    scores = results["scores"].tolist()

    return [
        {
            "label": canonicalize_label(str(label)),
            "confidence": float(score),
            "box": [float(v) for v in box],
        }
        for box, score, label in zip(boxes, scores, labels)
    ]


def _post_process_threshold_kwargs(processor: Any, box_threshold: float, text_threshold: float) -> dict:
    """transformers renamed `box_threshold` to `threshold` in
    post_process_grounded_object_detection (5.x). requirements.txt doesn't pin
    transformers, so a fresh `pip install` silently broke the whole local
    pipeline — pass whichever name the installed version accepts."""
    params = inspect.signature(processor.post_process_grounded_object_detection).parameters
    key = "threshold" if "threshold" in params else "box_threshold"
    return {key: box_threshold, "text_threshold": text_threshold}


# ── Stage 2: SAM 2 (image + boxes → one mask per box), run locally ─────────

# For a bed box SAM 2's single mask is the duvet and frame, without the
# headboard; its three candidates split the bed into headboard / body / whole
# (speckled). Together they are the whole bed. Tried on 18 bedrooms.
_UNION_LABELS = frozenset({"bed"})


def union_of_candidates(predictor: Any, box: list[float]):
    """Union of SAM 2's three candidate masks for a box (set_image already done)."""
    import numpy as np

    masks, _scores, _logits = predictor.predict(box=np.array(box), multimask_output=True)
    return np.logical_or.reduce(masks > 0)


def _autocast(device: str):
    import torch

    return torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else contextlib.nullcontext()


def _segment_boxes_with_sam2_local(
    img: Image.Image, boxes_px: list[list[float]], labels: list[str] | None = None
) -> list[Image.Image | None]:
    if not boxes_px:
        return []

    import numpy as np
    import torch

    predictor, device = _get_sam2_predictor()
    image_np = np.array(img.convert("RGB"))
    W, H = img.size
    box_pad = float(os.getenv("SAM_BOX_PAD", "0"))
    multimask = os.getenv("SAM_MULTIMASK", "0") != "0"
    margin = float(os.getenv("SAM_MASK_SCORE_MARGIN", "0.15"))

    masks_out: list[Image.Image | None] = []
    with _inference_lock, torch.inference_mode(), _autocast(device):
        predictor.set_image(image_np)
        for box, label in zip(boxes_px, labels or [""] * len(boxes_px)):
            try:
                floor = label in _FLOOR_LABELS
                padded = list(box) if floor else pad_box(box, W, H, box_pad)
                masks, scores, _logits = predictor.predict(
                    box=np.array(padded), multimask_output=multimask
                )
                masks = masks > 0
                best = select_best_mask(masks, scores, padded, margin) if multimask else 0
                mask = masks[best]
                if label in _UNION_LABELS:
                    mask = mask | union_of_candidates(predictor, padded)
                mask_arr = clean_mask(mask, padded, fill_holes=False).astype("uint8") * 255
                masks_out.append(Image.fromarray(mask_arr, mode="L"))
            except Exception as exc:
                print(f"[segmentation] SAM 2 failed on box {box}: {exc}")
                masks_out.append(None)

    return masks_out


# ── Gemini Vision fallback (boxes only, no true mask) ───────────────────────

async def _detect_with_gemini(image_bytes: bytes, mime_type: str) -> list[dict]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")

    if not api_key:
        raise HTTPException(500, "GEMINI_API_KEY is not configured.")

    # Gemini's native box format is [ymin, xmin, ymax, xmax] on a 0-1000 grid.
    # Ask for exactly that (and say the order out loud) instead of a custom
    # x-first format it doesn't follow — see gemini_box_to_xyxy.
    prompt = (
        "Analyze this interior design image. Identify every distinct furniture or decor item.\n"
        "Do not include doors, windows or ceiling fixtures.\n"
        "Do not include TVs, screens, electronics or any appliances (fridge, oven, "
        "washing machine, air conditioner, fan, speakers…). A TV stand is furniture.\n"
        "Return ONLY a JSON array — no markdown, no extra text. Each element:\n"
        '  "label": string  (e.g. "sofa", "coffee table", "floor lamp")\n'
        '  "box_2d": [ymin, xmin, ymax, xmax]  (integers on a 0-1000 grid relative to the '
        "image; note the order: y first, then x)\n"
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

    return parse_gemini_items(items)


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


def select_detections(raw: list[dict], mask_imgs: list, W: int, H: int) -> tuple[list[dict], list[dict]]:
    """Raw detections (+ their SAM 2 masks, or None) → (furniture, mask_only).

    Normalises boxes, drops non-furniture and implausibly small pieces, and
    merges the same object reported twice (chair as "armchair" + "office
    chair"). Pillows, cushions and blankets skip dedupe — a cushion box must
    never knock out its sofa, and SAM 2 paints a duvet box as nearly the whole
    bed — then join the piece they lie on. The second list is kept for
    callers and is always empty now. Anything seen through an open doorway
    is outside the room and dropped.
    """
    detections: list[dict] = []
    openings: list[tuple] = []
    tvs: list[tuple] = []
    for det, mask_img in zip(raw, mask_imgs):
        box = det.get("box", [])
        if len(box) != 4:
            continue
        if det["label"] in _TV_LABELS and float(det.get("confidence", 0.0)) >= _TV_MIN_CONFIDENCE:
            x1, y1, x2, y2 = (float(v) for v in box)
            if x2 > 1.5 or y2 > 1.5:
                x1, y1, x2, y2 = x1 / W, y1 / H, x2 / W, y2 / H
            tvs.append((x1, y1, x2, y2))
        is_opening = det["label"] in _OPENING_LABELS
        if not (is_opening or is_candidate_label(det["label"])):
            continue

        x1, y1, x2, y2 = (float(v) for v in box)

        # Normalise pixel coords if needed (Grounding DINO returns pixels;
        # the Gemini path already hands over 0-1).
        if x2 > 1.5 or y2 > 1.5:
            x1, y1, x2, y2 = x1 / W, y1 / H, x2 / W, y2 / H

        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        if x2 - x1 < 0.01 or y2 - y1 < 0.01:
            continue
        if is_opening:
            openings.append((x1, y1, x2, y2))
            continue
        if (x2 - x1) * (y2 - y1) * 100 < _MIN_AREA_PCT_BY_LABEL.get(det["label"], 0.0):
            continue

        detections.append(
            {
                "label": det["label"],
                "confidence": det["confidence"],
                "box": (x1, y1, x2, y2),
                "mask": mask_img,
            }
        )

    openings = real_openings(openings, [d["box"] for d in detections])
    detections = [
        d for d in detections
        if d["label"] not in _OUTDOOR_LABELS or not seen_through_opening(d["box"], openings)
    ]
    detections = drop_reflections(detections)
    soft =[d for d in detections if d["label"] in _SOFT_FURNISHING_LABELS]
    furniture = dedupe_detections([d for d in detections if d["label"] not in _SOFT_FURNISHING_LABELS])
    furniture = merge_parts_into_hosts(furniture + soft)
    # A pillow or throw that lies on nothing found is dropped, never a card.
    furniture = [d for d in furniture if d["label"] not in _SOFT_FURNISHING_LABELS]
    relabel_bedside_tables(furniture)
    relabel_by_position(furniture, tvs, W / H)
    return furniture, []


# A mirror shows a second copy of the room, and DINO finds furniture in it (the
# table lamp reflected in a floor mirror). A piece is a reflection when its box
# lies this far inside a mirror's box, it reaches no lower than the mirror's
# bottom edge (+ this share of the image height; what stands in front of a
# wall mirror, like a vase on the console under it, reaches below), and the
# same kind of piece is also found outside the mirror: the real one. A piece
# with no twin outside stays, so nothing real in front of a floor mirror goes
# missing just for standing there.
_REFLECTION_CONTAINMENT = 0.8
_REFLECTION_BOTTOM_MARGIN = 0.01
# Real mirrors in the test rooms scored 0.54-0.82; weak "mirrors" were other
# things (a laptop screen 0.32, a dark bedside lamp 0.37, a headboard 0.30),
# and a real lamp beside the bed went missing behind them.
_REFLECTION_MIN_MIRROR_CONFIDENCE = 0.5


def drop_reflections(dets: list[dict]) -> list[dict]:
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    beds = [d["box"] for d in dets if d["label"] == "bed"]
    # A "mirror" at a bed's head is its headboard (merge_parts_into_hosts),
    # and the pillows and lamps in front of it are real.
    mirrors = [
        d["box"] for d in dets
        if d["label"] == "mirror"
        and float(d.get("confidence", 1.0)) >= _REFLECTION_MIN_MIRROR_CONFIDENCE
        and not any(is_bed_head(d["box"], b) for b in beds)
    ]
    if not mirrors:
        return dets

    def mirror_holding(box: tuple) -> tuple | None:
        for m in mirrors:
            if box is m or area(box) >= area(m):
                continue
            if (
                _box_iou_and_containment(box, m)[1] >= _REFLECTION_CONTAINMENT
                and box[3] <= m[3] + _REFLECTION_BOTTOM_MARGIN
            ):
                return m
        return None

    family = lambda d: _LABEL_FAMILY.get(d["label"], d["label"])
    inside = {id(d): mirror_holding(d["box"]) for d in dets}
    kept = []
    for d in dets:
        m = inside[id(d)]
        if m is None or d["label"] in _SOFT_FURNISHING_LABELS or d["label"] == "mirror":
            kept.append(d)
            continue
        twin_outside = any(
            o is not d and family(o) == family(d) and inside[id(o)] is not m
            for o in dets
        )
        if not twin_outside:
            kept.append(d)
    return kept


# DINO calls a nightstand "console table", "desk", "side table" or "stool" in
# most bedrooms, which sends it to the wrong product category. A small table
# standing at either edge of a bed, within the bed's height, is a nightstand.
_BEDSIDE_LABELS = frozenset({"console table", "desk", "side table", "stool", "dresser", "cabinet"})
_BEDSIDE_MAX_EDGE_DIST = 0.1      # table centre to the bed box's side, share of image width
_BEDSIDE_MAX_AREA_SHARE = 0.15    # of the bed box
_BEDSIDE_MARGIN = 0.02            # share of image height


def relabel_bedside_tables(dets: list[dict]) -> None:
    area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
    beds = [d["box"] for d in dets if d["label"] == "bed"]
    for d in dets:
        if d["label"] not in _BEDSIDE_LABELS:
            continue
        x1, y1, x2, y2 = d["box"]
        cx = (x1 + x2) / 2
        for bx1, by1, bx2, by2 in beds:
            beside = min(abs(cx - bx1), abs(cx - bx2)) <= _BEDSIDE_MAX_EDGE_DIST
            level = y1 >= by1 - _BEDSIDE_MARGIN and y2 <= by2 + _BEDSIDE_MARGIN
            small = area(d["box"]) <= _BEDSIDE_MAX_AREA_SHARE * area((bx1, by1, bx2, by2))
            if beside and level and small:
                d["label"] = "nightstand"
                break


# DINO never says "TV bench", "coffee table" or "ottoman" for them (24-image
# answer key: 6/6 TV benches were a bookshelf, dresser, sideboard or console
# table; 5/7 coffee tables a desk or console table; the poufs before a sofa
# "stool"). Where a piece stands says what it is:
#   - low storage under a wall-mounted TV is a TV bench. The TV's bottom edge
#     must be above the piece's top: a monitor stands *on* its desk (three
#     desks with monitors in the 56 rooms), a wall TV hangs over its bench;
#   - a table standing in front of a sofa, lower than the sofa, is the coffee
#     table;
#   - a "stool" in front of a sofa or bed is an ottoman (a pouf); a bar stool
#     stands at a counter or desk.
# Only the TV itself is trusted from DINO's decoys, and only when it is sure.
_TV_LABELS = frozenset({"tv", "television"})
_TV_MIN_CONFIDENCE = 0.5
_TV_GAP_MAX = 0.15                # TV bottom to the piece's top, share of image height
_TV_GAP_MIN = -0.02
_TV_BENCH_LABELS = frozenset({"console table", "cabinet", "sideboard", "dresser", "bookshelf"})
_COFFEE_TABLE_LABELS = frozenset({"desk", "console table", "side table", "end table"})
_FRONT_HOSTS = frozenset({"sofa", "couch"})
_OTTOMAN_HOSTS = frozenset({"sofa", "couch", "bed"})
_SOFA_MIN_ASPECT = 1.5            # host width / height, in pixels


def _in_front_of(piece: tuple, host: tuple) -> bool:
    """`piece` stands on the floor in front of `host`: under its width, its
    bottom nearer the camera, its top below the host's upper third, and no
    taller than the host."""
    x1, y1, x2, y2 = piece
    hx1, hy1, hx2, hy2 = host
    cx = (x1 + x2) / 2
    host_h = hy2 - hy1
    return (
        hx1 <= cx <= hx2
        and y2 > hy2
        and y1 > hy1 + host_h / 3
        and (y2 - y1) <= host_h
        and y1 < hy2 + host_h / 2
    )


def relabel_by_position(dets: list[dict], tvs: list[tuple], aspect: float = 1.0) -> None:
    """Renames in place; the old name goes to "dino_label" and CLIP leaves the
    piece alone ("position_label")."""
    def rename(d: dict, label: str) -> None:
        d.setdefault("dino_label", d["label"])
        d["label"] = label
        d["position_label"] = True

    # DINO calls a lone armchair "sofa" (CLIP corrects it later); a real sofa is
    # far wider than tall, and a side table beside an armchair is no coffee table.
    def hosts(labels):
        return [
            b for b in (d["box"] for d in dets if d["label"] in labels)
            if d_is_wide(b)
        ]

    def d_is_wide(b: tuple) -> bool:
        return (b[2] - b[0]) * aspect >= _SOFA_MIN_ASPECT * (b[3] - b[1])

    for d in dets:
        x1, y1, x2, y2 = d["box"]
        if d["label"] in _TV_BENCH_LABELS and (x2 - x1) * aspect > (y2 - y1):
            for tx1, ty1, tx2, ty2 in tvs:
                if x1 <= (tx1 + tx2) / 2 <= x2 and _TV_GAP_MIN <= y1 - ty2 <= _TV_GAP_MAX:
                    rename(d, "tv stand")
                    break
        if d.get("position_label"):
            continue
        if d["label"] in _COFFEE_TABLE_LABELS:
            if any(_in_front_of(d["box"], h) for h in hosts(_FRONT_HOSTS)):
                rename(d, "coffee table")
        elif d["label"] == "stool":
            if any(_in_front_of(d["box"], h) for h in hosts(_OTTOMAN_HOSTS)):
                rename(d, "ottoman")


# A box this far inside a doorway, standing no lower than the doorway's sill
# (+ this share of the image height), is seen through it. A chair in front of
# the door reaches below the sill and stays.
_OPENING_CONTAINMENT = 0.8
_OPENING_SILL_MARGIN = 0.01


# Only what can stand outside is ever dropped as seen through a doorway. In
# photos DINO also boxes a curtained window or a gallery wall as "doorway", and
# everything inside that box went missing (24-image answer key: 22 real pieces,
# all curtains, artwork, lamps, a nightstand and a bookshelf). What the filter
# rightly removed in the 56 generated rooms was garden seating, plants, rugs
# and the cushions on them. Lamps stay out of the list: a desk lamp in front of
# a closed door was dropped, and only one garden "floor lamp" got through.
_OUTDOOR_LABELS = frozenset({
    "sofa", "couch", "armchair", "chair", "dining chair", "office chair", "bench", "ottoman", "stool",
    "coffee table", "dining table", "side table", "end table", "console table",
    "plant pot", "rug", "carpet", "pillow", "cushion", "throw pillow", "blanket",
})


# An opening reaching this close to the top of the photo is a glass wall or the
# room's own door, not a doorway; what stands before it is inside the room.
_OPENING_MIN_TOP = 0.05
# An "opening" this much the same box as a piece of furniture is that piece
# (a four-poster's frame).
_OPENING_MAX_FURNITURE_IOU = 0.5


def real_openings(openings: list[tuple], furniture_boxes: list[tuple]) -> list[tuple]:
    return [
        o for o in openings
        if o[1] > _OPENING_MIN_TOP
        and all(_box_iou_and_containment(o, b)[0] < _OPENING_MAX_FURNITURE_IOU for b in furniture_boxes)
    ]


def seen_through_opening(box: tuple, openings: list[tuple]) -> bool:
    for opening in openings:
        _iou, inside = _box_iou_and_containment(box, opening)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        opening_area = (opening[2] - opening[0]) * (opening[3] - opening[1])
        if box_area >= opening_area:
            continue  # the containment is the opening's, not the box's
        if inside >= _OPENING_CONTAINMENT and box[3] <= opening[3] + _OPENING_SILL_MARGIN:
            return True
    return False


# A piece whose box sits this far inside a bed's or sofa's box, and whose mask
# lies this far inside that bed's or sofa's own SAM 2 mask, is part of it.
_PART_BOX_CONTAINMENT = 0.85
_PART_MASK_CONTAINMENT = 0.7
# What DINO calls a bed's headboard. Folded into the bed only when shaped and
# placed like a headboard (looks_like_headboard): the box of a four-poster or
# canopy bed holds the wall behind it, and the box of any bed can hold the
# photo frames on its nightstands. All 10 of these folded on the 24-image
# answer key and the 56 generated rooms (grounding-dino-base) were real
# mirrors and artwork, 0.06-0.15 of the bed's width; the headboards DINO
# called "mirror" earlier spanned ~0.3 of it from the bed box's top.
_ALWAYS_PART_LABELS = frozenset({"headboard", "mirror", "artwork", "painting"})
_HEADBOARD_MIN_WIDTH_SHARE = 0.25   # of the bed box's width
_HEADBOARD_MAX_TOP_GAP = 0.1        # top below the bed box's top, share of its height


def looks_like_headboard(part: tuple, bed: tuple) -> bool:
    px1, py1, px2, _ = part
    bx1, by1, bx2, by2 = bed
    return (
        (px2 - px1) >= _HEADBOARD_MIN_WIDTH_SHARE * (bx2 - bx1)
        and py1 - by1 <= _HEADBOARD_MAX_TOP_GAP * (by2 - by1)
    )
# DINO reads an armchair's seat cushion as a "stool" or "ottoman".
_SEAT_HOSTS = frozenset({"armchair", "chair", "dining chair", "office chair"})
_SEAT_PART_LABELS = frozenset({"ottoman", "stool"})
_SEAT_PART_CONTAINMENT = 0.9
_SOFT_FURNISHING_LABELS = _MASK_ONLY_LABELS | _PART_ONLY_LABELS


# A headboard is the bed's, inside the bed's box or standing up out of its top
# edge (DINO's bed box often stops at the mattress): most of its width over the
# bed, reaching down into the bed's box but not past its upper part.
_HEAD_MIN_X_OVERLAP = 0.6         # of the headboard's width
_HEAD_MAX_BOTTOM = 0.6            # headboard bottom, share of the bed box height from its top


def is_bed_head(head: tuple, bed: tuple) -> bool:
    _iou, box_in = _box_iou_and_containment(head, bed)
    if box_in >= _PART_BOX_CONTAINMENT:
        return True
    hx1, hy1, hx2, hy2 = head
    bx1, by1, bx2, by2 = bed
    x_overlap = max(0.0, min(hx2, bx2) - max(hx1, bx1)) / max(hx2 - hx1, 1e-6)
    return (
        x_overlap >= _HEAD_MIN_X_OVERLAP
        and hy1 <= by1
        and by1 < hy2 <= by1 + _HEAD_MAX_BOTTOM * (by2 - by1)
    )


def merge_parts_into_hosts(dets: list[dict]) -> list[dict]:
    """Fold detections that are really part of a bed or sofa into it.

    DINO reads a bed's base as a "bench" or its headboard as a "mirror"; as
    the smaller box, that piece would then take those pixels away from the
    bed. It is part of the bed when its box sits inside the bed's box and
    either SAM 2 already painted it as the bed or it's a kind of thing that
    is a bed's "mirror"/"artwork" shaped and placed like a headboard
    (looks_like_headboard). Its mask joins the
    bed's, so the bed card shows its headboard; anything in front of it with a
    box of its own (a lamp on the nightstand) is cut back out in _settle_masks.
    A real bench at the foot of the bed reaches outside the bed's box and
    stays. Needs masks; box-only detections (Gemini) are left as they are.
    """
    import numpy as np

    def as_bool(mask):
        return None if mask is None else np.asarray(mask.convert("L")) > 127

    hosts = [d for d in dets if d["label"] in _PILLOW_HOSTS | _SEAT_HOSTS and d.get("mask") is not None]
    if not hosts:
        return dets
    host_masks = {id(h): as_bool(h["mask"]) for h in hosts}
    # A four-poster comes back as two beds: the frame and the mattress inside
    # it. The smaller one joins the larger.
    area = lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])
    for small in sorted(hosts, key=area):
        for big in hosts:
            if big is small or big["label"] != small["label"] or area(big) <= area(small):
                continue
            if big["label"] not in _PILLOW_HOSTS:
                continue
            if _box_iou_and_containment(small["box"], big["box"])[1] >= _PART_BOX_CONTAINMENT:
                host_masks[id(big)] = host_masks[id(big)] | host_masks[id(small)]
                dets = [d for d in dets if d is not small]
                hosts = [h for h in hosts if h is not small]
                break
    kept = []
    for det in dets:
        host_of = None
        mask = as_bool(det.get("mask"))
        if det not in hosts and mask is not None and mask.any():
            for host in sorted(hosts, key=area):  # the tightest host first
                if det["label"] in _BED_PART_LABELS:
                    if host["label"] == "bed" and is_bed_head(det["box"], host["box"]):
                        host_of = host
                        break
                    continue
                _iou, box_in = _box_iou_and_containment(det["box"], host["box"])
                host_area = (host["box"][2] - host["box"][0]) * (host["box"][3] - host["box"][1])
                det_area = (det["box"][2] - det["box"][0]) * (det["box"][3] - det["box"][1])
                if det_area >= host_area or box_in < _PART_BOX_CONTAINMENT:
                    continue
                if det["label"] in _SOFT_FURNISHING_LABELS:
                    # Pillows, seat cushions and throws belong to what they lie
                    # on: catalogue photos of sofas and beds have them too.
                    host_of = host
                    break
                if host["label"] in _SEAT_HOSTS:
                    # A chair only takes back its own seat, read as a stool or
                    # ottoman wholly inside the chair's box. A real footstool
                    # stands in front and reaches outside it.
                    if det["label"] in _SEAT_PART_LABELS and box_in >= _SEAT_PART_CONTAINMENT:
                        host_of = host
                        break
                    continue
                in_host_mask = (mask & host_masks[id(host)]).sum() / mask.sum()
                headboard = (
                    det["label"] in _ALWAYS_PART_LABELS
                    and host["label"] == "bed"
                    and looks_like_headboard(det["box"], host["box"])
                )
                if headboard or in_host_mask >= _PART_MASK_CONTAINMENT:
                    host_of = host
                    break
        if host_of is None:
            kept.append(det)
        else:
            host_masks[id(host_of)] = host_masks[id(host_of)] | mask
    for host in hosts:
        host["mask"] = Image.fromarray(host_masks[id(host)].astype("uint8") * 255, mode="L")
    return kept


# ── Second opinion on DINO's label from CLIP, across product types only ─────
# Tried on 579 cut-outs from 56 rooms: let loose on the whole vocabulary,
# zero-shot CLIP agreed with DINO on 39% and was often plainly wrong (artwork
# and vases as table lamps). Inside one product type its changes were wrong as
# often as right (desks as console tables) but matter little: matching maps
# both to "table". So CLIP only gets to move a piece to another type, and only
# along the mistakes DINO was seen making, when it is far more sure than of
# DINO's label. 11 corrections and no new errors on those rooms; the swaps
# allowed were chosen on the same rooms, so check new rooms with
# compare_segmentation.py.
_CLIP_TYPES = {
    "sofa": ["sofa", "couch"],
    "seat": ["armchair", "chair", "dining chair", "office chair", "stool", "bench", "ottoman"],
    "table": ["coffee table", "side table", "console table", "dining table", "desk"],
    "storage": ["sideboard", "tv stand", "dresser", "cabinet", "bookshelf", "wardrobe"],
    "light": ["floor lamp", "table lamp", "pendant lamp", "chandelier"],
    "decor": ["vase", "plant pot", "decoration"],
}
_CLIP_TYPE_OF = {label: t for t, labels in _CLIP_TYPES.items() for label in labels}
_CLIP_SWAPS = {
    "ottoman": ["coffee table", "side table"],   # a round coffee table
    "stool": ["coffee table", "side table"],
    "console table": ["sideboard", "tv stand"],  # a low cabinet
    "chandelier": ["vase", "decoration"],        # branches in a vase, a wall clock
    "pendant lamp": ["decoration"],
    "sofa": ["armchair"],                        # a lone armchair
    "couch": ["armchair"],
}
_CLIP_TEMPLATES = ("a photo of a {}.", "a {} on a white background.", "a product photo of a {}.")
_CLIP_MIN_P = 0.6
_CLIP_MIN_RATIO = 5.0


def clip_candidates(label: str) -> list[str] | None:
    """Labels CLIP may pick between for a DINO label: its own type plus the
    cross-type swaps seen in practice. None = CLIP leaves it alone."""
    if label not in _CLIP_SWAPS:
        return None
    return list(dict.fromkeys(_CLIP_TYPES[_CLIP_TYPE_OF[label]] + _CLIP_SWAPS[label]))


def clip_verdict(label: str, candidates: list[str], probs) -> str:
    """The label to keep: CLIP's pick only when it is another product type and
    CLIP is sure of it (>= _CLIP_MIN_P, >= _CLIP_MIN_RATIO x DINO's label)."""
    best = max(range(len(candidates)), key=lambda i: probs[i])
    pick, p = candidates[best], float(probs[best])
    if _CLIP_TYPE_OF.get(pick) == _CLIP_TYPE_OF.get(label):
        return label
    if p < _CLIP_MIN_P or float(probs[candidates.index(label)]) * _CLIP_MIN_RATIO > p:
        return label
    return pick


def _cutout_on_white(arr: Any, mask: Image.Image | None) -> Image.Image | None:
    """The piece's mask pixels on a white square, trimmed to the mask; None
    without a usable mask (under 50 pixels)."""
    import numpy as np

    if mask is None:
        return None
    m = np.asarray(mask.convert("L")) > 127
    ys, xs = np.nonzero(m)
    if len(xs) < 50:
        return None
    crop = np.where(m[..., None], arr, 255)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    side = int(max(crop.shape[:2]) * 1.1)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(Image.fromarray(crop.astype("uint8")), ((side - crop.shape[1]) // 2, (side - crop.shape[0]) // 2))
    return canvas


def _padded_crop(img: Image.Image, box: tuple, pad: float = 0.6) -> Image.Image:
    """The piece's box (0-1) plus `pad` of its size on every side, clamped."""
    W, H = img.size
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    return img.crop((
        int(max(0.0, x1 - bw * pad) * W), int(max(0.0, y1 - bh * pad) * H),
        int(min(1.0, x2 + bw * pad) * W), int(min(1.0, y2 + bh * pad) * H),
    ))


def labeler() -> str:
    """SEGMENTATION_LABELER: "gemini" (default; CLIP for what Gemini can't
    answer), "clip" (the old offline rule only) or "none" (DINO's label)."""
    return os.getenv("SEGMENTATION_LABELER", "gemini").strip().lower()


async def relabel_with_gemini(img: Image.Image, detections: list[dict]) -> set[int]:
    """Gemini names each piece from its cut-out plus a padded crop of the room
    (label_verification). Its answer replaces DINO's label; DINO's is kept in
    "dino_label". Returns the indices Gemini answered, so CLIP can try the
    rest. Never raises."""
    import numpy as np

    from label_verification import VerifyInput, label_for_type, verify_labels

    arr = np.array(img.convert("RGB"))
    W, H = img.size
    inputs = []
    for i, d in enumerate(detections):
        x1, y1, x2, y2 = d["box"]
        cut = _cutout_on_white(arr, d.get("mask"))
        if cut is None:
            cut = img.crop((int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)))
        inputs.append(VerifyInput(str(i), d["label"], cut, _padded_crop(img, d["box"])))
    verified = await verify_labels(inputs)
    for key, furniture_type in verified.items():
        d = detections[int(key)]
        new = label_for_type(d["label"], furniture_type)
        d["label_source"] = "gemini"
        if new != d["label"]:
            d.setdefault("dino_label", d["label"])
            d["label"] = new
    return {int(k) for k in verified}


async def name_pieces(img: Image.Image, detections: list[dict]) -> None:
    """The naming step after SAM 2, per SEGMENTATION_LABELER: Gemini names the
    pieces (DINO mixes up look-alike types); CLIP's offline rule only for what
    Gemini didn't answer (no key, failure). Never raises."""
    if labeler() == "none":
        return
    answered: set[int] = set()
    if labeler() == "gemini":
        try:
            answered = await relabel_with_gemini(img, detections)
        except Exception as exc:  # a second opinion must never fail the request
            print(f"[segmentation] Gemini relabel skipped: {exc}")
    rest = [d for i, d in enumerate(detections) if i not in answered]
    if rest:
        try:
            await asyncio.to_thread(relabel_with_clip, img, rest)
        except Exception as exc:
            print(f"[segmentation] CLIP relabel skipped: {exc}")


def relabel_with_clip(img: Image.Image, detections: list[dict]) -> None:
    """Run clip_verdict on each piece's cut-out, on white like product matching
    sees it. Keeps DINO's label in "dino_label" when it changes. Since
    2026-11-21 only the fallback for pieces Gemini didn't answer."""
    import numpy as np

    model = _get_clip()
    if model is None:
        return
    arr = np.array(img.convert("RGB"))
    for d in detections:
        cands = clip_candidates(d["label"])
        if not cands or d.get("mask") is None or d.get("position_label"):
            continue
        canvas = _cutout_on_white(arr, d["mask"])
        if canvas is None:
            continue
        with _inference_lock:
            v = model.encode(canvas, normalize_embeddings=True)
            text = model.encode(
                [t.format(c) for c in cands for t in _CLIP_TEMPLATES], normalize_embeddings=True
            ).reshape(len(cands), len(_CLIP_TEMPLATES), -1).mean(1)
        text /= np.linalg.norm(text, axis=1, keepdims=True)
        logits = 100 * text @ v
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        new = clip_verdict(d["label"], cands, probs)
        if new != d["label"]:
            d["dino_label"] = d["label"]
            d["label"] = new
            d["label_source"] = "clip"


# A piece left with less than this share of its own mask after the overlap
# split was a second box on something else (a lamp's shade reported as a
# "pendant lamp"); as a card it would fall back to a bare rectangle.
_MIN_KEPT_MASK_SHARE = 0.3
# The same after split_contested, which takes more: the office chair behind a
# desk keeps only its back (24%) and is still a card. A box on part of another
# piece (flowers in a vase as "chandelier", a drawing inside its frame) keeps
# next to nothing.
_MIN_KEPT_CONTESTED_SHARE = 0.1


def _settle_masks(detections: list[dict], W: int, H: int, ask=None) -> None:
    """Split shared pixels between overlapping masks so that no pixel is in
    two (`ask` as in split_contested), then grow each box to cover its whole
    mask — the padded SAM 2 prompt may reach past DINO's box, and the
    crop/cut-out are cut from the box. Marks `swallowed` on a piece that lost
    most of its mask."""
    import numpy as np

    with_masks = [d for d in detections if d["mask"] is not None]
    if not with_masks:
        return
    for d in with_masks:
        d["mask"] = np.asarray(d["mask"].convert("L")) > 127
    before = [int(d["mask"].sum()) for d in with_masks]
    resolve_overlaps(with_masks)
    behind = [int(d["mask"].sum()) for d in with_masks]
    split_contested(with_masks, ask)
    for d, area, area_behind in zip(with_masks, before, behind):
        d["swallowed"] = area > 0 and (
            area_behind < area * _MIN_KEPT_MASK_SHARE
            or d["mask"].sum() < area * _MIN_KEPT_CONTESTED_SHARE
        )
    for d in with_masks:
        extent = mask_bbox(d["mask"], W, H)
        if extent is not None:
            d["box"] = (
                min(d["box"][0], extent[0]), min(d["box"][1], extent[1]),
                max(d["box"][2], extent[2]), max(d["box"][3], extent[3]),
            )
        d["mask"] = Image.fromarray(d["mask"].astype("uint8") * 255, mode="L")


def _settle_masks_with_sam2(img: Image.Image, detections: list[dict]) -> None:
    """_settle_masks, with SAM 2 point prompts for the patches two similar-sized
    pieces share. The image is encoded again only if a patch comes up (none in
    most rooms); the lock keeps another request from swapping it mid-way."""
    import numpy as np
    import torch

    predictor, device = _get_sam2_predictor()
    encoded = False

    def ask(x: int, y: int):
        nonlocal encoded
        if not encoded:
            predictor.set_image(np.array(img.convert("RGB")))
            encoded = True
        masks, _scores, _logits = predictor.predict(
            point_coords=np.array([[x, y]], dtype=float), point_labels=np.array([1]),
            multimask_output=True,
        )
        return masks > 0

    with _inference_lock, torch.inference_mode(), _autocast(device):
        _settle_masks(detections, *img.size, ask=ask)


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
    fallback_reason: str | None = None
    mask_imgs: list[Image.Image | None] = []

    try:
        raw = await asyncio.to_thread(_detect_with_grounding_dino_local, img)
        boxes_px = [det["box"] for det in raw]
        labels = [det["label"] for det in raw]
        mask_imgs = (
            await asyncio.to_thread(_segment_boxes_with_sam2_local, img, boxes_px, labels)
            if raw else []
        )
    except Exception as exc:
        fallback_reason = f"{type(exc).__name__}: {exc}"
        print(f"[segmentation] local Grounding DINO / SAM 2 pipeline failed: {fallback_reason}")
        mask_imgs = []
        try:
            raw = await _detect_with_gemini(image_bytes, mime_type)
            method = "gemini_vision"
        except Exception as fallback_exc:
            # A segmentation failure must never take down /generate-room —
            # the generated room image is still a perfectly good result on
            # its own. Worst case here is an empty furniture list, not a
            # 502 for the whole request.
            print(f"[segmentation] Gemini Vision fallback also failed: {fallback_exc}")
            fallback_reason = f"{fallback_reason} | Gemini fallback: {fallback_exc}"
            raw = []
            method = "none"

    if len(mask_imgs) != len(raw):
        mask_imgs = [None] * len(raw)

    detections, mask_only = select_detections(raw, mask_imgs, W, H)
    if method == "grounding_dino_sam2":
        await asyncio.to_thread(_settle_masks_with_sam2, img, detections + mask_only)
    else:
        _settle_masks(detections + mask_only, W, H)  # Gemini boxes have no masks
    # Largest first, so the big pieces (sofa, bed) lead a list of cards
    # instead of a stray vase.
    detections = [d for d in detections if not d.get("swallowed")]
    if method == "grounding_dino_sam2":
        await name_pieces(img, detections)
    detections.sort(
        key=lambda d: (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]), reverse=True
    )

    items = [
        _build_item(img, d["label"], d["confidence"], idx, *d["box"], d["mask"])
        for idx, d in enumerate(detections)
    ]
    for item, d in zip(items, detections):
        item.dino_label = d.get("dino_label")
        item.label_source = d.get("label_source", "dino")

    counts = dict(Counter(item.label for item in items))
    result = SegmentationResult(
        items=items,
        counts=counts,
        total=len(items),
        method=method,
        fallback_reason=fallback_reason,
    )
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
    - `crop_image` — base64 JPEG thumbnail of the item's bounding box
    - `cutout_image` — base64 PNG with a transparent background, trimmed to the
      item (only when a real SAM 2 mask exists, else null → use `crop_image`)
    - `mask_image` — base64 PNG binary mask (full image size) for that one item
    - `mask_precise` — True if the mask came from SAM 2, False if it's a bbox approximation
    - `features` — dominant colours (of the item's own pixels when masked),
      area %, aspect ratio

    Items are de-duplicated, stripped of doors/windows, and sorted largest
    first. `method` says which detector produced them and `fallback_reason`
    says why the local pipeline wasn't used (null when it was).

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
