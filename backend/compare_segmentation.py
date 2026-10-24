"""
Before/after check for the SAM 2 stage, run locally (no server needed).
Usage:
    python compare_segmentation.py path/to/room.jpg [path/to/room2.jpg ...]

Runs Grounding DINO once per image, keeps the detections the pipeline keeps
(select_detections: filtering, size checks, dedupe), then SAM 2 twice on the
same boxes:
  • before — the original call: DINO's raw box, one default mask, no clean-up
  • after  — the current pipeline: clean-up, overlaps split, pillows cut out
and saves two images per room to ./compare/:
  • <name>_compare.png — boxes | before | after, one colour per item
  • <name>_items.png   — one row per item: before cut-out | after cut-out, on a
    checkerboard, i.e. what product matching gets for that piece
Needs the same setup as the local pipeline (see segmentation.py).
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import segmentation as seg

OUT_DIR = Path("compare")
COLOURS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128),
    (170, 110, 40), (128, 0, 0), (170, 255, 195), (128, 128, 0), (0, 0, 128),
    (255, 215, 180), (250, 190, 212), (220, 190, 255), (255, 250, 200), (128, 128, 128),
]
MASK_ONLY_COLOUR = (150, 150, 150)
CELL = 170
LABEL_W = 170


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def _overlay(img: Image.Image, masks: list, title: str) -> Image.Image:
    out = np.asarray(img.convert("RGB")).astype(float)
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        m = np.asarray(mask.convert("L")) > 127
        out[m] = out[m] * 0.45 + np.array(COLOURS[i % len(COLOURS)]) * 0.55
    panel = Image.fromarray(out.astype("uint8"))
    ImageDraw.Draw(panel).text((10, 10), title, fill=(255, 255, 255))
    return panel


def _before(img: Image.Image, boxes: list) -> list:
    """The SAM 2 call as it was: raw DINO box, multimask_output=False."""
    predictor, _device = seg._get_sam2_predictor()
    predictor.set_image(np.array(img.convert("RGB")))
    masks = []
    for box in boxes:
        m, _s, _l = predictor.predict(box=np.array(box), multimask_output=False)
        masks.append(Image.fromarray((m[0] > 0).astype("uint8") * 255, mode="L"))
    return masks


def _checker(w: int, h: int, s: int = 8) -> Image.Image:
    yy, xx = np.mgrid[0:h, 0:w]
    light = (((yy // s) + (xx // s)) % 2 == 0)[..., None]
    return Image.fromarray(np.where(light, 240, 205).astype("uint8").repeat(3, axis=2))


def _cutout_cell(img_np, mask, crop) -> Image.Image:
    x1, y1, x2, y2 = crop
    piece = np.dstack([img_np[y1:y2, x1:x2], mask[y1:y2, x1:x2].astype("uint8") * 255])
    h, w = piece.shape[:2]
    scale = min(CELL / w, CELL / h)
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    rgba = Image.fromarray(piece, mode="RGBA").resize(size, Image.LANCZOS)
    cell = _checker(CELL, CELL)
    cell.paste(rgba, ((CELL - size[0]) // 2, (CELL - size[1]) // 2), rgba)
    return cell


def _items_sheet(img: Image.Image, labels: list, columns: dict) -> Image.Image:
    """One row per item, one cut-out per column, all cropped to the same area."""
    img_np = np.array(img)
    W, H = img.size
    font, small = _font(13), _font(12)
    width = LABEL_W + (CELL + 8) * len(columns)
    rows = []
    for i, label in enumerate(labels):
        masks = [
            np.asarray(col[i].convert("L")) > 127 if col[i] is not None else np.zeros((H, W), bool)
            for col in columns.values()
        ]
        ys, xs = np.nonzero(np.logical_or.reduce(masks))
        if len(xs) == 0:
            continue
        crop = (max(0, xs.min() - 4), max(0, ys.min() - 4), min(W, xs.max() + 5), min(H, ys.max() + 5))
        row = Image.new("RGB", (width, CELL + 24), "white")
        draw = ImageDraw.Draw(row)
        draw.text((8, 8), f"{i + 1}. {label}", fill=(20, 30, 40), font=font)
        for j, (name, mask) in enumerate(zip(columns, masks)):
            x = LABEL_W + j * (CELL + 8)
            row.paste(_cutout_cell(img_np, mask, crop), (x, 0))
            draw.text((x + 4, CELL + 4), f"{name}: {int(mask.sum()):,} px", fill=(70, 80, 90), font=small)
        rows.append(row)

    head = Image.new("RGB", (width, 28), "white")
    draw = ImageDraw.Draw(head)
    for j, name in enumerate(columns):
        draw.text((LABEL_W + j * (CELL + 8) + 4, 8), name.upper(), fill=(29, 111, 122), font=font)
    sheet = Image.new("RGB", (width, head.height + sum(r.height for r in rows)), "white")
    sheet.paste(head, (0, 0))
    y = head.height
    for row in rows:
        sheet.paste(row, (0, y))
        y += row.height
    return sheet


def compare(path: Path) -> None:
    img = Image.open(path).convert("RGB")
    W, H = img.size
    # The same detections the pipeline keeps: filtered, size-checked, deduped.
    found = seg._detect_with_grounding_dino_local(img)
    dets, mask_only = seg.select_detections(found, [None] * len(found), W, H)
    to_px = lambda b: [b[0] * W, b[1] * H, b[2] * W, b[3] * H]
    boxes = [to_px(d["box"]) for d in dets]
    print(f"{path.name}: {len(dets)} items — {', '.join(d['label'] for d in dets)}")
    if mask_only:
        print(f"  cut out of other masks: {', '.join(d['label'] for d in mask_only)}")

    before = _before(img, boxes)
    all_dets = dets + mask_only
    after_raw = seg._segment_boxes_with_sam2_local(
        img, [to_px(d["box"]) for d in all_dets], [d["label"] for d in all_dets]
    )
    for d, m in zip(all_dets, after_raw):
        d["mask"] = m
    seg._settle_masks(all_dets, W, H)
    after = [d["mask"] for d in dets]
    labels = [d["label"] + (" (dropped)" if d.get("swallowed") else "") for d in dets]
    dropped = [d["label"] for d in dets if d.get("swallowed")]
    if dropped:
        print(f"  dropped, mask taken by other items: {', '.join(dropped)}")

    boxed = img.copy()
    draw = ImageDraw.Draw(boxed)
    for i, (label, b) in enumerate(zip(labels, boxes)):
        draw.rectangle(b, outline=COLOURS[i % len(COLOURS)], width=3)
        draw.text((b[0] + 4, b[1] + 4), label, fill=COLOURS[i % len(COLOURS)])
    for d in mask_only:
        draw.rectangle(to_px(d["box"]), outline=MASK_ONLY_COLOUR, width=1)

    panels = [boxed, _overlay(img, before, "before"), _overlay(img, after, "after")]
    sheet = Image.new("RGB", (W * 3, H), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, (W * i, 0))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{path.stem}_compare.png"
    sheet.save(out)
    print(f"  saved {out}")

    items = OUT_DIR / f"{path.stem}_items.png"
    _items_sheet(img, labels, {"before": before, "after": after}).save(items)
    print(f"  saved {items}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_segmentation.py image1.jpg [image2.jpg ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            compare(Path(arg))
        else:
            print(f"File not found: {arg}")
