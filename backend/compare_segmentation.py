"""
Before/after check for the SAM 2 stage, run locally (no server needed).
Usage:
    python compare_segmentation.py path/to/room.jpg [path/to/room2.jpg ...]

Runs Grounding DINO once per image, then SAM 2 twice on the same boxes:
  • before — the old call: DINO's raw box, one default mask, no clean-up
  • after  — padded box, best of 3 candidate masks, clean-up, overlaps split
and saves <name>_compare.png to ./compare/: boxes | before | after, one colour
per item. Needs the same setup as the local pipeline (see segmentation.py).
"""

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import segmentation as seg

OUT_DIR = Path("compare")
COLOURS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48),
    (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (0, 128, 128),
    (170, 110, 40), (128, 0, 0), (170, 255, 195), (128, 128, 0), (0, 0, 128),
    (255, 215, 180), (250, 190, 212), (220, 190, 255), (255, 250, 200), (128, 128, 128),
]


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


def compare(path: Path) -> None:
    img = Image.open(path).convert("RGB")
    W, H = img.size
    dets = [d for d in seg._detect_with_grounding_dino_local(img) if seg.is_furniture_label(d["label"])]
    boxes = [d["box"] for d in dets]
    print(f"{path.name}: {len(dets)} items — {', '.join(d['label'] for d in dets)}")

    before = _before(img, boxes)
    labels = [d["label"] for d in dets]
    after_raw = seg._segment_boxes_with_sam2_local(img, boxes, labels)
    settled = [
        {"box": (b[0] / W, b[1] / H, b[2] / W, b[3] / H), "mask": m, "label": l}
        for b, m, l in zip(boxes, after_raw, labels)
    ]
    seg._settle_masks(settled, W, H)
    after = [d["mask"] for d in settled]

    boxed = img.copy()
    draw = ImageDraw.Draw(boxed)
    for i, (d, b) in enumerate(zip(dets, boxes)):
        draw.rectangle(b, outline=COLOURS[i % len(COLOURS)], width=3)
        draw.text((b[0] + 4, b[1] + 4), d["label"], fill=COLOURS[i % len(COLOURS)])

    panels = [boxed, _overlay(img, before, "before"), _overlay(img, after, "after")]
    sheet = Image.new("RGB", (W * 3, H), "white")
    for i, panel in enumerate(panels):
        sheet.paste(panel, (W * i, 0))
    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / f"{path.stem}_compare.png"
    sheet.save(out)
    print(f"  saved {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compare_segmentation.py image1.jpg [image2.jpg ...]")
        sys.exit(1)
    for arg in sys.argv[1:]:
        if os.path.exists(arg):
            compare(Path(arg))
        else:
            print(f"File not found: {arg}")
