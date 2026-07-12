"""
Quick test script for /segment-furniture.
Usage:
    python test_segmentation.py path/to/room.jpg [path/to/room2.jpg ...]

Prints a summary of detected furniture and saves each item's crop to ./crops/
"""

import argparse
import base64
import json
import os
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
CROPS_DIR = Path("crops")


def test_image(image_path: str, bbox_padding: float | None = None, remove_bg: bool = False) -> None:
    path = Path(image_path)
    print(f"\n{'='*60}")
    print(f"Image: {path.name}")
    print("="*60)

    with open(path, "rb") as f:
        image_bytes = f.read()

    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    data: dict = {}
    if bbox_padding is not None:
        data["bbox_padding"] = str(bbox_padding)
    if remove_bg:
        data["remove_bg"] = "true"

    response = httpx.post(
        f"{BASE_URL}/segment-furniture",
        files={"image": (path.name, image_bytes, mime)},
        data=data,
        timeout=120,
    )

    if response.status_code != 200:
        print(f"ERROR {response.status_code}: {response.text}")
        return

    result = response.json()

    print(f"Method used : {result['method']}")
    print(f"Total items : {result['total']}")
    print(f"Counts      : {json.dumps(result['counts'], indent=2)}")
    print()

    # Save crops and print per-item summary
    CROPS_DIR.mkdir(exist_ok=True)
    for item in result["items"]:
        label = item["label"].replace(" ", "_")
        conf  = item["confidence"]
        bbox  = item["bbox"]
        colors = ", ".join(item["features"]["dominant_colors"])
        area  = item["features"]["area_pct"]

        print(
            f"  [{item['id']}] {item['label']:<20} conf={conf:.2f}  "
            f"area={area:.1f}%  colors={colors}"
        )
        print(
            f"       bbox x({bbox['x_min']:.2f}–{bbox['x_max']:.2f}) "
            f"y({bbox['y_min']:.2f}–{bbox['y_max']:.2f})"
        )

        # Save crop image
        _, encoded = item["crop_image"].split(",", 1)
        crop_bytes = base64.b64decode(encoded)
        crop_path = CROPS_DIR / f"{path.stem}_{item['id']}_{label}.png"
        crop_path.write_bytes(crop_bytes)

    print(f"\nCrops saved to: {CROPS_DIR.resolve()}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="Image file(s) to segment")
    parser.add_argument(
        "--padding", type=float, default=None,
        help="Bbox padding fraction per side (e.g. 0.05). Overrides BBOX_PADDING in .env."
    )
    parser.add_argument(
        "--remove-bg", action="store_true", default=False,
        help="Remove background from each crop using rembg u2net model."
    )
    args = parser.parse_args()

    for img in args.images:
        if not os.path.exists(img):
            print(f"File not found: {img}")
            continue
        test_image(img, bbox_padding=args.padding, remove_bg=args.remove_bg)
