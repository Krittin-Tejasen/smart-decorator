"""
Quick test script for /segment-furniture.
Usage:
    python test_segmentation.py path/to/room.jpg [path/to/room2.jpg ...]

Prints a summary of detected furniture and saves each item's crop to ./crops/
"""

import base64
import json
import os
import sys
from pathlib import Path

import httpx

BASE_URL = "http://localhost:8000"
CROPS_DIR = Path("crops")


def test_image(image_path: str) -> None:
    path = Path(image_path)
    print(f"\n{'='*60}")
    print(f"Image: {path.name}")
    print("="*60)

    with open(path, "rb") as f:
        image_bytes = f.read()

    mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"

    response = httpx.post(
        f"{BASE_URL}/segment-furniture",
        files={"image": (path.name, image_bytes, mime)},
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
        mask_kind = "sam2" if item["mask_precise"] else "bbox-approx"

        print(
            f"  [{item['id']}] {item['label']:<20} conf={conf:.2f}  "
            f"area={area:.1f}%  colors={colors}  mask={mask_kind}"
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

        # Save mask image (one mask per furniture item)
        _, mask_encoded = item["mask_image"].split(",", 1)
        mask_bytes = base64.b64decode(mask_encoded)
        mask_path = CROPS_DIR / f"{path.stem}_{item['id']}_{label}_mask.png"
        mask_path.write_bytes(mask_bytes)

    print(f"\nCrops saved to: {CROPS_DIR.resolve()}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_segmentation.py image1.jpg [image2.jpg ...]")
        sys.exit(1)

    for img in sys.argv[1:]:
        if not os.path.exists(img):
            print(f"File not found: {img}")
            continue
        test_image(img)
