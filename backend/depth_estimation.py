"""
depth_estimation.py
───────────────────
Monocular metric depth estimation for uploaded room photos, used when the
user scanned the room with LiDAR but chose to upload their own photo instead
of capturing one inside the AR session.

Model: Depth-Anything-V2-Metric-Indoor-Small-hf
  • ~24 M parameters, CPU-friendly (~1–2 s per 640-px image)
  • Returns absolute depth in metres, calibrated for indoor scenes
  • Auto-downloaded from HuggingFace on first call and cached

Output format matches the iOS LiDAR depth encoding used by the rest of the
pipeline (closer = brighter, 0 = no valid depth), so spatial_measurement.py's
decode_depth_png() can consume it unchanged.
"""

from __future__ import annotations

import base64
import threading
from io import BytesIO
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from transformers import Pipeline

_MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
_pipe: "Pipeline | None" = None
_lock = threading.Lock()


def _get_pipeline() -> "Pipeline":
    global _pipe
    if _pipe is None:
        with _lock:
            if _pipe is None:
                from transformers import pipeline as hf_pipeline
                _pipe = hf_pipeline(
                    task="depth-estimation",
                    model=_MODEL_ID,
                )
    return _pipe


def estimate_depth(
    image_bytes: bytes,
    target_size: int = 640,
) -> tuple[str, float, float]:
    """Run monocular depth estimation on a room photo.

    Returns (depth_png_b64, depth_min_m, depth_max_m) using the same encoding
    as the iOS LiDAR pipeline: pixel = (1 - (d - min) / range) * 255, so
    closer pixels are brighter and 0 marks invalid/missing depth.

    ``target_size`` caps the longer side before inference for speed; the
    output depth map is resized back to the original image dimensions.
    """
    original = Image.open(BytesIO(image_bytes)).convert("RGB")
    orig_w, orig_h = original.size

    # Downscale for inference — Depth Anything V2 works at any resolution but
    # smaller is faster and accuracy is nearly identical at 640 px.
    scale = min(target_size / max(orig_w, orig_h), 1.0)
    infer_w = max(int(orig_w * scale) // 14 * 14, 14)  # multiple of patch size
    infer_h = max(int(orig_h * scale) // 14 * 14, 14)
    infer_img = original.resize((infer_w, infer_h), Image.LANCZOS)

    pipe = _get_pipeline()
    result = pipe(infer_img)
    depth_arr: np.ndarray = np.array(result["depth"], dtype=np.float32)

    # Resize back to original dimensions so masks from the full-res segmentation
    # align correctly when spatial_measurement.py scales pixel coordinates.
    depth_pil = Image.fromarray(depth_arr).resize((orig_w, orig_h), Image.BILINEAR)
    depth_arr = np.array(depth_pil, dtype=np.float32)

    # Replace any non-positive values (sky/background with no valid depth) with
    # a large sentinel so they don't corrupt the min, then mask them to 0.
    valid = depth_arr > 0.05
    depth_min_m = float(depth_arr[valid].min()) if valid.any() else 0.1
    depth_max_m = float(depth_arr[valid].max()) if valid.any() else 5.0

    # Encode to the same greyscale PNG format as iOS LiDAR:
    # pixel = (1 - (depth - min) / range) * 255, invalid → 0
    depth_range = max(depth_max_m - depth_min_m, 1e-4)
    encoded = np.where(
        valid,
        ((1.0 - (depth_arr - depth_min_m) / depth_range) * 255).clip(0, 255),
        0,
    ).astype(np.uint8)

    buf = BytesIO()
    Image.fromarray(encoded, mode="L").save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()

    return b64, depth_min_m, depth_max_m
