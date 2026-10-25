"""
spatial_measurement.py
───────────────────────
Estimates real-world furniture dimensions (width, depth, height) for items
segmented in a generated room image, using the room's LiDAR depth map and
camera intrinsics/extrinsics captured during the original scan.

This relies on generation.py's prompt explicitly preserving camera framing
and room layout (see _STRUCTURE_PRESERVATION_CLAUSE there) — the original
scan's floor depth is only valid for the generated image's pixels because
the camera angle/perspective is guaranteed not to change. If that prompt
constraint is ever relaxed to allow re-framing the scene, these numbers
would be wrong without any error being raised. Treat it as load-bearing.

Approach, mirroring the iOS app's own depth-sampling math
(ARCameraView.swift's depthSamplePoint):
  1. Take the segmentation mask's bottom contour (assumes the item rests on
     the floor — wrong for wall-mounted items or furniture stacked on other
     furniture, e.g. a lamp on a table).
  2. Look up real depth at those pixels from the original scan's depth map
     (the floor itself didn't move, even though what's drawn on it is new).
  3. Unproject into 3-D world points via the pinhole camera model, then
     rotate into room-aligned axes using the floor-plan corner points.
  4. width/depth = footprint extents; height via similar triangles, using
     the mask's vertical pixel extent at a reliable reference column and
     that column's known camera distance.

Returns None (never a guessed number) whenever too few valid depth samples
survive — e.g. the original floor was occluded at that spot, or the scan
simply has no depth data there.

Known limitation to verify on a real device: this assumes the mask's pixel
grid, the depth map's pixel grid, and the intrinsics' image space all share
one consistent orientation (landscape, matching ARFrame.capturedImage and
ARDepthData.depthMap's native sensor orientation). If measured dimensions
come out swapped or nonsensical, check orientation handling in the capture
pipeline (ARCameraView.swift's handleCapturePhoto saves the JPEG without an
explicit UIImage orientation) before suspecting the math below.
"""

from __future__ import annotations

import base64
import math
from io import BytesIO
from typing import Any

import numpy as np
from PIL import Image

# Minimum number of valid depth samples along the mask's bottom contour
# before a footprint is trusted enough to report a size.
_MIN_VALID_SAMPLES = 5

# Depth values at or below this (metres) are treated as "no real sample" —
# matches how the iOS encoder leaves unmeasured pixels at 0.
_MIN_VALID_DEPTH_M = 0.05


def decode_depth_png(png_b64: str, min_m: float, max_m: float) -> np.ndarray:
    """Inverse of iOS's RoomCaptureState.extractDepthPNG, which encodes
    pixel = (1 - (depth - min)/range) * 255 (closer = brighter). Returns a
    float32 array in metres; 0.0 marks pixels with no valid depth at capture.
    """
    raw = base64.b64decode(png_b64)
    img = Image.open(BytesIO(raw)).convert("L")
    pixels = np.asarray(img, dtype=np.float32)
    depth_range = max(max_m - min_m, 1e-4)
    depth_m = max_m - (pixels / 255.0) * depth_range
    # iOS leaves "no valid depth" pixels at 0, which decodes to max_m above —
    # mask those back out to 0 so they're correctly treated as missing.
    return np.where(pixels > 0, depth_m, 0.0)


def decode_mask_png(png_b64: str) -> np.ndarray:
    """Full-size binary mask (L mode, from segmentation.py) -> bool array."""
    raw = base64.b64decode(png_b64)
    img = Image.open(BytesIO(raw)).convert("L")
    return np.asarray(img) > 127


def unproject_pixel(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: list[float],
    extrinsics: list[float],
) -> np.ndarray:
    """Pinhole unprojection — mirrors ARCameraView.swift's depthSamplePoint.

    `intrinsics`: row-major [fx,0,cx, 0,fy,cy, 0,0,1] (9 values).
    `extrinsics`: column-major 4x4 camera-to-world transform (16 values).
    """
    fx, cx = intrinsics[0], intrinsics[2]
    fy, cy = intrinsics[4], intrinsics[5]

    x_cam = (u - cx) / fx * depth_m
    y_cam = (v - cy) / fy * depth_m
    cam_point = np.array([x_cam, y_cam, -depth_m, 1.0])  # ARKit: camera looks down -Z

    m = np.array(extrinsics, dtype=np.float64).reshape(4, 4, order="F")
    world_point = m @ cam_point
    return world_point[:3]


def camera_position(extrinsics: list[float]) -> np.ndarray:
    """Camera world position = column 3 of the extrinsics matrix (indices 12-14)."""
    return np.array(extrinsics[12:15], dtype=np.float64)


def room_rotation_angle(floor_plan_points: list[dict[str, Any]]) -> float:
    """Angle (radians) to rotate world X/Z so the room's walls align to the
    axes, taken from the floor polygon's longest edge.

    Points are {"x":, "y":} where "y" holds world Z (matches the Flutter app's
    lidar_view.dart, which stores floor-plan points as Offset(x, z)).
    """
    if len(floor_plan_points) < 2:
        return 0.0
    pts = [(float(p["x"]), float(p["y"])) for p in floor_plan_points]
    best_len = -1.0
    best_angle = 0.0
    for i in range(len(pts)):
        x1, z1 = pts[i]
        x2, z2 = pts[(i + 1) % len(pts)]
        dx, dz = x2 - x1, z2 - z1
        length = math.hypot(dx, dz)
        if length > best_len:
            best_len = length
            best_angle = math.atan2(dz, dx)
    return -best_angle


def _rotate_xz(x: float, z: float, angle: float) -> tuple[float, float]:
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    return x * cos_a - z * sin_a, x * sin_a + z * cos_a


def measure_furniture(
    mask_png_b64: str,
    depth_map_png_b64: str,
    depth_min_m: float,
    depth_max_m: float,
    intrinsics: list[float],
    extrinsics: list[float],
    floor_plan_points: list[dict[str, Any]],
) -> dict[str, float] | None:
    """Returns {"width_m", "depth_m", "height_m"}, or None if there wasn't
    enough valid depth under this item's footprint to trust a result."""
    mask = decode_mask_png(mask_png_b64)
    depth = decode_depth_png(depth_map_png_b64, depth_min_m, depth_max_m)

    mh, mw = mask.shape
    dh, dw = depth.shape
    if mw == 0 or mh == 0 or dw == 0 or dh == 0:
        return None

    cols_with_mask = np.where(np.any(mask, axis=0))[0]
    if cols_with_mask.size == 0:
        return None

    angle = room_rotation_angle(floor_plan_points)

    footprint_xz: list[tuple[float, float]] = []
    sample_world: list[np.ndarray] = []
    sample_mask_col: list[int] = []

    # Sample a bounded number of columns across the footprint — the depth
    # map's resolution is much lower than the mask's, so every column isn't
    # needed or meaningful.
    step = max(1, cols_with_mask.size // 64)
    for col in cols_with_mask[::step]:
        rows_in_col = np.where(mask[:, col])[0]
        if rows_in_col.size == 0:
            continue
        bottom_row = int(rows_in_col.max())

        # Scale this mask-space pixel into the depth map's own resolution.
        du = min(max(int(col / mw * dw), 0), dw - 1)
        dv = min(max(int(bottom_row / mh * dh), 0), dh - 1)

        depth_m = float(depth[dv, du])
        if depth_m <= _MIN_VALID_DEPTH_M:
            continue  # occluded or missing in the original scan

        world = unproject_pixel(float(du), float(dv), depth_m, intrinsics, extrinsics)
        rx, rz = _rotate_xz(float(world[0]), float(world[2]), angle)
        footprint_xz.append((rx, rz))
        sample_world.append(world)
        sample_mask_col.append(int(col))

    if len(footprint_xz) < _MIN_VALID_SAMPLES:
        return None

    xs = [p[0] for p in footprint_xz]
    zs = [p[1] for p in footprint_xz]
    width_m = abs(max(xs) - min(xs))
    depth_m_out = abs(max(zs) - min(zs))

    # Height via similar triangles, using the sample closest to the camera
    # (shortest distance = most reliable depth reading) as the reference.
    cam_pos = camera_position(extrinsics)
    distances = [float(np.linalg.norm(w - cam_pos)) for w in sample_world]
    ref_idx = int(np.argmin(distances))
    ref_col = sample_mask_col[ref_idx]
    ref_distance = distances[ref_idx]

    rows_in_ref_col = np.where(mask[:, ref_col])[0]
    pixel_height = int(rows_in_ref_col.max() - rows_in_ref_col.min()) + 1
    fy = intrinsics[4]
    height_m = abs((pixel_height / fy) * ref_distance)

    return {
        "width_m": round(width_m, 3),
        "depth_m": round(depth_m_out, 3),
        "height_m": round(height_m, 3),
    }


# ── Monocular-depth path ──────────────────────────────────────────────────────

# Assumed horizontal field-of-view for an uploaded photo when no EXIF focal
# length is available — 68° is typical for a modern smartphone wide camera.
_DEFAULT_HFOV_DEG = 68.0


def _estimate_intrinsics(image_w: int, image_h: int) -> list[float]:
    """Return a synthetic 3×3 row-major intrinsics matrix from image dimensions
    and an assumed horizontal FOV. Good enough for furniture-size estimation
    from monocular depth; actual FOV error of ±10° produces ~15% size error.
    """
    hfov_rad = math.radians(_DEFAULT_HFOV_DEG)
    fx = (image_w / 2.0) / math.tan(hfov_rad / 2.0)
    fy = fx  # assume square pixels
    cx = image_w / 2.0
    cy = image_h / 2.0
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def measure_furniture_monocular(
    mask_png_b64: str,
    depth_map_png_b64: str,
    depth_min_m: float,
    depth_max_m: float,
    floor_plan_points: list[dict[str, Any]],
    room_height_m: float | None = None,
) -> dict[str, float] | None:
    """Estimate furniture dimensions from a monocular metric depth map.

    Compared with measure_furniture() (the LiDAR path), this uses:
      • Pinhole formula directly in camera space (no extrinsics needed — the
        camera orientation is unknown for an uploaded photo).
      • Width and height via angular size × depth — accurate to ~15% assuming
        a roughly frontal view of the furniture.
      • Furniture depth (near-to-far) from the depth variance inside the mask —
        less reliable; treated as a secondary dimension.

    ``room_height_m`` is used as a sanity-clamp: if the computed height exceeds
    the known ceiling height, it's clamped to that value.

    Returns None when fewer than _MIN_VALID_SAMPLES survive the depth filter,
    matching the behaviour of the LiDAR path.
    """
    mask = decode_mask_png(mask_png_b64)
    depth = decode_depth_png(depth_map_png_b64, depth_min_m, depth_max_m)

    mh, mw = mask.shape
    dh, dw = depth.shape
    if mw == 0 or mh == 0 or dw == 0 or dh == 0:
        return None

    intrinsics = _estimate_intrinsics(mw, mh)
    fx, fy, cx, cy = intrinsics[0], intrinsics[4], intrinsics[2], intrinsics[5]

    # Sample depth at mask pixels (scaled to depth map resolution)
    mask_rows, mask_cols = np.where(mask)
    if mask_rows.size == 0:
        return None

    step = max(1, mask_rows.size // 256)
    sampled_rows = mask_rows[::step]
    sampled_cols = mask_cols[::step]

    dv = np.clip((sampled_rows / mh * dh).astype(int), 0, dh - 1)
    du = np.clip((sampled_cols / mw * dw).astype(int), 0, dw - 1)
    depths = depth[dv, du]

    valid = depths > _MIN_VALID_DEPTH_M
    if valid.sum() < _MIN_VALID_SAMPLES:
        return None

    valid_depths = depths[valid]
    valid_cols = sampled_cols[valid].astype(float)
    valid_rows = sampled_rows[valid].astype(float)

    # Median depth of the item — more robust than mean for partially-occluded masks
    median_depth = float(np.median(valid_depths))

    # Width: angular span of the mask's column extent at median depth
    col_min = float(valid_cols.min())
    col_max = float(valid_cols.max())
    width_m = abs((col_max - col_min) / fx * median_depth)

    # Height: angular span of the mask's row extent at median depth
    row_min = float(valid_rows.min())
    row_max = float(valid_rows.max())
    height_m = abs((row_max - row_min) / fy * median_depth)
    if room_height_m is not None and room_height_m > 0:
        height_m = min(height_m, room_height_m)

    # Furniture depth (near-to-far): depth range inside the mask. This is noisy
    # for monocular depth, so we use the IQR (25th–75th percentile) to suppress
    # outliers, then multiply by a small scale factor to compensate for the
    # known tendency of monocular models to under-estimate thin extents.
    p25, p75 = float(np.percentile(valid_depths, 25)), float(np.percentile(valid_depths, 75))
    furniture_depth_m = max(abs(p75 - p25) * 1.5, 0.01)

    return {
        "width_m": round(width_m, 3),
        "depth_m": round(furniture_depth_m, 3),
        "height_m": round(height_m, 3),
    }
