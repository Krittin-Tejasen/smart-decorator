"""
feature_extraction.py
─────────────────────
Colour + material features for a furniture item cut out of a generated room.

Two independent parts, deliberately split by what each is good at:

1. **Colour — measured, local, free.** The item's own pixels (alpha > 0) are
   clustered in CIELAB, near-identical clusters are merged (ΔE2000 < 6), and
   each swatch is mapped to the nearest of the product catalog's 12 colour
   words so it can be matched against `ikea_furniture.color_extracted`.

2. **Material (+ finish, colour name, usable flag) — Gemini, one call per room.**
   Prototype on 51 hand-labelled cut-outs from 5 saved rooms: Gemini 50/51
   top-1 vs. CLIP zero-shot 13–19/51, so CLIP is not used for this. All the
   items of a room go in a single request (same accuracy as one call per item,
   far fewer calls). Failure here never fails segmentation: items just keep the
   pixel-based colours and `meta.source == "pixels_only"`.

Env:
  FEATURE_PROVIDER      "gemini" (default) | "none"  — "none" skips the Gemini call
  GEMINI_API_KEY        required for "gemini"
  GEMINI_FEATURE_MODEL  default: GEMINI_VISION_MODEL, else gemini-3.6-flash
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
from io import BytesIO
from typing import Any, Awaitable, Callable, NamedTuple

import httpx
import numpy as np
from PIL import Image, ImageFilter
from pydantic import BaseModel

from feature_vocab import CATALOG_COLOR_SHADES, MATERIALS

# ── Response models (embedded in segmentation.FurnitureFeatures) ────────────


class ColorSwatch(BaseModel):
    hex: str                 # "#c3a686" — measured
    catalog_color: str       # one of the catalog's 12 colour words — use this for matching
    share: float             # fraction of the item's (kept) pixels, shares sum to ~1
    lab: list[float]         # [L, a, b] (D65) — for ΔE comparisons


class MaterialGuess(BaseModel):
    name: str                # one of feature_vocab.MATERIALS
    confidence: float        # 0-1, as stated by the model (not calibrated)


class FeatureMeta(BaseModel):
    source: str = "pixels_only"   # "gemini" | "pixels_only"
    usable: bool = True           # False: Gemini says this cut-out isn't a recognisable item


class Enrichment(BaseModel):
    materials: list[MaterialGuess] = []
    finish: str | None = None
    color_name: str | None = None
    usable: bool = True


# ── Colour maths (CIELAB, CIEDE2000) ────────────────────────────────────────

_D65 = np.array([0.95047, 1.0, 1.08883])


def srgb_to_lab(rgb: Any) -> np.ndarray:
    """(N,3) sRGB 0-255 -> (N,3) CIELAB under D65."""
    c = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    m = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ]
    )
    xyz = (lin @ m.T) / _D65
    f = np.where(xyz > 216 / 24389, np.cbrt(xyz), (24389 / 27 * xyz + 16) / 116)
    return np.stack([116 * f[:, 1] - 16, 500 * (f[:, 0] - f[:, 1]), 200 * (f[:, 1] - f[:, 2])], axis=1)


def lab_to_hex(lab: Any) -> str:
    L, a, b = (float(v) for v in np.asarray(lab).reshape(3))
    fy = (L + 16) / 116
    f = np.array([fy + a / 500, fy, fy - b / 200])
    xyz = np.where(f**3 > 216 / 24389, f**3, (116 * f - 16) / (24389 / 27)) * _D65
    m_inv = np.array(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ]
    )
    lin = np.clip(m_inv @ xyz, 0, 1)
    srgb = np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * lin ** (1 / 2.4) - 0.055)
    r, g, bl = (int(round(float(v) * 255)) for v in np.clip(srgb, 0, 1))
    return f"#{r:02x}{g:02x}{bl:02x}"


def ciede2000(lab1: Any, lab2: Any) -> np.ndarray:
    """CIEDE2000 between broadcastable (...,3) Lab arrays."""
    lab1, lab2 = np.asarray(lab1, dtype=np.float64), np.asarray(lab2, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm**7 / (Cm**7 + 25**7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1 = np.degrees(np.arctan2(b1, a1p)) % 360
    h2 = np.degrees(np.arctan2(b2, a2p)) % 360
    dL, dC = L2 - L1, C2p - C1p
    dh = h2 - h1
    dh = np.where(dh > 180, dh - 360, np.where(dh < -180, dh + 360, dh))
    dh = np.where(C1p * C2p == 0, 0, dh)
    dH = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dh / 2))
    Lm, Cmp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1 + h2
    hm = np.where(
        C1p * C2p == 0,
        hs,
        np.where(np.abs(h1 - h2) <= 180, hs / 2, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2)),
    )
    T = (
        1
        - 0.17 * np.cos(np.radians(hm - 30))
        + 0.24 * np.cos(np.radians(2 * hm))
        + 0.32 * np.cos(np.radians(3 * hm + 6))
        - 0.20 * np.cos(np.radians(4 * hm - 63))
    )
    Sl = 1 + 0.015 * (Lm - 50) ** 2 / np.sqrt(20 + (Lm - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cmp, 1 + 0.015 * Cmp * T
    d_theta = 30 * np.exp(-(((hm - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cmp**7 / (Cmp**7 + 25**7))
    Rt = -np.sin(np.radians(2 * d_theta)) * Rc
    return np.sqrt((dL / Sl) ** 2 + (dC / Sc) ** 2 + (dH / Sh) ** 2 + Rt * (dC / Sc) * (dH / Sh))


# One row per shade; several rows can belong to the same colour word.
_SHADE_NAMES = [name for name, shades in CATALOG_COLOR_SHADES.items() for _ in shades]
_SHADE_LAB = srgb_to_lab([rgb for shades in CATALOG_COLOR_SHADES.values() for rgb in shades])


def catalog_color_for_lab(lab: Any) -> str:
    """The catalog colour word (white, beige, gray …) closest to a Lab colour."""
    distances = ciede2000(np.asarray(lab, dtype=np.float64).reshape(1, 3), _SHADE_LAB)
    return _SHADE_NAMES[int(np.argmin(distances))]


# ── Pixel-based swatches ────────────────────────────────────────────────────

_MAX_SAMPLE_PIXELS = 4000
_MERGE_DELTA_E = 6.0
_L_PERCENTILES = (8, 97)   # drop deep shadow / specular highlight


def _kmeans(points: np.ndarray, k: int, iters: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic k-means++ on (N,3) points -> (centres, counts)."""
    rng = np.random.RandomState(0)
    centres = [points[rng.randint(len(points))]]
    for _ in range(1, k):
        d2 = np.min(((points[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        if d2.sum() <= 1e-9:
            break
        centres.append(points[rng.choice(len(points), p=d2 / d2.sum())])
    centres_arr = np.array(centres)
    for _ in range(iters):
        labels = ((points[:, None, :] - centres_arr[None]) ** 2).sum(-1).argmin(1)
        updated = np.array(
            [
                points[labels == j].mean(0) if np.any(labels == j) else centres_arr[j]
                for j in range(len(centres_arr))
            ]
        )
        if np.allclose(updated, centres_arr, atol=1e-3):
            break
        centres_arr = updated
    labels = ((points[:, None, :] - centres_arr[None]) ** 2).sum(-1).argmin(1)
    return centres_arr, np.bincount(labels, minlength=len(centres_arr))


def _item_pixels(rgba: Image.Image) -> np.ndarray:
    """RGB pixels of the item itself: alpha > 127, eroded a little so the
    blended edge (wall bleeding through) doesn't pollute the colours."""
    arr = np.asarray(rgba.convert("RGBA"))
    alpha = Image.fromarray(((arr[..., 3] > 127) * 255).astype("uint8"), "L")
    core_alpha = alpha
    if int((np.asarray(alpha) > 0).sum()) > 2000:
        radius = max(1, min(5, round(min(alpha.size) * 0.01)))
        core_alpha = alpha.filter(ImageFilter.MinFilter(2 * radius + 1))
    selected = arr[..., :3][np.asarray(core_alpha) > 0]
    if len(selected) < 64:
        selected = arr[..., :3][np.asarray(alpha) > 0]
    if len(selected) > _MAX_SAMPLE_PIXELS:
        selected = selected[np.linspace(0, len(selected) - 1, _MAX_SAMPLE_PIXELS).astype(int)]
    return selected


def extract_swatches(rgba: Image.Image, k: int = 3) -> list[ColorSwatch]:
    """Up to `k` dominant colours of the item's own pixels, most common first.
    Returns [] when the image has (almost) no opaque pixels."""
    pixels = _item_pixels(rgba)
    if len(pixels) < 16:
        return []

    lab = srgb_to_lab(pixels)
    lo, hi = np.percentile(lab[:, 0], _L_PERCENTILES)
    kept = lab[(lab[:, 0] >= lo) & (lab[:, 0] <= hi)]
    if len(kept) < 32:
        kept = lab

    centres, counts = _kmeans(kept, min(5, len(kept)))
    order = np.argsort(-counts)
    merged: list[dict] = []
    for j in order:
        centre, weight = centres[j], counts[j] / counts.sum()
        for m in merged:
            if float(ciede2000(m["lab"], centre)) < _MERGE_DELTA_E:
                total = m["share"] + weight
                m["lab"] = (m["lab"] * m["share"] + centre * weight) / total
                m["share"] = total
                break
        else:
            merged.append({"lab": centre.copy(), "share": float(weight)})

    merged.sort(key=lambda m: -m["share"])
    top = merged[:k]
    total_share = sum(m["share"] for m in top) or 1.0
    swatches = []
    for m in top:
        group = catalog_color_for_lab(m["lab"])
        swatches.append(
            ColorSwatch(
                hex=lab_to_hex(m["lab"]),
                catalog_color=group,
                share=round(m["share"] / total_share, 2),
                lab=[round(float(v), 1) for v in m["lab"]],
            )
        )
    return swatches


# ── Gemini enrichment: material, finish, colour name, usable ────────────────

_BATCH_SIZE = 12          # cut-outs per request
_IMAGE_PX = 256
_MAX_NAME_LEN = 40


class EnrichInput(NamedTuple):
    id: str
    label: str
    image: Image.Image     # RGBA cut-out, or an RGB crop when there is no mask


PostFn = Callable[[dict], Awaitable[dict]]

# The Gemini API key travels as a `?key=` query parameter, and httpx puts the full
# URL in its error messages — so an error must never be printed as-is.
_SECRET_QUERY = re.compile(r"([?&](?:key|api_key|apikey|token)=)[^&\s]+", re.IGNORECASE)


def safe_error(exc: BaseException) -> str:
    """`ExcType: message` for logs, with any API key in a URL blanked out."""
    text = _SECRET_QUERY.sub(r"\1REDACTED", str(exc))
    return f"{type(exc).__name__}: {text}"


def feature_provider() -> str:
    return os.getenv("FEATURE_PROVIDER", "gemini").strip().lower()


def prepare_image(image: Image.Image) -> bytes:
    """Cut-out flattened onto white (what the model saw in the prototype), small JPEG."""
    rgba = image.convert("RGBA")
    flat = Image.new("RGB", rgba.size, (255, 255, 255))
    flat.paste(rgba, mask=rgba.split()[-1])
    flat.thumbnail((_IMAGE_PX, _IMAGE_PX), Image.LANCZOS)
    buf = BytesIO()
    flat.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


_INSTRUCTIONS = f"""Each image below is one piece of furniture/decor cut out of the SAME AI-generated interior photo (background removed, shown on white). Each is preceded by its id and a type hint.
For EVERY id, describe what it is MADE OF as it visually appears. Materials: choose ONLY from {list(MATERIALS)} (up to 2, dominant first, with confidence 0-1).
Return a JSON array, one object per id:
{{"id": <id>, "materials": [{{"name": <word>, "confidence": <0-1>}}], "finish": <2-4 words, e.g. "light oak, matte">, "color_name": <plain colour name of the main surface, e.g. "terracotta">, "usable": <false if the image is not a recognisable item of the given type, e.g. a floor strip or fragment>}}"""


def build_request(inputs: list[EnrichInput]) -> dict:
    parts: list[dict] = [{"text": _INSTRUCTIONS}]
    for item in inputs:
        parts.append({"text": f'id={item.id} type="{item.label}"'})
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(prepare_image(item.image)).decode(),
                }
            }
        )
    return {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
    }


def _clean_text(value: Any, limit: int = _MAX_NAME_LEN) -> str | None:
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text[:limit] or None


def parse_enrichment(payload: Any, wanted_ids: set[str]) -> dict[str, Enrichment]:
    """Model JSON -> {id: Enrichment}. Tolerant: ignores unknown ids and words
    outside the vocabulary, clamps confidences, never raises on bad shapes."""
    if isinstance(payload, dict):
        payload = payload.get("items") or payload.get("results") or []
    if not isinstance(payload, list):
        return {}

    out: dict[str, Enrichment] = {}
    for entry in payload:
        if not isinstance(entry, dict):
            continue
        item_id = str(entry.get("id", ""))
        if item_id not in wanted_ids:
            continue

        guesses: list[MaterialGuess] = []
        seen: set[str] = set()
        for raw in entry.get("materials") or []:
            if not isinstance(raw, dict):
                continue
            name = str(raw.get("name", "")).strip().lower()
            if name not in MATERIALS or name in seen:
                continue
            try:
                conf = float(raw.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            guesses.append(MaterialGuess(name=name, confidence=round(min(max(conf, 0.0), 1.0), 2)))
            seen.add(name)

        out[item_id] = Enrichment(
            materials=guesses[:2],
            finish=_clean_text(entry.get("finish")),
            color_name=_clean_text(entry.get("color_name")),
            usable=entry.get("usable") is not False,
        )
    return out


def _response_json(response: dict) -> Any:
    text = response["candidates"][0]["content"]["parts"][0]["text"]
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    return json.loads(text)


async def _post_to_gemini(body: dict) -> dict:
    api_key = os.getenv("GEMINI_API_KEY", "")
    model = os.getenv("GEMINI_FEATURE_MODEL") or os.getenv("GEMINI_VISION_MODEL", "gemini-3.6-flash")
    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            json=body,
        )
        response.raise_for_status()
        return response.json()


async def enrich_features(
    inputs: list[EnrichInput], post: PostFn | None = None
) -> dict[str, Enrichment]:
    """Material / finish / colour name / usable for each input, keyed by id.

    One request per `_BATCH_SIZE` items (so one per room in practice). Returns
    {} — never raises — when the provider is "none", there is no API key, or
    the call/parse fails; callers treat missing ids as "not enriched".
    """
    if not inputs or feature_provider() != "gemini":
        return {}
    if post is None:
        if not os.getenv("GEMINI_API_KEY"):
            return {}
        post = _post_to_gemini

    async def one_batch(batch: list[EnrichInput]) -> dict[str, Enrichment]:
        try:
            response = await post(build_request(batch))
            return parse_enrichment(_response_json(response), {b.id for b in batch})
        except Exception as exc:  # network, quota, malformed JSON …
            print(f"[features] Gemini enrichment failed for {len(batch)} item(s): {safe_error(exc)}")
            return {}

    batches = [inputs[i : i + _BATCH_SIZE] for i in range(0, len(inputs), _BATCH_SIZE)]
    merged: dict[str, Enrichment] = {}
    for result in await asyncio.gather(*(one_batch(b) for b in batches)):
        merged.update(result)
    return merged
