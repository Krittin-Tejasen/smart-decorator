"""Shared CLIP embedding logic.

Used by both the FastAPI app (query time, once product matching is wired up)
and the offline ingestion script (scripts/ingest_products.py). Keeping this in
one place matters: catalog vectors and query vectors must come from the exact
same model and preprocessing, otherwise cosine similarity between them is
meaningless.
"""

from functools import lru_cache
from io import BytesIO

import httpx
from PIL import Image
from sentence_transformers import SentenceTransformer

from app.core.config import settings


@lru_cache
def get_model() -> SentenceTransformer:
    return SentenceTransformer(settings.clip_model_name)


def _flatten_to_white(image: Image.Image) -> Image.Image:
    """Paste any transparent image onto a white background.

    Segmented furniture crops (from SAM) have transparent backgrounds, while
    the IKEA catalog photos are shot on plain white. Without this step the two
    sets of embeddings drift apart and stop being comparable.
    """
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")

    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[-1])
    return background


def encode_image(image: Image.Image) -> list[float]:
    prepared = _flatten_to_white(image)
    return get_model().encode(prepared).tolist()


def encode_images(images: list[Image.Image], batch_size: int = 64) -> list[list[float]]:
    prepared = [_flatten_to_white(image) for image in images]
    vectors = get_model().encode(prepared, batch_size=batch_size, show_progress_bar=False)
    return vectors.tolist()


def download_image(url: str, timeout: float = 10.0) -> Image.Image | None:
    try:
        response = httpx.get(url, timeout=timeout, follow_redirects=True)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        image.load()
        return image
    except Exception:
        return None


def encode_image_url(url: str) -> list[float] | None:
    image = download_image(url)
    if image is None:
        return None
    return encode_image(image)
