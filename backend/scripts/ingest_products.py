"""Offline job: embed the IKEA product catalog and upsert vectors into Pinecone.

Run from the backend/ directory:
    python -m scripts.ingest_products

This replaces the old prepare_data_for_pinecone.py (a Colab notebook export).
Compared to that version, this script:
- reuses app.services.embedding so ingestion and query-time embeddings are
  always produced by the same model/preprocessing and stay comparable
- downloads and encodes images in batches instead of one request at a time
- checkpoints progress to disk, so a crash or timeout resumes instead of
  re-downloading and re-upserting everything from row 0
"""

import gc
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from app.services.embedding import download_image, encode_images
from app.services.pinecone_client import upsert_batch
from app.services.product_repo import fetch_all_products

CHECKPOINT_FILE = Path(__file__).with_name(".ingest_checkpoint.json")
# Kept modest (rather than e.g. 64) because long CPU-bound encode loops on
# Windows fragment the process heap over thousands of images; a smaller batch
# needs a smaller contiguous allocation each time, which fails less often.
BATCH_SIZE = 32
DOWNLOAD_WORKERS = 16


def load_checkpoint() -> set[str]:
    if CHECKPOINT_FILE.exists():
        return set(json.loads(CHECKPOINT_FILE.read_text()))
    return set()


def save_checkpoint(done_ids: set[str]) -> None:
    CHECKPOINT_FILE.write_text(json.dumps(sorted(done_ids)))


def _safe_float(value: Any) -> float | None:
    """depth_cm is stored as text in Supabase (unlike width_cm/height_cm), and
    any of these can contain blank/non-numeric junk, so parse defensively."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_metadata(row: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "sku": row.get("sku") or "",
        "main_title": row.get("main_title") or "",
        "category": row.get("category") or "",
        "url": row.get("url") or "",
        "color": row.get("color_extracted") or "",
        "size_label": row.get("size_label") or "",
        "material_tags": row.get("material_tags") or "",
        "material_tag": row.get("material_tag") or "",
        "in_stock": bool(row.get("in_stock")),
        "is_wood": bool(row.get("is_wood")),
        "is_metal": bool(row.get("is_metal")),
        "is_fabric": bool(row.get("is_fabric")),
        "is_plastic": bool(row.get("is_plastic")),
        "is_glass": bool(row.get("is_glass")),
    }

    # Pinecone metadata can't hold null values, so numeric fields are only
    # included when they parse cleanly - this also doubles as filterable
    # "size" metadata (width_cm/height_cm/depth_cm) for product matching,
    # instead of relying on the CLIP vector to encode physical dimensions.
    for key in ("width_cm", "height_cm", "depth_cm", "thb_price"):
        parsed = _safe_float(row.get(key))
        if parsed is not None:
            metadata[key] = parsed

    return metadata


def download_batch(rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], Any]]:
    """Download a batch of catalog images concurrently, keeping row/image pairs aligned."""
    slots: list[tuple[dict[str, Any], Any] | None] = [None] * len(rows)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {
            pool.submit(download_image, row["main_image"]): i
            for i, row in enumerate(rows)
            if row.get("main_image")
        }
        for future in as_completed(futures):
            i = futures[future]
            image = future.result()
            if image is not None:
                slots[i] = (rows[i], image)

    return [pair for pair in slots if pair is not None]


def main() -> None:
    print("Fetching product catalog from Supabase...")
    products = fetch_all_products()
    print(f"Loaded {len(products)} rows.")

    done_ids = load_checkpoint()
    pending = [row for row in products if str(row.get("unique_id")) not in done_ids]
    print(f"{len(done_ids)} already ingested, {len(pending)} remaining.")

    for start in range(0, len(pending), BATCH_SIZE):
        chunk = pending[start : start + BATCH_SIZE]
        try:
            pairs = download_batch(chunk)
            if not pairs:
                continue

            rows = [row for row, _ in pairs]
            images = [image for _, image in pairs]
            vectors = encode_images(images, batch_size=BATCH_SIZE)

            upsert_batch(
                [
                    {
                        "id": str(row["unique_id"]),
                        "values": vector,
                        "metadata": build_metadata(row),
                    }
                    for row, vector in zip(rows, vectors)
                ]
            )

            done_ids.update(str(row["unique_id"]) for row in rows)
            save_checkpoint(done_ids)
            print(f"Upserted {len(done_ids)}/{len(products)}...")
        except Exception as exc:
            # Don't let one bad batch (e.g. a transient memory allocation
            # failure) kill a job that's hours into a ~60k-row catalog. It
            # isn't marked in the checkpoint, so the next run retries it.
            print(f"Batch at row {start} failed, will retry on next run: {type(exc).__name__}: {exc}")
        finally:
            # Proactively release the batch's images/tensors instead of
            # waiting for Python's GC to get around to it - this is what
            # keeps process heap fragmentation (and the crash above) from
            # building back up over thousands of images.
            gc.collect()

    print("Done.")


if __name__ == "__main__":
    main()
