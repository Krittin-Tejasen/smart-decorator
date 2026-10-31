"""Read-only snapshot of every vector's metadata in the Pinecone `ikea-products`
index, taken before a catalog data migration touches it (e.g. adding measured-
colour fields). Embedding values are intentionally NOT saved: the migration only
ever calls index.update(..., set_metadata=...) to add new fields, never
re-upserts or deletes a vector, so the vectors themselves are never at risk and
re-downloading ~60k x 512-float vectors would be slow for no benefit. If the
embeddings themselves ever need restoring, re-run scripts/ingest_products.py.

Run from backend/:
    python -m scripts.backup_pinecone

Writes backend/catalog_backups/pinecone_metadata_<UTC timestamp>.jsonl (one
{"id": ..., "metadata": {...}} per line) and a sibling .manifest.json.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.pinecone_client import get_index
from app.services.product_repo import get_supabase_client, TABLE_NAME

BACKUP_DIR = Path(__file__).resolve().parents[1] / "catalog_backups"
ID_PAGE = 1000      # unique_ids read from Supabase per page
FETCH_BATCH = 1000  # ids per Pinecone fetch() call
RETRIES = 5


def all_unique_ids() -> list[str]:
    """Every id Pinecone's vectors are keyed by (ingest_products.py uses the
    Supabase unique_id as the vector id 1:1), read fresh so a vector added
    since the last ingest isn't missed."""
    client = get_supabase_client()
    ids: list[str] = []
    start = 0
    while True:
        batch = (
            client.table(TABLE_NAME).select("unique_id").order("unique_id").range(start, start + ID_PAGE - 1).execute().data
        )
        ids.extend(str(r["unique_id"]) for r in batch)
        if len(batch) < ID_PAGE:
            break
        start += ID_PAGE
    return ids


def fetch_with_retries(index, ids: list[str]):
    last_exc = None
    for attempt in range(RETRIES):
        try:
            return index.fetch(ids=ids)
        except Exception as exc:
            last_exc = exc
            wait = 3 * (attempt + 1)
            print(f"  batch of {len(ids)} failed ({type(exc).__name__}), retry {attempt + 1}/{RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up on a batch of {len(ids)} ids after {RETRIES} attempts") from last_exc


def main() -> None:
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BACKUP_DIR / f"pinecone_metadata_{timestamp}.jsonl"
    manifest_path = BACKUP_DIR / f"pinecone_metadata_{timestamp}.manifest.json"

    index = get_index()
    stats = index.describe_index_stats()
    print(f"Index reports {stats.total_vector_count} vectors.")

    print("Listing unique_ids from Supabase...")
    ids = all_unique_ids()
    print(f"{len(ids)} ids to back up.")

    found = 0
    missing: list[str] = []
    t0 = time.monotonic()
    with out_path.open("w", encoding="utf-8") as f:
        for i in range(0, len(ids), FETCH_BATCH):
            chunk = ids[i : i + FETCH_BATCH]
            response = fetch_with_retries(index, chunk)
            returned = response.vectors
            for vector_id in chunk:
                vector = returned.get(vector_id)
                if vector is None:
                    missing.append(vector_id)
                    continue
                f.write(json.dumps({"id": vector_id, "metadata": vector.metadata or {}}, ensure_ascii=False) + "\n")
                found += 1
            if (i // FETCH_BATCH) % 10 == 0:
                print(f"  {found}/{len(ids)} ({time.monotonic() - t0:.0f}s)...")

    manifest = {
        "index": "ikea-products (see PINECONE_INDEX_NAME)",
        "taken_at_utc": timestamp,
        "ids_checked": len(ids),
        "vectors_backed_up": found,
        "ids_not_in_index": len(missing),
        "file": out_path.name,
        "seconds": round(time.monotonic() - t0, 1),
        "note": "metadata only, no embedding values — see this script's docstring",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {found} vectors' metadata ({len(missing)} ids had no vector) -> {out_path}"
          f" ({out_path.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print(f"  ids with no vector in the index (first 10): {missing[:10]}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
