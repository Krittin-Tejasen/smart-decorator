"""Read-only snapshot of every vector's metadata in the Pinecone index named by
PINECONE_INDEX_NAME, taken before a catalog data migration touches it (e.g.
adding measured-colour fields). Embedding values are intentionally NOT saved:
the migration only ever calls index.update(..., set_metadata=...) to add new
fields, never re-upserts or deletes a vector, so the vectors themselves are
never at risk and re-downloading tens of thousands of 512-float vectors would
be slow for no benefit. If the embeddings themselves ever need restoring,
re-run scripts/ingest_products.py.

Run from backend/:
    python -m scripts.backup_pinecone
    python -m scripts.backup_pinecone --table ikea_furniture_v2

--table only affects which Supabase table's unique_ids are checked against the
index to report ids with no vector (e.g. the many rows of a table that share a
photo with another row, so only one of them was embedded) — it does not change
which Pinecone index is read; that always follows PINECONE_INDEX_NAME.

Writes backend/catalog_backups/pinecone_metadata_<UTC timestamp>.jsonl (one
{"id": ..., "metadata": {...}} per line) and a sibling .manifest.json.
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings
from app.services.pinecone_client import get_index
from app.services.product_repo import TABLE_NAME as DEFAULT_TABLE_NAME, get_supabase_client

BACKUP_DIR = Path(__file__).resolve().parents[1] / "catalog_backups"
ID_PAGE = 1000      # unique_ids read from Supabase per page
FETCH_BATCH = 1000  # ids per Pinecone fetch() call
RETRIES = 5


def all_unique_ids(table: str) -> list[str]:
    """Every unique_id in `table`, to check against the index's vector ids
    (ingest_products.py uses a row's unique_id as its vector id 1:1, though not
    every row necessarily has one — see this script's docstring), read fresh so
    a row added since the last ingest isn't missed."""
    client = get_supabase_client()
    ids: list[str] = []
    start = 0
    while True:
        batch = (
            client.table(table).select("unique_id").order("unique_id").range(start, start + ID_PAGE - 1).execute().data
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=DEFAULT_TABLE_NAME, help=f"Supabase table whose unique_ids to check (default: {DEFAULT_TABLE_NAME})")
    args = parser.parse_args()

    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    index_name = settings.pinecone_index_name
    out_path = BACKUP_DIR / f"pinecone_metadata_{index_name}_{timestamp}.jsonl"
    manifest_path = BACKUP_DIR / f"pinecone_metadata_{index_name}_{timestamp}.manifest.json"

    index = get_index()
    stats = index.describe_index_stats()
    print(f"Index '{index_name}' reports {stats.total_vector_count} vectors.")

    print(f"Listing unique_ids from Supabase table '{args.table}'...")
    ids = all_unique_ids(args.table)
    print(f"{len(ids)} ids to check.")

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
        "index": index_name,
        "table_checked_against": args.table,
        "taken_at_utc": timestamp,
        "ids_checked": len(ids),
        "vectors_backed_up": found,
        "ids_not_in_index": len(missing),
        "file": out_path.name,
        "seconds": round(time.monotonic() - t0, 1),
        "note": (
            "metadata only, no embedding values — see this script's docstring. "
            "ids_not_in_index is expected to be large when several table rows "
            "share one photo and only one of them was embedded, not an error."
        ),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {found} vectors' metadata ({len(missing)} checked ids had no vector in '{index_name}') -> {out_path}"
          f" ({out_path.stat().st_size / 1e6:.1f} MB)")
    if missing:
        print(f"  ids with no vector in the index (first 10): {missing[:10]}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
