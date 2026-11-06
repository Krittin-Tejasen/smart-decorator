"""Read-only snapshot of a Supabase catalog table, taken before any catalog
data migration (colour clean-up, material clean-up, dedupe...) so the
migration can be rolled back by restoring from this file.

Run from backend/:
    python -m scripts.backup_catalog                       # the default table (ikea_furniture)
    python -m scripts.backup_catalog --table ikea_furniture_v2

Writes backend/catalog_backups/<table>_<UTC timestamp>.jsonl — one JSON object
per row (every column, `select *`), so a restore can replay it row by row
without needing to know today's schema in advance. Also writes a sibling
`<name>.manifest.json` with the row count and a few sanity checks.

Paginated and retried like app.services.product_repo.fetch_all_products_light:
a page that still fails after several tries raises, so a partial file is never
mistaken for a full backup (its manifest would be missing).
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.product_repo import TABLE_NAME as DEFAULT_TABLE_NAME, get_supabase_client

BACKUP_DIR = Path(__file__).resolve().parents[1] / "catalog_backups"
PAGE_SIZE = 300
RETRIES = 6


def fetch_page(client, table: str, start: int, end: int):
    last_exc = None
    for attempt in range(RETRIES):
        try:
            return client.table(table).select("*").order("unique_id").range(start, end).execute().data
        except Exception as exc:  # network drop, statement timeout, ...
            last_exc = exc
            wait = 3 * (attempt + 1)
            print(f"  page {start}-{end} failed ({type(exc).__name__}), retry {attempt + 1}/{RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up on page {start}-{end} after {RETRIES} attempts") from last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", default=DEFAULT_TABLE_NAME, help=f"Supabase table to back up (default: {DEFAULT_TABLE_NAME})")
    args = parser.parse_args()
    table = args.table

    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = BACKUP_DIR / f"{table}_{timestamp}.jsonl"
    manifest_path = BACKUP_DIR / f"{table}_{timestamp}.manifest.json"

    client = get_supabase_client()
    total = 0
    start = 0
    columns: set[str] = set()
    t0 = time.monotonic()
    with out_path.open("w", encoding="utf-8") as f:
        while True:
            batch = fetch_page(client, table, start, start + PAGE_SIZE - 1)
            if not batch:
                break
            for row in batch:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                columns |= row.keys()
            total += len(batch)
            start += PAGE_SIZE
            if total % 3000 == 0:
                print(f"  {total} rows ({time.monotonic() - t0:.0f}s)...")
            if len(batch) < PAGE_SIZE:
                break

    manifest = {
        "table": table,
        "taken_at_utc": timestamp,
        "row_count": total,
        "columns": sorted(columns),
        "file": out_path.name,
        "seconds": round(time.monotonic() - t0, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nDone: {total} rows, {len(columns)} columns -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
