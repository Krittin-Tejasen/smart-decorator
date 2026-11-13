"""Load the cleaned catalog CSV into the Supabase table ikea_furniture_v2.

Run from the backend/ directory:
    python -m scripts.load_catalog PATH/catalog_clean.csv              dry run: checks the file, writes nothing
    python -m scripts.load_catalog PATH/catalog_clean.csv --apply      upsert every row (resumable)

Create the table first with supabase/migrations/20261101000000_ikea_furniture_v2.sql.
It only writes to ikea_furniture_v2: ikea_furniture is never touched. Rows are
upserted on unique_id, so a run that stops half-way can be started again and
a second full run changes nothing.
"""

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any

from app.services.product_repo import get_supabase_client

TABLE = "ikea_furniture_v2"
BATCH = 500
csv.field_size_limit(2**31 - 1)

TEXT = [
    "unique_id", "sku", "group_key", "main_title", "description", "search_context", "main_image", "url",
    "category", "category_id", "furniture_type", "type_source", "country_code", "currency", "size_label",
    "size", "material_tags", "material_tag", "color", "color_extracted",
]
NUMBER = ["final_price", "thb_price", "width_cm", "height_cm", "depth_cm", "length_cm", "diameter_cm"]
BOOLEAN = ["in_stock", "has_dims", "is_wood", "is_metal", "is_fabric", "is_plastic", "is_glass"]
REQUIRED = ["unique_id", "sku", "group_key", "furniture_type"]


def convert(raw: dict[str, str]) -> dict[str, Any]:
    """One CSV row as a table row: blank -> null, numbers and booleans typed."""
    row: dict[str, Any] = {}
    for key in TEXT:
        value = raw.get(key)
        row[key] = value if value not in (None, "") else None
    for key in NUMBER:
        value = raw.get(key)
        row[key] = float(value) if value not in (None, "") else None
    for key in BOOLEAN:
        value = raw.get(key)
        row[key] = {"True": True, "False": False}.get(value) if value not in (None, "") else None
    row["has_dims"] = bool(row["has_dims"])
    return row


def check(rows: list[dict[str, Any]]) -> list[str]:
    problems = []
    ids = [r["unique_id"] for r in rows]
    if len(set(ids)) != len(ids):
        problems.append(f"{len(ids) - len(set(ids))} duplicate unique_id")
    for key in REQUIRED:
        missing = sum(1 for r in rows if not r[key])
        if missing:
            problems.append(f"{missing} rows without {key}")
    unnamed = sum(1 for r in rows if r["furniture_type"] == "other")
    if unnamed:
        problems.append(f"{unnamed} rows with furniture_type 'other' (every product must say what it is)")
    bad_source = sum(1 for r in rows if r["type_source"] not in ("title", "category"))
    if bad_source:
        problems.append(f"{bad_source} rows with a type_source other than title/category")
    return problems


def upsert(client, rows: list[dict[str, Any]], retries: int = 5) -> None:
    for attempt in range(retries):
        try:
            client.table(TABLE).upsert(rows, on_conflict="unique_id").execute()
            return
        except Exception:  # noqa: BLE001 - network drop: retried, then raised
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--apply", action="store_true", help="write to Supabase (default: dry run)")
    args = parser.parse_args()

    with open(args.csv_path, encoding="utf-8", newline="") as handle:
        rows = [convert(r) for r in csv.DictReader(handle)]
    print(f"Read {len(rows)} rows from {args.csv_path}.")

    problems = check(rows)
    if problems:
        sys.exit("Not loading: " + "; ".join(problems))
    print("File checks passed.")

    client = get_supabase_client()
    try:
        before = client.table(TABLE).select("unique_id", count="exact").limit(1).execute().count
    except Exception as exc:  # noqa: BLE001
        sys.exit(f"Table {TABLE} is not readable ({exc}). Run the migration first.")
    print(f"{TABLE} has {before} rows now.")

    if not args.apply:
        print("Dry run: nothing written. Add --apply to load.")
        return

    for start in range(0, len(rows), BATCH):
        upsert(client, rows[start : start + BATCH])
        if (start // BATCH) % 20 == 0:
            print(f"  {min(start + BATCH, len(rows))}/{len(rows)}")
    after = client.table(TABLE).select("unique_id", count="exact").limit(1).execute().count
    print(f"Done. {TABLE} has {after} rows (file had {len(rows)}).")
    if after != len(rows):
        sys.exit("Row count differs from the file: check before using the table.")


if __name__ == "__main__":
    main()
