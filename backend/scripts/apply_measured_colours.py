"""Write the colour measured by catalog_colour.py into ikea_furniture_v2's
measured_color / measured_color_low_confidence / measured_swatches /
measured_at columns (added by hand via SQL migration — this script never
touches schema).

Run from backend/:
    python -m scripts.apply_measured_colours --list                      # show candidate files
    python -m scripts.apply_measured_colours --from <file>.jsonl          # dry-run (default), no writes
    python -m scripts.apply_measured_colours --from <file>.jsonl --apply  # actually write

Input is a measure_catalog_colours.py output file (one JSON object per
product/photo-group: group_key, unique_ids, old_colors_seen, measured:
{swatches, foreground_share, low_confidence}). Every unique_id in a group
gets the same colour, since they share the same photo.

Supabase/PostgREST upsert validates NOT NULL constraints (e.g. `sku`) on the
row as it would be INSERTed, even when the row already exists and only an
UPDATE happens on conflict — so a partial payload of only unique_id +
measured_* columns fails outright. To avoid that, each row is merged onto
its full base row from the backup_catalog.py snapshot this run was given
(the --from file's sibling `ikea_furniture_v2_*.jsonl`, or pass
--base-snapshot explicitly), so every NOT NULL column is still present and
unchanged; only the measured_* columns actually move.

For a low_confidence group, measured_color is left NULL (we don't trust the
top swatch enough to call it the product's colour) but measured_swatches /
measured_at / measured_color_low_confidence=true are still written, so the
attempt is visible rather than silently missing — a future query can always
tell "not measured yet" (measured_at IS NULL) apart from "measured but not
confident" (measured_color_low_confidence).

--dry-run (the default) only prints counts and a sample of before/after
diffs against the live table, and never calls .upsert(). Nothing is written
until --apply is passed explicitly and you are asked to confirm the row
count — same pattern as restore_catalog.py.

Rollback: this only touches the 4 new columns, so undoing it is
    ALTER TABLE ikea_furniture_v2
      DROP COLUMN measured_color, DROP COLUMN measured_color_low_confidence,
      DROP COLUMN measured_swatches, DROP COLUMN measured_at;
(the full-row snapshot from backup_catalog.py is the belt-and-suspenders
fallback if something else also needs undoing).
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.product_repo import get_supabase_client

BACKUP_DIR = Path(__file__).resolve().parent.parent / "catalog_backups"
TABLE = "ikea_furniture_v2"
BATCH_SIZE = 200


def load_measured(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_base_rows(path: Path) -> dict[str, dict[str, Any]]:
    """unique_id -> full row, from a backup_catalog.py snapshot of this table."""
    rows = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                rows[row["unique_id"]] = row
    return rows


def build_updates(
    groups: list[dict[str, Any]], measured_at: str, base_rows: dict[str, dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    updates = []
    missing_base = 0
    for g in groups:
        m = g["measured"]
        low_conf = bool(m["low_confidence"])
        swatches = m["swatches"]
        color = None if low_conf or not swatches else swatches[0]["catalog_color"]
        for uid in g["unique_ids"]:
            base = base_rows.get(uid)
            if base is None:
                missing_base += 1
                continue
            updates.append({
                **base,
                "measured_color": color,
                "measured_color_low_confidence": low_conf,
                "measured_swatches": swatches,
                "measured_at": measured_at,
            })
    return updates, missing_base


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list measured_colours_*.jsonl files and exit")
    parser.add_argument("--from", dest="source", help="measure_catalog_colours.py output .jsonl (in catalog_backups/ or a full path)")
    parser.add_argument("--base-snapshot", help="backup_catalog.py snapshot of ikea_furniture_v2 to merge onto (default: newest ikea_furniture_v2_*.jsonl in catalog_backups/)")
    parser.add_argument("--apply", action="store_true", help="actually write (default is dry-run)")
    parser.add_argument("--sample", type=int, default=20, help="rows to diff in dry-run mode (default 20)")
    args = parser.parse_args()

    if args.list or not args.source:
        files = sorted(BACKUP_DIR.glob("measured_colours_*.jsonl"))
        if not files:
            print(f"No measured_colours_*.jsonl in {BACKUP_DIR}. Run measure_catalog_colours.py first.")
        for f in files:
            print(f"  {f.name:48s} {f.stat().st_size / 1e6:6.1f} MB")
        return

    src_path = Path(args.source)
    if not src_path.is_absolute() and not src_path.exists():
        src_path = BACKUP_DIR / src_path
    if not src_path.exists():
        sys.exit(f"File not found: {src_path}")

    base_path = Path(args.base_snapshot) if args.base_snapshot else None
    if base_path and not base_path.is_absolute() and not base_path.exists():
        base_path = BACKUP_DIR / base_path
    if base_path is None:
        candidates = sorted(BACKUP_DIR.glob(f"{TABLE}_*.jsonl"))
        if not candidates:
            sys.exit(f"No {TABLE}_*.jsonl snapshot in {BACKUP_DIR} to merge onto (needed so NOT NULL "
                      f"columns stay populated on upsert). Run backup_catalog.py --table {TABLE} first, "
                      f"or pass --base-snapshot.")
        base_path = candidates[-1]
    print(f"Loading base rows from {base_path} ...")
    base_rows = load_base_rows(base_path)

    print(f"Loading {src_path} ...")
    groups = load_measured(src_path)
    measured_at = datetime.now(timezone.utc).isoformat()
    updates, missing_base = build_updates(groups, measured_at, base_rows)

    n_low_conf = sum(1 for g in groups if g["measured"]["low_confidence"])
    n_colored = sum(1 for u in updates if u["measured_color"] is not None)
    print(f"{len(groups)} product groups -> {len(updates)} rows (unique_ids) to update in '{TABLE}'.")
    print(f"  {n_colored} rows get a measured_color; {len(updates) - n_colored} rows stay NULL (low_confidence or no swatch).")
    print(f"  ({n_low_conf}/{len(groups)} groups were low_confidence.)")
    if missing_base:
        print(f"  {missing_base} unique_ids from the measured file had no row in {base_path.name} (skipped, not updated).")

    client = get_supabase_client()

    if not args.apply:
        print(f"\nDRY RUN — diffing {min(args.sample, len(updates))} sample rows against the live table (no writes):")
        sample = updates[:: max(1, len(updates) // args.sample)][: args.sample]
        sample_ids = [u["unique_id"] for u in sample]
        live_by_id = {}
        for i in range(0, len(sample_ids), 50):
            chunk = sample_ids[i : i + 50]
            cols = "unique_id,measured_color,measured_color_low_confidence,measured_at"
            for row in client.table(TABLE).select(cols).in_("unique_id", chunk).execute().data:
                live_by_id[row["unique_id"]] = row
        for u in sample:
            live = live_by_id.get(u["unique_id"])
            if live is None:
                print(f"  {u['unique_id']:16s} <not found in live table!>")
                continue
            was = live.get("measured_color")
            will_be = u["measured_color"]
            print(f"  {u['unique_id']:16s} measured_color: {was!r} -> {will_be!r}"
                  f"{'  (low_confidence)' if u['measured_color_low_confidence'] else ''}")
        print("\nNothing was written. Re-run with --apply to write for real.")
        return

    print(f"\nAbout to write measured_color/measured_swatches/measured_at for {len(updates)} rows of '{TABLE}'.")
    confirm = input(f"Type the row count ({len(updates)}) to confirm: ").strip()
    if confirm != str(len(updates)):
        sys.exit("Row count didn't match — aborted, nothing written.")

    written = 0
    t0 = time.monotonic()
    for i in range(0, len(updates), BATCH_SIZE):
        batch = updates[i : i + BATCH_SIZE]
        for attempt in range(5):
            try:
                client.table(TABLE).upsert(batch, on_conflict="unique_id").execute()
                break
            except Exception as exc:
                print(f"  batch {i} failed ({type(exc).__name__}), retry {attempt + 1}/5")
                time.sleep(3 * (attempt + 1))
        else:
            sys.exit(f"Gave up on batch starting at row {i}; {written} rows written so far.")
        written += len(batch)
        if written % 3000 == 0:
            print(f"  wrote {written}/{len(updates)} ({time.monotonic() - t0:.0f}s)...")

    print(f"\nWrote {written} rows from {src_path.name}.")


if __name__ == "__main__":
    main()
