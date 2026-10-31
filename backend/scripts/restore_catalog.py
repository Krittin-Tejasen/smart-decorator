"""Roll back `ikea_furniture` to a snapshot taken by backup_catalog.py.

Run from backend/:
    python -m scripts.restore_catalog --list                       # show snapshots
    python -m scripts.restore_catalog --from <file>.jsonl --dry-run # preview, no writes
    python -m scripts.restore_catalog --from <file>.jsonl --apply   # actually restore

--dry-run (the default) only reports, row by row, which columns differ between
the live table and the snapshot, and never calls .upsert(). Nothing is written
until --apply is passed explicitly and you are asked to confirm the row count.

Restoring writes back EVERY column of every row in the snapshot, so it only
makes sense as a full undo of a migration that touched this table — not as a
way to merge in a handful of rows.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from app.services.product_repo import TABLE_NAME, get_supabase_client

BACKUP_DIR = Path(__file__).resolve().parent.parent / "catalog_backups"
BATCH_SIZE = 200


def load_snapshot(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def diff_row(live: dict[str, Any] | None, snap: dict[str, Any]) -> list[str]:
    if live is None:
        return ["<row missing live>"]
    changed = []
    for key, snap_value in snap.items():
        if live.get(key) != snap_value:
            changed.append(key)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list available snapshots and exit")
    parser.add_argument("--from", dest="snapshot", help="snapshot .jsonl file (in catalog_backups/ or a full path)")
    parser.add_argument("--apply", action="store_true", help="actually write the restore (default is dry-run)")
    parser.add_argument("--sample", type=int, default=20, help="rows to diff in dry-run mode (default 20)")
    args = parser.parse_args()

    if args.list or not args.snapshot:
        manifests = sorted(BACKUP_DIR.glob("ikea_furniture_*.manifest.json"))
        if not manifests:
            print(f"No snapshots in {BACKUP_DIR}. Run backup_catalog.py first.")
        for m in manifests:
            info = json.loads(m.read_text())
            print(f"  {info['file']:48s} {info['row_count']:6d} rows  taken {info['taken_at_utc']}")
        return

    snap_path = Path(args.snapshot)
    if not snap_path.is_absolute() and not snap_path.exists():
        snap_path = BACKUP_DIR / snap_path
    if not snap_path.exists():
        sys.exit(f"Snapshot not found: {snap_path}")

    print(f"Loading snapshot {snap_path} ...")
    rows = load_snapshot(snap_path)
    print(f"{len(rows)} rows in snapshot.")

    client = get_supabase_client()

    if not args.apply:
        print(f"\nDRY RUN — diffing {min(args.sample, len(rows))} sample rows against the live table (no writes):")
        sample_ids = [r["unique_id"] for r in rows[:: max(1, len(rows) // args.sample)][: args.sample]]
        live_by_id = {}
        for i in range(0, len(sample_ids), 50):
            chunk = sample_ids[i : i + 50]
            for row in client.table(TABLE_NAME).select("*").in_("unique_id", chunk).execute().data:
                live_by_id[row["unique_id"]] = row
        for snap_row in rows[:: max(1, len(rows) // args.sample)][: args.sample]:
            uid = snap_row["unique_id"]
            changed = diff_row(live_by_id.get(uid), snap_row)
            print(f"  {uid:16s} {'unchanged' if not changed else 'DIFFERS: ' + ', '.join(changed)}")
        print("\nNothing was written. Re-run with --apply to restore for real.")
        return

    print(f"\nAbout to overwrite {len(rows)} rows of '{TABLE_NAME}' with the snapshot's values.")
    confirm = input(f"Type the row count ({len(rows)}) to confirm: ").strip()
    if confirm != str(len(rows)):
        sys.exit("Row count didn't match — aborted, nothing written.")

    written = 0
    t0 = time.monotonic()
    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        for attempt in range(5):
            try:
                client.table(TABLE_NAME).upsert(batch, on_conflict="unique_id").execute()
                break
            except Exception as exc:
                print(f"  batch {i} failed ({type(exc).__name__}), retry {attempt + 1}/5")
                time.sleep(3 * (attempt + 1))
        else:
            sys.exit(f"Gave up on batch starting at row {i}; {written} rows restored so far.")
        written += len(batch)
        if written % 3000 == 0:
            print(f"  restored {written}/{len(rows)} ({time.monotonic() - t0:.0f}s)...")

    print(f"\nRestored {written} rows from {snap_path.name}.")


if __name__ == "__main__":
    main()
