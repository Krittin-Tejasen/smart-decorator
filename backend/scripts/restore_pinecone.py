"""Roll back Pinecone vector metadata to a snapshot taken by backup_pinecone.py.

Run from backend/:
    python -m scripts.restore_pinecone --list
    python -m scripts.restore_pinecone --from <file>.jsonl --dry-run
    python -m scripts.restore_pinecone --from <file>.jsonl --apply

Important: Pinecone's update(set_metadata=...) MERGES the given fields into a
vector's existing metadata — it does not replace the whole object. So this
restores every field the snapshot had back to its old value, but a field a
later migration ADDED (that didn't exist in the snapshot) is NOT removed by
this script. Pass --also-clear-keys key1,key2 to explicitly blank those keys
(set them to "") on every restored vector — use it with the exact field names
the migration you're undoing introduced.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from app.services.pinecone_client import get_index

BACKUP_DIR = Path(__file__).resolve().parent.parent / "catalog_backups"
BATCH_SIZE = 1  # Pinecone's update() takes one vector's metadata per call


def load_snapshot(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--from", dest="snapshot")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--also-clear-keys", default="", help="comma-separated metadata keys to blank out too")
    parser.add_argument("--sample", type=int, default=10)
    args = parser.parse_args()

    if args.list or not args.snapshot:
        manifests = sorted(BACKUP_DIR.glob("pinecone_metadata_*.manifest.json"))
        if not manifests:
            print(f"No Pinecone snapshots in {BACKUP_DIR}. Run backup_pinecone.py first.")
        for m in manifests:
            info = json.loads(m.read_text())
            print(f"  {info['file']:48s} {info['vectors_backed_up']:6d} vectors  taken {info['taken_at_utc']}")
        return

    snap_path = Path(args.snapshot)
    if not snap_path.is_absolute() and not snap_path.exists():
        snap_path = BACKUP_DIR / snap_path
    if not snap_path.exists():
        sys.exit(f"Snapshot not found: {snap_path}")

    rows = load_snapshot(snap_path)
    print(f"{len(rows)} vectors in snapshot {snap_path.name}.")
    clear_keys = [k.strip() for k in args.also_clear_keys.split(",") if k.strip()]
    if clear_keys:
        print(f"Will also blank these keys on every restored vector: {clear_keys}")

    index = get_index()

    if not args.apply:
        print(f"\nDRY RUN — showing {min(args.sample, len(rows))} sample vectors' metadata that WOULD be restored:")
        for row in rows[:: max(1, len(rows) // args.sample)][: args.sample]:
            print(f"  {row['id']}: {row['metadata']}")
        print("\nNothing was written. Re-run with --apply to restore for real.")
        return

    print(f"\nAbout to restore metadata on {len(rows)} Pinecone vectors.")
    confirm = input(f"Type the vector count ({len(rows)}) to confirm: ").strip()
    if confirm != str(len(rows)):
        sys.exit("Count didn't match — aborted, nothing written.")

    written = 0
    t0 = time.monotonic()
    for row in rows:
        metadata = dict(row["metadata"])
        for key in clear_keys:
            metadata[key] = ""
        for attempt in range(5):
            try:
                index.update(id=row["id"], set_metadata=metadata)
                break
            except Exception as exc:
                print(f"  {row['id']} failed ({type(exc).__name__}), retry {attempt + 1}/5")
                time.sleep(2 * (attempt + 1))
        else:
            sys.exit(f"Gave up on vector {row['id']}; {written} restored so far.")
        written += 1
        if written % 2000 == 0:
            print(f"  restored {written}/{len(rows)} ({time.monotonic() - t0:.0f}s)...")

    print(f"\nRestored metadata on {written} vectors from {snap_path.name}.")


if __name__ == "__main__":
    main()
