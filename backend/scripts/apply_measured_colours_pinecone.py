"""Add measured_color / measured_color_low_confidence to each vector's
metadata in the Pinecone index named by PINECONE_INDEX_NAME (ikea-products-v2),
alongside the existing fields. Only ever calls index.update(..., set_metadata=
...) to ADD these two keys — never touches the existing `color` key, never
re-upserts or deletes a vector, so the vectors/embeddings themselves are
never at risk.

Run from backend/:
    python -m scripts.apply_measured_colours_pinecone --list                      # show candidate files
    python -m scripts.apply_measured_colours_pinecone --from <file>.jsonl          # dry-run (default), no writes
    python -m scripts.apply_measured_colours_pinecone --from <file>.jsonl --apply  # actually write

Input is a measure_catalog_colours.py output file (group_key -> measured
colour). To know which vector id represents which group_key, this reads the
*local* backup_pinecone.py snapshot (pinecone_metadata_<index>_*.jsonl) for
its id -> metadata.group_key map — no extra live query needed, and it means
a stale/missing local snapshot fails loudly instead of silently updating the
wrong (or no) vectors. Re-run backup_pinecone.py first if that snapshot is
older than the measured-colours file.

Rollback: re-run with --apply and a measured-colours file that has no match
for a vector to leave it untouched, or just stop reading these two keys in
the matcher (they're additive, nothing else changes); to physically remove
them, index.update(id=..., set_metadata={"measured_color": None,
"measured_color_low_confidence": None}) clears a metadata key in Pinecone.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.pinecone_client import get_index

BACKUP_DIR = Path(__file__).resolve().parent.parent / "catalog_backups"
UPDATE_BATCH = 1  # Pinecone's update() is per-vector, no batch endpoint
RETRIES = 5


def load_measured(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def latest_pinecone_snapshot(index_name: str) -> Path | None:
    files = sorted(BACKUP_DIR.glob(f"pinecone_metadata_{index_name}_*.jsonl"))
    return files[-1] if files else None


def build_group_key_to_color(groups: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out = {}
    for g in groups:
        m = g["measured"]
        low_conf = bool(m["low_confidence"])
        swatches = m["swatches"]
        color = None if low_conf or not swatches else swatches[0]["catalog_color"]
        out[g["group_key"]] = {"measured_color": color, "measured_color_low_confidence": low_conf}
    return out


def update_with_retries(index, vector_id: str, set_metadata: dict) -> None:
    # Pinecone metadata values must be string/number/boolean/list-of-strings —
    # None is rejected outright (not even as a way to unset a key), so a
    # low_confidence group's absent colour is sent by omitting the key
    # entirely rather than as null.
    set_metadata = {k: v for k, v in set_metadata.items() if v is not None}
    last_exc = None
    for attempt in range(RETRIES):
        try:
            index.update(id=vector_id, set_metadata=set_metadata)
            return
        except Exception as exc:
            last_exc = exc
            wait = 3 * (attempt + 1)
            print(f"  {vector_id} failed ({type(exc).__name__}), retry {attempt + 1}/{RETRIES} in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {vector_id} after {RETRIES} attempts") from last_exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="list measured_colours_*.jsonl files and exit")
    parser.add_argument("--from", dest="source", help="measure_catalog_colours.py output .jsonl (in catalog_backups/ or a full path)")
    parser.add_argument("--apply", action="store_true", help="actually write (default is dry-run)")
    parser.add_argument("--sample", type=int, default=20, help="vectors to preview in dry-run mode (default 20)")
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

    index_name = settings.pinecone_index_name
    snap_path = latest_pinecone_snapshot(index_name)
    if snap_path is None:
        sys.exit(f"No local pinecone_metadata_{index_name}_*.jsonl snapshot in {BACKUP_DIR}. "
                  f"Run `python -m scripts.backup_pinecone` first (needed for the id -> group_key map).")

    print(f"Loading {src_path} ...")
    groups = load_measured(src_path)
    by_group_key = build_group_key_to_color(groups)

    print(f"Loading vector id -> group_key map from {snap_path} ...")
    vector_updates: list[tuple[str, dict]] = []
    unmatched = 0
    with snap_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            group_key = rec["metadata"].get("group_key")
            colour = by_group_key.get(group_key)
            if colour is None:
                unmatched += 1
                continue
            vector_updates.append((rec["id"], colour))

    n_colored = sum(1 for _, c in vector_updates if c["measured_color"] is not None)
    print(f"{len(vector_updates)} vectors in '{index_name}' will get measured_color metadata "
          f"({n_colored} with a colour, {len(vector_updates) - n_colored} low_confidence/null).")
    if unmatched:
        print(f"  {unmatched} vectors in the snapshot had no matching group in {src_path.name} (left untouched).")

    if not args.apply:
        print(f"\nDRY RUN — {min(args.sample, len(vector_updates))} sample updates that would be sent (no writes):")
        sample = vector_updates[:: max(1, len(vector_updates) // args.sample)][: args.sample]
        for vector_id, colour in sample:
            print(f"  {vector_id:20s} set_metadata={colour}")
        print("\nNothing was written. Re-run with --apply to write for real.")
        return

    print(f"\nAbout to call index.update(set_metadata=...) for {len(vector_updates)} vectors in '{index_name}'.")
    confirm = input(f"Type the vector count ({len(vector_updates)}) to confirm: ").strip()
    if confirm != str(len(vector_updates)):
        sys.exit("Count didn't match — aborted, nothing written.")

    index = get_index()
    written = 0
    t0 = time.monotonic()
    for vector_id, colour in vector_updates:
        update_with_retries(index, vector_id, colour)
        written += 1
        if written % 500 == 0:
            print(f"  updated {written}/{len(vector_updates)} ({time.monotonic() - t0:.0f}s)...")

    print(f"\nUpdated {written} vectors' metadata from {src_path.name}.")


if __name__ == "__main__":
    main()
