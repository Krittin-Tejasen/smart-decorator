"""Add or change the `furniture_type` metadata field on the vectors already in
the Pinecone index (no re-embedding, no other field touched), and undo it.

Run from the backend/ directory:
    python -m scripts.add_furniture_type                  dry run: counts only, writes nothing
    python -m scripts.add_furniture_type --csv PATH       dry run from a catalog CSV instead of Supabase
    python -m scripts.add_furniture_type --sample 20      write 20 vectors, verify, stop
    python -m scripts.add_furniture_type --apply          write every typed vector (resumable)
    python -m scripts.add_furniture_type --verify         re-check the audit sample
    python -m scripts.add_furniture_type --rollback       dry run: what going back would do
    python -m scripts.add_furniture_type --rollback --apply   put every value back as it was

Before the first write the script saves the current `furniture_type` of every
vector it may touch (value, or null when the vector had none) to
.furniture_type_full_snapshot.json. It never overwrites that file, so a second
run cannot replace the "before" state with a half-written one. Delete the file
by hand only when you accept the current state as the new starting point.

Rollback puts the old value back with `update`. Pinecone's `update` cannot
delete a field, so a vector that had none is re-upserted without it (its vector
and its other metadata are fetched and written back unchanged).

Pinecone `update(id, set_metadata=...)` merges into the existing metadata.
"""

import csv
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from app.services.furniture_types import types_by_unique_id

HERE = Path(__file__).resolve().parent
WRITTEN = HERE / ".furniture_type_written.json"  # id -> value this script wrote
SNAPSHOT = HERE / ".furniture_type_full_snapshot.json"  # id -> value before any write, or null
FIELD = "furniture_type"
WORKERS = 8
AUDIT_SIZE = 300


def load_rows(csv_path: str | None) -> list[dict[str, Any]]:
    if csv_path:
        csv.field_size_limit(10_000_000)
        with open(csv_path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    from app.services.product_repo import fetch_all_products_light

    return fetch_all_products_light()


def fetch_vectors(index, ids: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for i in range(0, len(ids), 100):
        res = index.fetch(ids=ids[i : i + 100])
        out.update(res.vectors)
    return out


def fetch_meta(index, ids: list[str]) -> dict[str, dict[str, Any]]:
    return {key: dict(v.metadata or {}) for key, v in fetch_vectors(index, ids).items()}


def update_one(index, uid: str, value: str) -> bool:
    last: Exception | None = None
    for attempt in range(5):
        try:
            index.update(id=uid, set_metadata={FIELD: value})
            return True
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            last = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"  FAILED {uid}: {last}")
    return False


def load_written() -> dict[str, str]:
    if not WRITTEN.exists():
        return {}
    data = json.loads(WRITTEN.read_text())
    return data if isinstance(data, dict) else {}  # an old list-style checkpoint says nothing about values


def apply(index, targets: dict[str, str], ids: list[str]) -> int:
    """Write the ids whose value in Pinecone is not already the target. Progress
    records the value written, so a changed target is written again."""
    written = load_written()
    todo = [uid for uid in ids if written.get(uid) != targets[uid]]
    print(f"Writing {len(todo)} vectors ({len(ids) - len(todo)} already at the target value).")
    n_ok = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(todo), 500):
            chunk = todo[start : start + 500]
            for uid, ok in zip(chunk, pool.map(lambda u: update_one(index, u, targets[u]), chunk)):
                if ok:
                    written[uid] = targets[uid]
                    n_ok += 1
            WRITTEN.write_text(json.dumps(written))
            print(f"  {min(start + 500, len(todo))}/{len(todo)} (ok so far {n_ok})")
    return n_ok


def verify(index, targets: dict[str, str], ids: list[str], before: dict[str, dict[str, Any]] | None = None) -> bool:
    after = fetch_meta(index, ids)
    missing = [u for u in ids if u not in after]
    wrong = [u for u in ids if u in after and after[u].get(FIELD) != targets[u]]
    changed: list[str] = []
    if before:
        changed = [
            u
            for u in ids
            if u in after
            and u in before
            and {k: v for k, v in after[u].items() if k != FIELD} != {k: v for k, v in before[u].items() if k != FIELD}
        ]
    print(
        f"Verified {len(ids)} ids: not in index={len(missing)}, wrong/absent {FIELD}={len(wrong)}, "
        f"other fields changed={len(changed)}"
    )
    return not (wrong or changed)


def take_snapshot(index, ids: list[str]) -> dict[str, str | None]:
    """The current value of the field for every id, saved once and never replaced."""
    if SNAPSHOT.exists():
        return json.loads(SNAPSHOT.read_text())
    print(f"Saving the current {FIELD} of {len(ids)} vectors before the first write...")
    current = fetch_meta(index, ids)
    snapshot = {uid: current[uid].get(FIELD) for uid in ids if uid in current}
    SNAPSHOT.write_text(json.dumps(snapshot))
    had = sum(1 for v in snapshot.values() if v is not None)
    print(f"Saved {SNAPSHOT.name}: {len(snapshot)} vectors, {had} with a value, {len(snapshot) - had} with none.")
    return snapshot


def remove_field(index, ids: list[str]) -> None:
    """`update` cannot delete a field, so these vectors are written back whole
    (vector and other metadata unchanged), without it."""
    for start in range(0, len(ids), 100):
        batch = fetch_vectors(index, ids[start : start + 100])
        index.upsert(
            vectors=[
                {"id": key, "values": v.values, "metadata": {k: m for k, m in (v.metadata or {}).items() if k != FIELD}}
                for key, v in batch.items()
            ]
        )
        print(f"  removed {min(start + 100, len(ids))}/{len(ids)}")


def rollback(index, apply_it: bool) -> None:
    if not SNAPSHOT.exists():
        sys.exit(f"No {SNAPSHOT.name}: there is no saved 'before' state to go back to.")
    snapshot: dict[str, str | None] = json.loads(SNAPSHOT.read_text())
    current = fetch_meta(index, sorted(snapshot))
    to_restore = {u: v for u, v in snapshot.items() if v is not None and current.get(u, {}).get(FIELD) != v}
    to_remove = [u for u, v in snapshot.items() if v is None and FIELD in current.get(u, {})]
    print(f"Rollback: restore the old value on {len(to_restore)} vectors, remove the field from {len(to_remove)}.")
    if not apply_it:
        print("Dry run: nothing written. Add --apply to do it.")
        return

    ids = sorted(to_restore)
    with ThreadPoolExecutor(WORKERS) as pool:
        for start in range(0, len(ids), 500):
            chunk = ids[start : start + 500]
            list(pool.map(lambda u: update_one(index, u, to_restore[u]), chunk))
            print(f"  restored {min(start + 500, len(ids))}/{len(ids)}")

    remove_field(index, to_remove)

    time.sleep(5)
    after = fetch_meta(index, sorted(snapshot))
    bad = [u for u, v in snapshot.items() if u in after and after[u].get(FIELD) != v]
    print("Rollback verified." if not bad else f"Rollback left {len(bad)} vectors different from the snapshot.")
    WRITTEN.unlink(missing_ok=True)


def arg_value(flag: str) -> str | None:
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else None


def main() -> None:
    if "--rollback" in sys.argv:
        from app.services.pinecone_client import get_index

        rollback(get_index(), "--apply" in sys.argv)
        return

    rows = load_rows(arg_value("--csv"))
    targets = types_by_unique_id(rows)
    by_type = Counter(targets.values())
    print(f"Catalog rows: {len(rows)}; typed: {len(targets)} ({len(targets) / max(len(rows), 1):.1%})")
    for name, count in by_type.most_common():
        print(f"  {name:<20}{count:>7}")

    writes = "--apply" in sys.argv or "--sample" in sys.argv
    if not writes and "--verify" not in sys.argv:
        print("Dry run: nothing written. Use --sample N first, then --apply.")
        return
    if "--csv" in sys.argv:
        sys.exit("--csv is for dry runs only; writes use the Supabase rows that were embedded.")

    from app.services.pinecone_client import get_index

    index = get_index()
    ids_all = sorted(targets)
    audit = random.Random(1).sample(ids_all, min(AUDIT_SIZE, len(ids_all)))

    if "--verify" in sys.argv:
        sys.exit(0 if verify(index, targets, audit) else 1)

    sample_n = arg_value("--sample")
    ids = random.Random(0).sample(ids_all, int(sample_n)) if sample_n else ids_all

    # The 'before' state of every id this script could ever write, taken once.
    take_snapshot(index, ids_all)
    check = sorted(set(ids) if sample_n else set(audit) | set(ids[:50]))
    pre = fetch_meta(index, check)  # to prove no other field changes

    apply(index, targets, ids)
    # Vectors this script wrote earlier that no longer get a type (the rules
    # changed): take the field off rather than leave a stale value.
    stale = sorted(u for u in load_written() if u not in targets) if not sample_n else []
    if stale:
        print(f"{len(stale)} vectors written earlier no longer have a type; removing the field from them.")
        remove_field(index, stale)
        written = load_written()
        for u in stale:
            written.pop(u, None)
        WRITTEN.write_text(json.dumps(written))
    time.sleep(5)  # serverless indexes update metadata with a short delay
    print("OK" if verify(index, targets, check, pre) else "VERIFICATION FAILED - stop and investigate")


if __name__ == "__main__":
    main()
