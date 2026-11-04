"""Offline job: build the Pinecone index for the rebuilt catalog (ikea_furniture_v2).

Run from the backend/ directory:
    python -m scripts.ingest_catalog_v2                                   dry run: the plan, writes nothing
    python -m scripts.ingest_catalog_v2 --apply                           create the index and ingest
    python -m scripts.ingest_catalog_v2 --apply --cloud gcp --region us-central1

One vector per PRODUCT, not per row. The same product on every country's site
shares one picture (group_key), so its rows would embed to the same vector;
matching already collapses them into one card. The vector's id is the
unique_id of the product's representative row (the English listing when there
is one), so the id still finds that row in ikea_furniture_v2.

A product the old index already holds is not embedded again: its vector is
copied from the old index (same picture, same model). Only products that are
new in the rebuilt catalog are downloaded and encoded.

It never touches the old index or the old table. Switch the app over afterwards
by setting, in backend/.env,
    PINECONE_INDEX_NAME=<the new index>   CATALOG_TABLE=ikea_furniture_v2
Progress is checkpointed, so a stopped run resumes.
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.services.embedding import download_image, encode_images
from app.services.furniture_types import image_group_key, subtypes_by_unique_id
from app.services.pinecone_client import get_pinecone_client
from app.services.product_repo import get_supabase_client
from scripts.ingest_products import BATCH_SIZE, build_metadata, download_batch

NEW_TABLE = "ikea_furniture_v2"
OLD_TABLE = "ikea_furniture"
DEFAULT_INDEX = "ikea-products-v2"
CHECKPOINT = Path(__file__).with_name(".ingest_v2_checkpoint.json")
PAGE = 1000
FETCH_BATCH = 100
UPSERT_BATCH = 100
# The English listing is the one a user should see first (same rule as product_matching).
PREFERRED_COUNTRIES = ("gb", "us")

NEW_COLUMNS = (
    "unique_id,sku,group_key,main_title,main_image,url,category,country_code,furniture_type,"
    "thb_price,in_stock,width_cm,height_cm,depth_cm,size_label,material_tags,material_tag,"
    "is_wood,is_metal,is_fabric,is_plastic,is_glass,color_extracted"
)


# ── Pure helpers (tested in tests/test_ingest_v2.py) ───────────────────────

def pick_representative(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The row that stands for a product: the English listing first (gb, us),
    then the smallest unique_id. Only rows with a picture can be embedded."""
    candidates = [r for r in rows if r.get("main_image")]
    if not candidates:
        return None

    def rank(row: dict[str, Any]) -> tuple[int, str]:
        country = (row.get("country_code") or "").lower()
        order = PREFERRED_COUNTRIES.index(country) if country in PREFERRED_COUNTRIES else len(PREFERRED_COUNTRIES)
        return order, str(row["unique_id"])

    return min(candidates, key=rank)


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["group_key"], []).append(row)
    return groups


def product_metadata(row: dict[str, Any], subtype: str | None) -> dict[str, Any]:
    """Pinecone metadata of a product: the same fields ingest_products writes,
    plus group_key and country_code (the listing the vector stands for)."""
    metadata = build_metadata(row, row.get("furniture_type"), subtype)
    metadata["group_key"] = row.get("group_key") or ""
    metadata["country_code"] = row.get("country_code") or ""
    return metadata


def plan_vectors(
    groups: dict[str, list[dict[str, Any]]],
    old_ids_by_group: dict[str, str],
    done: set[str],
) -> tuple[list[tuple[dict[str, Any], str | None]], int]:
    """(representative row, old index id to copy the vector from or None) for every
    product not yet ingested, and how many products have no usable picture."""
    plan: list[tuple[dict[str, Any], str | None]] = []
    no_picture = 0
    for key in sorted(groups):
        representative = pick_representative(groups[key])
        if representative is None:
            no_picture += 1
            continue
        if str(representative["unique_id"]) in done:
            continue
        plan.append((representative, old_ids_by_group.get(key)))
    return plan, no_picture


# ── Reading ────────────────────────────────────────────────────────────────

def fetch_rows(table: str, columns: str, retries: int = 5) -> list[dict[str, Any]]:
    client = get_supabase_client()
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        for attempt in range(retries):
            try:
                batch = (
                    client.table(table).select(columns).order("unique_id").range(start, start + PAGE - 1).execute().data
                )
                break
            except Exception:  # noqa: BLE001 - network drop: retried, then raised
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        start += PAGE


def old_ids_by_group() -> dict[str, str]:
    """group_key -> an id of the OLD index, for every product the old table has.
    The key is the picture's id, the same function that built group_key."""
    mapping: dict[str, str] = {}
    for row in fetch_rows(OLD_TABLE, "unique_id,sku,main_image"):
        mapping.setdefault(image_group_key(row.get("main_image"), row.get("sku")), str(row["unique_id"]))
    return mapping


def fetch_old_vectors(old_index, ids: list[str]) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for start in range(0, len(ids), FETCH_BATCH):
        response = old_index.fetch(ids=ids[start : start + FETCH_BATCH])
        for vector_id, vector in response.vectors.items():
            found[vector_id] = list(vector.values)
    return found


# ── Index ──────────────────────────────────────────────────────────────────

def ensure_index(name: str, cloud: str, region: str):
    pc = get_pinecone_client()
    if name not in pc.list_indexes().names():
        from pinecone import ServerlessSpec

        print(f"Creating index {name} ({cloud}/{region}, {settings.embedding_dimension} dims, cosine)...")
        pc.create_index(
            name=name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(2)
    return pc.Index(name)


def load_checkpoint() -> set[str]:
    return set(json.loads(CHECKPOINT.read_text())) if CHECKPOINT.exists() else set()


def save_checkpoint(done: set[str]) -> None:
    CHECKPOINT.write_text(json.dumps(sorted(done)))


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index", default=DEFAULT_INDEX, help=f"new index name (default {DEFAULT_INDEX})")
    parser.add_argument("--cloud", default="aws", choices=["aws", "gcp", "azure"])
    parser.add_argument("--region", default="us-east-1", help="e.g. us-east-1 (aws), us-central1 (gcp)")
    parser.add_argument("--no-reuse", action="store_true", help="embed every product, do not copy old vectors")
    parser.add_argument("--apply", action="store_true", help="create the index and write (default: dry run)")
    args = parser.parse_args()

    if args.index == settings.pinecone_index_name:
        sys.exit(f"--index {args.index} is the index the app uses now. Pick a new name.")

    print(f"Reading {NEW_TABLE}...")
    rows = fetch_rows(NEW_TABLE, NEW_COLUMNS)
    groups = group_rows(rows)
    types = {str(r["unique_id"]): r["furniture_type"] for r in rows}
    subtypes = subtypes_by_unique_id(rows, types)
    print(f"{len(rows)} rows = {len(groups)} products.")

    reusable = {} if args.no_reuse else old_ids_by_group()
    done = load_checkpoint()
    plan, no_picture = plan_vectors(groups, reusable, done)
    copy = [(row, old) for row, old in plan if old]
    embed = [row for row, old in plan if not old]
    print(
        f"Plan: {len(plan)} vectors to write ({len(done)} already done). "
        f"Copy from the old index: {len(copy)}. Download and encode: {len(embed)}. "
        f"Products without a picture: {no_picture}."
    )
    pc = get_pinecone_client()
    exists = args.index in pc.list_indexes().names()
    print(f"Index {args.index}: {'exists' if exists else f'will be created on {args.cloud}/{args.region}'}.")
    if not args.apply:
        print("Dry run: nothing written. Add --apply to create the index and ingest.")
        return

    index = ensure_index(args.index, args.cloud, args.region)
    old_index = pc.Index(settings.pinecone_index_name)

    def metadata_of(row: dict[str, Any]) -> dict[str, Any]:
        return product_metadata(row, subtypes.get(str(row["unique_id"])))

    # 1. Products the old index already has: copy the vector.
    missing_in_old: list[dict[str, Any]] = []
    for start in range(0, len(copy), UPSERT_BATCH):
        chunk = copy[start : start + UPSERT_BATCH]
        vectors = fetch_old_vectors(old_index, [old for _, old in chunk])
        records = []
        for row, old in chunk:
            if old in vectors:
                records.append({"id": str(row["unique_id"]), "values": vectors[old], "metadata": metadata_of(row)})
            else:
                missing_in_old.append(row)  # the old index never got this picture: encode it
        if records:
            index.upsert(vectors=records)
            done.update(r["id"] for r in records)
            save_checkpoint(done)
        print(f"  copied {min(start + UPSERT_BATCH, len(copy))}/{len(copy)}")

    # 2. New products (and any the old index lacked): download and encode.
    todo = embed + missing_in_old
    for start in range(0, len(todo), BATCH_SIZE):
        chunk = todo[start : start + BATCH_SIZE]
        try:
            pairs = download_batch(chunk)
            if not pairs:
                continue
            vectors = encode_images([image for _, image in pairs], batch_size=BATCH_SIZE)
            index.upsert(
                vectors=[
                    {"id": str(row["unique_id"]), "values": vector, "metadata": metadata_of(row)}
                    for (row, _), vector in zip(pairs, vectors)
                ]
            )
            done.update(str(row["unique_id"]) for row, _ in pairs)
            save_checkpoint(done)
            print(f"  encoded {min(start + BATCH_SIZE, len(todo))}/{len(todo)}")
        except Exception as exc:  # noqa: BLE001 - a bad batch must not end an hours-long job
            print(f"Batch at {start} failed, will retry on the next run: {type(exc).__name__}: {exc}")
        finally:
            gc.collect()

    stats = index.describe_index_stats()
    print(f"Done. {args.index} holds {stats.total_vector_count} vectors (expected about {len(groups) - no_picture}).")


if __name__ == "__main__":
    main()
