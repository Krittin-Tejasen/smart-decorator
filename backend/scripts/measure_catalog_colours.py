"""Measure each catalog PRODUCT's own colour from its photo (catalog_colour.py),
instead of trusting `color_extracted`. Part of the colour clean-up plan.

DRY RUN ONLY: reads from a local backup_catalog.py snapshot, downloads photos to
a local cache, and writes results to a local file. Nothing is written to
Supabase or Pinecone — that is a separate, later step this script does not do.

Run from backend/:
    python -m scripts.measure_catalog_colours                 # the whole catalog, resumable
    python -m scripts.measure_catalog_colours --sample 500     # a random subset, for a first look
    python -m scripts.measure_catalog_colours --from ikea_furniture_<ts>.jsonl

One row of main_image/sku defines a "product" (furniture_types.image_group_key
— the same key /match-products groups by): every row sharing a photo gets the
same measurement, so a 59,846-row table only needs ~23,168 photos measured.

Output: catalog_backups/measured_colours_<UTC timestamp>.jsonl, one JSON object
per product group: {group_key, unique_ids, representative_title,
old_colors_seen, measured: {swatches, foreground_share, low_confidence}}.
Re-running without --sample skips groups already present in the newest such
file, so an interrupted run resumes instead of re-downloading everything.
"""

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.furniture_types import image_group_key
from catalog_colour import measure_product_colour

BACKUP_DIR = Path(__file__).resolve().parent.parent / "catalog_backups"
IMAGE_CACHE = BACKUP_DIR / "product_images"
DOWNLOAD_WORKERS = 12
EN_COUNTRIES = {"gb", "us", "ie", "ca", "au", "nz", "sg", "my", "in", "ae", "sa", "qa", "kw", "om", "bh", "jo", "eg", "ph"}


def latest_catalog_backup() -> Path:
    files = sorted(BACKUP_DIR.glob("ikea_furniture_*.jsonl"))
    if not files:
        sys.exit(f"No catalog backup found in {BACKUP_DIR}. Run `python -m scripts.backup_catalog` first.")
    return files[-1]


def load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_groups(rows: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        if not row.get("unique_id") or not row.get("main_image"):
            continue
        groups.setdefault(image_group_key(row.get("main_image"), row.get("sku")), []).append(row)
    return groups


def representative(group: list[dict]) -> dict:
    english = [r for r in group if (r.get("country_code") or "").lower() in EN_COUNTRIES]
    return (english or group)[0]


def cache_path(group_key: str) -> Path:
    import re

    return IMAGE_CACHE / (re.sub(r"[^A-Za-z0-9_.-]", "_", group_key) + ".jpg")


def download(group_key: str, url: str) -> Path | None:
    dest = cache_path(group_key)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        response = httpx.get(url, timeout=20, follow_redirects=True)
        response.raise_for_status()
        dest.write_bytes(response.content)
        return dest
    except Exception as exc:
        print(f"  download failed for {group_key}: {type(exc).__name__}: {exc}")
        return None


def already_done(latest_output: Path | None) -> set[str]:
    if not latest_output or not latest_output.exists():
        return set()
    done = set()
    with latest_output.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                done.add(json.loads(line)["group_key"])
    return done


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="snapshot", help="catalog backup .jsonl (default: the newest in catalog_backups/)")
    parser.add_argument("--sample", type=int, default=0, help="process a random N-group subset instead of everything")
    parser.add_argument("--seed", type=int, default=7, help="--sample's random seed, for a reproducible subset")
    parser.add_argument("--limit", type=int, default=0, help="stop after this many NEW groups this run (0 = no cap)")
    parser.add_argument("--resume-from", dest="resume_from", help="skip group_keys already in this earlier output file")
    args = parser.parse_args()

    BACKUP_DIR.mkdir(exist_ok=True)
    IMAGE_CACHE.mkdir(exist_ok=True)

    snapshot_path = Path(args.snapshot) if args.snapshot else latest_catalog_backup()
    if not snapshot_path.is_absolute() and not snapshot_path.exists():
        snapshot_path = BACKUP_DIR / snapshot_path
    print(f"Reading {snapshot_path} ...")
    rows = load_rows(snapshot_path)
    groups = build_groups(rows)
    print(f"{len(rows)} rows -> {len(groups)} distinct products (by photo).")

    keys = sorted(groups)
    if args.sample:
        random.Random(args.seed).shuffle(keys)
        keys = sorted(keys[: args.sample])  # sorted again so a resumed/sampled run is still legible in order
        skip: set[str] = set()
    else:
        resume_path = Path(args.resume_from) if args.resume_from else None
        if resume_path and not resume_path.is_absolute():
            resume_path = BACKUP_DIR / resume_path
        skip = already_done(resume_path or _latest_output())
        if skip:
            print(f"Resuming: {len(skip)} products already measured in a previous run, skipping those.")

    todo = [k for k in keys if k not in skip]
    if args.limit:
        todo = todo[: args.limit]
    print(f"Measuring {len(todo)} product(s)...")

    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = BACKUP_DIR / f"measured_colours_{timestamp}.jsonl"

    # carry over anything already measured in a prior run, so this output file
    # is always a complete, growing record (not just this invocation's slice)
    carried = 0
    with out_path.open("w", encoding="utf-8") as out:
        source = (args.resume_from and (Path(args.resume_from) if Path(args.resume_from).is_absolute() else BACKUP_DIR / args.resume_from)) or _latest_output()
        if not args.sample and source and source.exists():
            with source.open(encoding="utf-8") as prev:
                for line in prev:
                    if line.strip():
                        row = json.loads(line)
                        if row["group_key"] in skip:
                            out.write(line if line.endswith("\n") else line + "\n")
                            carried += 1
        if carried:
            print(f"Carried over {carried} previously measured product(s) into the new output file.")

        low_conf = 0
        agree = disagree = no_old_colour = 0
        t0 = time.monotonic()

        def fetch_one(key):
            rep = representative(groups[key])
            return key, download(key, rep["main_image"])

        with ThreadPoolExecutor(DOWNLOAD_WORKERS) as pool:
            futures = {pool.submit(fetch_one, k): k for k in todo}
            for n, future in enumerate(as_completed(futures), 1):
                key, path = future.result()
                group = groups[key]
                rep = representative(group)
                if path is None:
                    continue
                try:
                    with Image.open(path) as im:
                        im = im.convert("RGB")
                        im.thumbnail((400, 400))
                        result = measure_product_colour(im)
                except Exception as exc:
                    print(f"  measuring {key} failed: {type(exc).__name__}: {exc}")
                    continue

                old_colors = sorted({r["color_extracted"] for r in group if r.get("color_extracted")})
                if result.low_confidence:
                    low_conf += 1
                elif not old_colors:
                    no_old_colour += 1
                elif result.swatches[0].catalog_color in old_colors:
                    agree += 1
                else:
                    disagree += 1

                record = {
                    "group_key": key,
                    "unique_ids": sorted(r["unique_id"] for r in group),
                    "representative_title": rep.get("main_title"),
                    "old_colors_seen": old_colors,
                    "measured": {
                        "swatches": [s.model_dump() for s in result.swatches],
                        "foreground_share": round(result.foreground_share, 3),
                        "low_confidence": result.low_confidence,
                    },
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                if n % 200 == 0:
                    out.flush()
                    print(f"  {n}/{len(todo)} ({time.monotonic() - t0:.0f}s)...")

    measured = len(todo) - (len(todo) - (agree + disagree + no_old_colour + low_conf))
    total_considered = agree + disagree + no_old_colour + low_conf
    print(f"\nDone: {total_considered} measured this run -> {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  low_confidence (kept the old colour): {low_conf}")
    if agree + disagree:
        print(f"  measured top colour agrees with the catalog's color_extracted: {agree}/{agree + disagree} ({agree/(agree+disagree):.0%})")
    print(f"  measured but the catalog had no color_extracted at all: {no_old_colour}")


def _latest_output() -> Path | None:
    files = sorted(BACKUP_DIR.glob("measured_colours_*.jsonl"))
    return files[-1] if files else None


if __name__ == "__main__":
    main()
