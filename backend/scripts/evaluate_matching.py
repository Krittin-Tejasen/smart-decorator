"""Offline check of product matching on saved cutouts (backend/crops/).

Run from the backend/ directory:
    python -m scripts.evaluate_matching --out ../eval_out/results.json

For every cutout it runs the real matcher (CLIP -> Pinecone -> Supabase) twice
with the score threshold switched off, so the raw scores are visible:

  typed       what /match-products does for a label that maps to a furniture type
              (the search is limited to that type)
  image_only  the same image with no type filter, to show what the type filter
              keeps out

The JSON it writes holds, per cutout, the top results of both searches with
their scores, the type each result's title names, and the product image URLs
(used to build the report images). It reads the catalog and index only.

File names are `<room>_<n>_<label>.png`, e.g. room2_5_table_lamp.png.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from PIL import Image

from app.services.furniture_types import classify_title, label_to_type

CROPS_DIR = Path(__file__).resolve().parents[1] / "crops"
THRESHOLDS = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
_NAME_RE = re.compile(r"^(?P<room>[^_]+)_(?P<index>\d+)_(?P<label>.+)\.png$")


def parse_name(path: Path) -> tuple[str, str]:
    match = _NAME_RE.match(path.name)
    if not match:
        return path.stem, path.stem.replace("_", " ")
    return match["room"], match["label"].replace("_", " ")


def summarize(result, limit: int) -> list[dict]:
    return [
        {
            "score": round(candidate.score, 4),
            "title": candidate.row.get("main_title") or "",
            "type_by_title": classify_title(candidate.row.get("main_title")),
            "image_url": candidate.row.get("main_image"),
            "price": candidate.row.get("thb_price"),
            "width_cm": candidate.row.get("width_cm"),
            "url": candidate.row.get("url"),
        }
        for candidate in result.candidates[:limit]
    ]


def evaluate(paths: list[Path], top_k: int) -> list[dict]:
    from product_matching import find_candidates

    records = []
    for path in paths:
        room, label = parse_name(path)
        expected = label_to_type(label)
        image = Image.open(path)
        image.load()
        print(f"{path.name}: label={label!r} type={expected}")

        # min_score=0 keeps every candidate; top_k only trims what we report.
        typed = find_candidates(image, label, min_score=0.0, top_k=top_k)
        # "decoration" names no furniture type, so find_candidates searches by
        # image only, which is the untyped run we want.
        untyped = find_candidates(image, "decoration", min_score=0.0, top_k=top_k)

        typed_scores = [c.score for c in typed.candidates]
        records.append(
            {
                "file": path.name,
                "room": room,
                "label": label,
                "expected_type": expected,
                "typed_filter_level": typed.filter_level,
                "typed": summarize(typed, top_k),
                "image_only": summarize(untyped, top_k),
                "typed_top1": typed_scores[0] if typed_scores else None,
                "image_only_same_type": sum(
                    1 for c in untyped.candidates[:top_k] if expected and classify_title(c.row.get("main_title")) == expected
                ),
            }
        )
    return records


def threshold_table(records: list[dict]) -> dict[float, int]:
    """How many typed queries still have at least one result at each threshold."""
    return {
        t: sum(1 for r in records if r["typed_top1"] is not None and r["typed_top1"] >= t)
        for t in THRESHOLDS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=Path, default=CROPS_DIR)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--only", help="comma-separated substrings of file names to run")
    args = parser.parse_args()

    paths = sorted(args.crops.glob("*.png"))
    if args.only:
        wanted = [w.strip() for w in args.only.split(",")]
        paths = [p for p in paths if any(w in p.name for w in wanted)]
    if not paths:
        sys.exit("No cutouts found.")

    records = evaluate(paths, args.top_k)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps({"records": records, "queries_with_result": threshold_table(records)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {args.out}")
    print("Queries with at least one typed result, by threshold:")
    for t, n in threshold_table(records).items():
        print(f"  >= {t:.2f}: {n}/{len(records)}")


if __name__ == "__main__":
    main()
