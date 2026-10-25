"""Read-only check: can CLIP name a catalog product's furniture type from its
stored image vector?

Run from the backend/ directory:
    python -m scripts.zero_shot_types --out ../eval_out/zero_shot.json

The catalog vectors already in Pinecone are compared with text prompts
("a photo of a lamp"), so no image is downloaded or re-embedded. Nothing is
written to Pinecone or Supabase.

Two samples are scored:
  labeled    rows that already have a furniture_type (from the title patterns).
             How often CLIP agrees with it is the accuracy figure.
  unlabeled  rows with no type. CLIP's guess and its confidence are listed so
             a confidence cut-off can be chosen by eye.
"""

import argparse
import json
import random
import sys

import numpy as np

from app.services.embedding import get_model
from app.services.furniture_types import FURNITURE_TYPES, types_by_unique_id
from app.services.pinecone_client import get_index
import time

from app.services.product_repo import CARD_COLUMNS, TABLE_NAME, get_supabase_client

# What each type is called in a photo caption. "other" classes soak up parts
# and non-furniture so those rows are not forced into a furniture type.
PROMPTS: dict[str, list[str]] = {
    "tv_bench": ["a TV bench", "a TV stand"],
    "coffee_table": ["a coffee table"],
    "nightstand": ["a bedside table", "a nightstand"],
    "armchair": ["an armchair"],
    "stool": ["a stool"],
    "sofa": ["a sofa", "a couch"],
    "bed": ["a bed frame", "a bed"],
    "dining_table": ["a dining table"],
    "chair": ["a chair", "a dining chair"],
    "bookshelf": ["a bookcase", "a shelf unit"],
    "lamp": ["a lamp", "a ceiling light", "a chandelier", "a floor lamp"],
    "wardrobe": ["a wardrobe", "a closet"],
    "desk": ["a desk"],
    "side_table": ["a side table"],
    "dresser": ["a chest of drawers", "a dresser"],
    "cabinet": ["a cabinet", "a sideboard"],
    "ottoman": ["an ottoman", "a pouf"],
    "bench": ["a bench"],
    "rug": ["a rug", "a carpet"],
    "curtain": ["a curtain"],
    "mirror": ["a mirror"],
    "plant_pot": ["a plant pot", "a flower pot"],
    "vase": ["a vase"],
    "artwork": ["a framed picture", "a wall art print", "a painting"],
}
OTHER_PROMPTS = {
    "other:kitchen_part": ["a kitchen cabinet door", "a kitchen drawer front", "a kitchen worktop"],
    "other:bathroom": ["a bathroom sink vanity", "a faucet", "a bathtub"],
    "other:textile": ["a blanket", "a pillow", "a duvet cover", "a towel"],
    "other:small_item": ["a kitchen utensil", "a candle", "a storage box", "a picture frame"],
    "other:mattress": ["a mattress"],
}


def fetch_light_catalog() -> list[dict]:
    """Every row with only the card columns (select * drops the connection on
    the long text columns), retrying a page that fails."""
    client = get_supabase_client()
    rows, start, page = [], 0, 1000
    while True:
        for attempt in range(5):
            try:
                batch = (
                    client.table(TABLE_NAME).select(CARD_COLUMNS).order("unique_id")
                    .range(start, start + page - 1).execute().data
                )
                break
            except Exception as exc:
                print(f"retry {attempt + 1} at {start}: {type(exc).__name__}")
                time.sleep(2)
        else:
            raise RuntimeError(f"catalog page at {start} kept failing")
        rows.extend(batch)
        if len(batch) < page:
            return rows
        start += page


def build_text_matrix() -> tuple[list[str], np.ndarray]:
    model = get_model()
    classes, vectors = [], []
    for name, texts in {**PROMPTS, **OTHER_PROMPTS}.items():
        embedded = model.encode([f"a photo of {t}, product photo on a white background" for t in texts])
        embedded = embedded / np.linalg.norm(embedded, axis=1, keepdims=True)
        mean = embedded.mean(axis=0)
        classes.append(name)
        vectors.append(mean / np.linalg.norm(mean))
    return classes, np.stack(vectors)


def fetch_vectors(ids: list[str]) -> dict[str, np.ndarray]:
    index = get_index()
    found: dict[str, np.ndarray] = {}
    for start in range(0, len(ids), 100):
        response = index.fetch(ids=ids[start : start + 100])
        for key, vec in response.vectors.items():
            found[key] = np.array(vec.values, dtype=float)
    return found


def classify(vectors: dict[str, np.ndarray], classes: list[str], text: np.ndarray, scale: float = 100.0):
    out = {}
    for key, v in vectors.items():
        v = v / np.linalg.norm(v)
        logits = scale * (text @ v)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = np.argsort(-p)
        out[key] = (classes[order[0]], float(p[order[0]]), classes[order[1]])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--labeled", type=int, default=600)
    parser.add_argument("--unlabeled", type=int, default=600)
    args = parser.parse_args()
    random.seed(7)

    print("Loading catalog...")
    rows = fetch_light_catalog()
    types = types_by_unique_id(rows)
    by_id = {str(r["unique_id"]): r for r in rows}
    labeled_ids = [i for i in by_id if i in types]
    unlabeled_ids = [i for i in by_id if i not in types]
    print(f"{len(labeled_ids)} labeled, {len(unlabeled_ids)} unlabeled")

    classes, text = build_text_matrix()
    lab = random.sample(labeled_ids, min(args.labeled, len(labeled_ids)))
    unl = random.sample(unlabeled_ids, min(args.unlabeled, len(unlabeled_ids)))
    vecs = fetch_vectors(lab + unl)
    guesses = classify(vecs, classes, text)

    # accuracy on labeled rows, overall and per type, by confidence band
    per_type: dict[str, list[bool]] = {}
    bands = {0.0: [], 0.5: [], 0.7: [], 0.9: []}
    for i in lab:
        if i not in guesses:
            continue
        guess, conf, _ = guesses[i]
        ok = guess == types[i]
        per_type.setdefault(types[i], []).append(ok)
        for b in bands:
            if conf >= b:
                bands[b].append(ok)
    print("\nAgreement with existing labels (labeled sample):")
    for b, oks in bands.items():
        if oks:
            print(f"  confidence >= {b:.1f}: {sum(oks)}/{len(oks)} = {sum(oks)/len(oks):.1%}")
    print("\nPer type (all confidences):")
    for t in sorted(per_type):
        oks = per_type[t]
        print(f"  {t:13s} {sum(oks):3d}/{len(oks):3d} = {sum(oks)/len(oks):.0%}")

    guessed_counts: dict[str, int] = {}
    unl_records = []
    for i in unl:
        if i not in guesses:
            continue
        guess, conf, second = guesses[i]
        guessed_counts[guess] = guessed_counts.get(guess, 0) + (1 if conf >= 0.7 else 0)
        r = by_id[i]
        unl_records.append({"id": i, "title": r.get("main_title"), "image": r.get("main_image"), "guess": guess, "confidence": round(conf, 3), "second": second})
    print("\nUnlabeled sample, CLIP guesses with confidence >= 0.7:")
    for k, n in sorted(guessed_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:20s} {n}")
    print(f"  (total unlabeled scored: {len(unl_records)})")

    json.dump(
        {
            "labeled": [
                {"id": i, "title": by_id[i].get("main_title"), "image": by_id[i].get("main_image"), "expected": types[i], "guess": guesses[i][0], "confidence": round(guesses[i][1], 3)}
                for i in lab if i in guesses
            ],
            "unlabeled": unl_records,
        },
        open(args.out, "w", encoding="utf-8"),
        ensure_ascii=False,
        indent=1,
    )
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
