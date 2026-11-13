"""Measure product matching on rooms generated from catalog products, so the right
answer of every piece in the picture is known.

Run from the backend/ directory, in this order (every step resumes where it stopped):
    python -m scripts.room_eval sample    --run pilot --rooms 10 --seed 1     pick the products
  or, to generate the way the app does (from a room photo, with the app's own prompt):
    python -m scripts.room_eval prepare   --run app --photos DIR --room-types living_room,bedroom --seed 2
    python -m scripts.room_eval preflight --run pilot                         check catalog, index, SAM 2
    python -m scripts.room_eval generate  --run pilot --max-usd 1             Muse Image via OpenRouter (costs money)
    python -m scripts.room_eval segment   --run pilot                         Grounding DINO + SAM 2
    python -m scripts.room_eval review    --run pilot                         a person checks each piece, blind
    python -m scripts.room_eval match     --run pilot                         the real matcher
    python -m scripts.room_eval equivalents --run pilot                       which results are the answer under another listing
    python -m scripts.room_eval judge     --run pilot                         optional: a person grades each result by eye, blind
    python -m scripts.room_eval report    --run pilot                         report.md, metrics.json, results.json

`match` refuses to start until every piece is reviewed, so nobody can see search
results before judging whether the generator drew the product faithfully.

Output goes to eval_out/rooms/<run>/. Only `generate` spends money; catalog and
index are only read.
"""

import argparse
import asyncio
import base64
import json
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from typing import Any

import httpx
from PIL import Image

from scripts.room_eval_metrics import (
    first_word,
    one_per_kind,
    rank_by_similarity,
    JUDGE_GRADES,
    human_judgments,
    EQUIVALENT_DEPTH,
    canonical_sku,
    MIN_PRODUCTS_PER_TYPE,
    bootstrap_ci,
    calibration_points,
    choose_threshold,
    detected_only,
    faithful_only,
    is_usable_product,
    item_outcomes,
    label_precision,
    plan_rooms,
    precision_curve,
    scores_by_type,
    split_rooms,
    summarize,
    threshold_grid,
)

BACKEND = Path(__file__).resolve().parents[1]
RUNS_DIR = BACKEND / "eval_out" / "rooms"
CATALOG_TABLE = "ikea_furniture_v2"
CATALOG_COLUMNS = "unique_id,sku,main_title,main_image,url,group_key,furniture_type,type_source,country_code"
PAGE = 1000
REF_MAX_PX = 1024

OPENROUTER_URL = "https://openrouter.ai/api/v1/images"
GENERATE_ATTEMPTS = 3
STYLES = ("Scandinavian", "modern minimalist", "warm contemporary", "Japandi", "mid-century modern", "neutral transitional")

SEGMENTATION_METHOD = "grounding_dino_sam2"
VERDICTS = ("same", "different", "unsure")
JUDGE_TOP = 5

# Below this many rooms the thresholds would follow noise, not the system.
MIN_ROOMS_TO_CALIBRATE = 50
TARGET_HIGH = 0.80
TARGET_MEDIUM = 0.50
MIN_N_PER_TYPE = 20
EQUIVALENT = "equivalent_rank"


# ── Files ───────────────────────────────────────────────────────────────────

def run_dir(name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        sys.exit("--run may only hold letters, digits, - and _.")
    return RUNS_DIR / name


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def write_json(path: Path, data: Any) -> None:
    """Through a temporary file, so a stopped run never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_manifest(run: Path) -> dict[str, Any]:
    manifest = read_json(run / "manifest.json")
    if manifest is None:
        sys.exit(f"No manifest in {run}. Run the sample step first.")
    return manifest


def load_env() -> None:
    # generation.py and segmentation.py read os.getenv; only main.py loads the file.
    from dotenv import load_dotenv

    load_dotenv(BACKEND / ".env")
    load_dotenv(BACKEND / "backend.env")


def require_v2_catalog() -> None:
    """The ids of the v1 and v2 tables differ: mixed, every match would be dropped."""
    from app.core.config import settings

    if settings.catalog_table != CATALOG_TABLE or not settings.pinecone_index_name.endswith("-v2"):
        sys.exit(f"Needs CATALOG_TABLE={CATALOG_TABLE} with the v2 index; got {settings.catalog_table} / {settings.pinecone_index_name}.")


# ── sample ──────────────────────────────────────────────────────────────────

def fetch_catalog() -> list[dict[str, Any]]:
    from app.services.product_repo import get_supabase_client

    client = get_supabase_client()
    rows: list[dict[str, Any]] = []
    while True:
        batch = (
            client.table(CATALOG_TABLE)
            .select(CATALOG_COLUMNS)
            .order("unique_id")
            .range(len(rows), len(rows) + PAGE - 1)
            .execute()
            .data
        )
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows


def usable_products_by_type(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Each product once, by its representative row: the one whose id is in the index."""
    from app.services.furniture_types import FURNITURE_TYPES
    from scripts.ingest_catalog_v2 import group_rows, pick_representative

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in group_rows(rows).values():
        representative = pick_representative(group)
        if representative and representative["furniture_type"] in FURNITURE_TYPES and is_usable_product(representative):
            by_type[representative["furniture_type"]].append(representative)
    return by_type


def used_groups(names: list[str] | None) -> set[str]:
    """Products of earlier runs, so a main run doesn't repeat the pilot's."""
    used: set[str] = set()
    for name in names or []:
        for room in load_manifest(run_dir(name))["rooms"]:
            used.update(p["group"] for p in room["products"])
    return used


def reference_kind(image: Image.Image) -> str:
    """"studio" for a product on a plain light background, "scene" for a picture
    of a furnished room. The catalog has both, and the index embeds whichever a
    product has, so a scene's vector stands for the whole room, not the product."""
    size = 64
    pixels = image.convert("RGB").resize((size, size)).load()
    border = [pixels[x, y] for x in range(size) for y in range(size) if x in (0, size - 1) or y in (0, size - 1)]
    light = sum(1 for pixel in border if min(pixel) >= 225)
    return "studio" if light / len(border) >= 0.6 else "scene"


def draw_product(pool: list[dict[str, Any]], type_name: str, index: Any, refs: Path) -> dict[str, Any]:
    """The next product of a shuffled pool that the index holds and whose picture
    still downloads, as a test case with its picture saved."""
    from app.services.embedding import download_image
    from app.services.furniture_types import image_group_key

    while pool:
        row = pool.pop()
        if str(row["unique_id"]) not in index.fetch(ids=[str(row["unique_id"])]).vectors:
            continue
        image = download_image(row["main_image"])
        if image is None:
            continue
        group = image_group_key(row["main_image"], row["sku"])
        path = refs / f"{re.sub(r'[^A-Za-z0-9_-]', '_', group)}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        image = image.convert("RGB")
        image.thumbnail((REF_MAX_PX, REF_MAX_PX))
        image.save(path, quality=92)
        return {
            "type": type_name,
            "group": group,
            "unique_id": str(row["unique_id"]),
            "sku": row["sku"],
            "title": row["main_title"],
            "image_url": row["main_image"],
            "url": row.get("url"),
            "ref": path.name,
            "ref_kind": reference_kind(image),
        }
    sys.exit(f"Ran out of usable {type_name} products.")


def cmd_sample(args: argparse.Namespace) -> None:
    from app.services.pinecone_client import get_index

    require_v2_catalog()
    run = run_dir(args.run)
    if (run / "manifest.json").exists():
        sys.exit(f"{run} already has a manifest; the answers of a run never change. Use a new --run.")

    by_type = usable_products_by_type(fetch_catalog())
    excluded = used_groups(args.exclude_run)
    from app.services.furniture_types import image_group_key

    for type_name in by_type:
        by_type[type_name] = [r for r in by_type[type_name] if image_group_key(r["main_image"], r["sku"]) not in excluded]
    print("Usable products per type:")
    for type_name, pool in sorted(by_type.items(), key=lambda kv: len(kv[1])):
        print(f"  {type_name:14} {len(pool)}")
    available = [t for t, pool in by_type.items() if len(pool) >= MIN_PRODUCTS_PER_TYPE]
    print("Left out, too few products:", sorted(set(by_type) - set(available)) or "none")

    rng = random.Random(args.seed)
    for type_name in sorted(by_type):
        by_type[type_name].sort(key=lambda r: str(r["unique_id"]))
        rng.shuffle(by_type[type_name])

    index = get_index()
    rooms = []
    for planned in plan_rooms(args.rooms, sorted(available), rng):
        products = [draw_product(by_type[t], t, index, run / "refs") for t in planned["types"]]
        rooms.append({"id": planned["id"], "kind": planned["kind"], "style": rng.choice(STYLES), "products": products})
        print(f"{planned['id']} {planned['kind']}: {', '.join(planned['types'])}")

    write_json(
        run / "manifest.json",
        {
            "run": args.run,
            "seed": args.seed,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "catalog_table": CATALOG_TABLE,
            "usable_products": {t: len(p) for t, p in by_type.items()},
            "rooms": rooms,
        },
    )
    print(f"Wrote {run / 'manifest.json'}")


# ── prepare (the app's own generation flow) ─────────────────────────────────

# The app's room, style and colour ids (lib/features/home/presentation/home_screen.dart),
# in an order that walks through all three styles before repeating one.
APP_STYLES = (
    ("japandi", "warm_oat_cream"),
    ("industrial_loft", "raw_concrete_matte_black"),
    ("modern_luxury", "ivory_champagne_gold"),
    ("japandi", "muted_sage_green"),
    ("industrial_loft", "rusty_brick_leather"),
    ("modern_luxury", "emerald_green_brass"),
    ("japandi", "soft_terracotta"),
    ("industrial_loft", "dark_navy_blue"),
    ("modern_luxury", "midnight_blue_silver"),
)
# Kitchen is left out: the matcher has no kitchen types to search.
APP_ROOM_KINDS = {"living_room": "living room", "bedroom": "bedroom", "dining_room": "dining room", "home_office": "home office"}
PHOTO_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# Without a usable pick from the composer, the first candidates are placed.
FALLBACK_PRODUCTS = 3


def product_title(product: dict[str, Any]) -> str:
    return product["title"].replace(" - IKEA", "").strip()


def composer_text(room_type: str, style_desc: str, color_desc: str, candidates: list[dict[str, Any]], note: str | None = None) -> str:
    """The message the app sends its prompt composer, with catalog products to
    choose from added. The composer picks what suits the room, as it picks
    furniture in the app, and names its picks so they are the answers. The
    app's own system prompt is used unchanged."""
    lines = "\n".join(f"{n}. {product_title(p)}" for n, p in enumerate(candidates, start=1))
    return (
        f"Room type: {room_type}\n"
        f"Style: {style_desc}\n"
        f"Color: {color_desc}\n\n"
        "Furniture to choose from (catalog products, numbered):\n"
        f"{lines}\n\n"
        "Furnish the room only with products from this list: pick the ones that suit this room and its style, "
        "as many as fit comfortably (at least one, and at most one of each kind of furniture), and decide where "
        "each goes. Do not describe any other furniture. "
        f"After the paragraph, add one last line naming the numbers you used, exactly like: {CHOICE_PREFIX} 1, 3\n\n"
        + (f"{note}\n\n" if note else "")
        + "Here is the room photo:"
    )


CHOICE_PREFIX = "PRODUCTS:"
_CHOICE_LINE = re.compile(r"^\s*\**\s*PRODUCTS\s*:\s*\**\s*([0-9 ,]+?)\s*\**\s*$", re.IGNORECASE | re.MULTILINE)


def parse_composer_choice(text: str, count: int) -> tuple[str, list[int]]:
    """The composer's paragraph and the products it picked (0-based), from its
    answer. A missing, empty, repeated or out-of-range pick is an error: the
    picks are the answers of the room, so a doubtful one can't be used."""
    lines = list(_CHOICE_LINE.finditer(text))
    if not lines:
        raise ValueError(f"no '{CHOICE_PREFIX}' line")
    last = lines[-1]
    numbers = [int(n) for n in re.findall(r"\d+", last[1])]
    if not numbers:
        raise ValueError("no product numbers")
    if len(set(numbers)) != len(numbers):
        raise ValueError(f"a product picked twice: {numbers}")
    if any(not 1 <= n <= count for n in numbers):
        raise ValueError(f"a product number outside 1-{count}: {numbers}")
    paragraph = (text[: last.start()] + text[last.end() :]).strip()
    if not paragraph:
        raise ValueError("no paragraph")
    return paragraph, [n - 1 for n in numbers]


def product_clause(products: list[dict[str, Any]]) -> str:
    """Tells the image model which reference shows which product. Reference 1
    is the room photo, as in the app; the products follow it."""
    lines = "\n".join(f"Reference image {n}: {product_title(p)}" for n, p in enumerate(products, start=2))
    return (
        "Reference image 1 is the room to redecorate. The other reference images show the exact "
        "products to place in it, one of each:\n"
        f"{lines}\n"
        "Reproduce each product exactly as in its reference image: the same shape, proportions, colour, "
        "material and details. Do not add any other large furniture."
    )


async def compose_with_products(
    room_type: str, style: str, color: str, photo: bytes, mime_type: str, candidates: list[dict[str, Any]], note: str | None = None
) -> str:
    """generation.compose_design_prompt with the products to choose from in its message."""
    import generation

    payload = {
        "systemInstruction": {"parts": [{"text": generation._COMPOSER_SYSTEM_PROMPT}]},
        "contents": [
            {
                "parts": [
                    {
                        "text": composer_text(
                            room_type,
                            generation._STYLE_DESCRIPTIONS.get(style, style),
                            generation._COLOR_DESCRIPTIONS.get(color, color),
                            candidates,
                            note,
                        )
                    },
                    {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(photo).decode("utf-8")}},
                ]
            }
        ],
    }
    model = os.getenv("GEMINI_PROMPT_MODEL", "gemini-3.6-flash")
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"https://generativelanguage.googleapis.com/v1/models/{model}:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"], "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
    return response.json()["candidates"][0]["content"]["parts"][0]["text"].strip()


def app_prompt(paragraph: str | None, room_type: str, style: str, color: str, products: list[dict[str, Any]]) -> str:
    """As run_generate_room builds it: the composer's paragraph, then the
    structure clause; the app's static prompt when the composer failed. The
    product clause goes before the structure clause in both."""
    import generation

    clause = generation._STRUCTURE_PRESERVATION_CLAUSE
    if paragraph is None:
        fallback = generation.build_static_fallback_prompt(room_type, style, color)
        opening = fallback[: fallback.index(clause)].rstrip()
        return f"{opening}\n\n{product_clause(products)}\n\n{clause}"
    return f"{paragraph}\n\n{product_clause(products)}\n\n{clause}"


def style_text(type_name: str, style: str, color: str) -> str:
    """What a product of this type should look like in this room, in the app's
    own words for the style and colour (short enough for CLIP's 77 tokens)."""
    import generation

    return (
        f"A {type_name.replace('_', ' ')}. "
        f"{generation._STYLE_DESCRIPTIONS.get(style, style)}. {generation._COLOR_DESCRIPTIONS.get(color, color)}"
    )


def index_vectors(index: Any, rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    """The products' picture vectors already in the index, by unique_id."""
    ids = [str(r["unique_id"]) for r in rows]
    found: dict[str, list[float]] = {}
    for start in range(0, len(ids), 100):
        for vector_id, vector in index.fetch(ids=ids[start : start + 100]).vectors.items():
            found[vector_id] = list(vector.values)
    return found


def styled_candidates(
    pool: list[dict[str, Any]], vectors: dict[str, list[float]], text_vector: list[float], take: int,
    type_name: str, index: Any, refs: Path, used: set[str],
) -> list[dict[str, Any]]:
    """The `take` products of a type whose pictures CLIP finds closest to the
    room's style and colour, one per product line so they aren't one design in
    two colours, and none already given to another room."""
    from app.services.furniture_types import image_group_key

    by_id = {str(r["unique_id"]): r for r in pool}
    queue = [by_id[i] for i in reversed(rank_by_similarity(vectors, text_vector))]
    picked: list[dict[str, Any]] = []
    lines: set[str] = set()
    while queue and len(picked) < take:
        row = queue[-1]
        if image_group_key(row["main_image"], row["sku"]) in used or first_word(row["main_title"]) in lines:
            queue.pop()
            continue
        product = draw_product(queue, type_name, index, refs)  # pops it, and any whose picture won't download
        if first_word(product["title"]) in lines or product["group"] in used:
            continue
        lines.add(first_word(product["title"]))
        used.add(product["group"])
        picked.append(product)
    return picked


# Furniture words for the app's colour ids. The app's own colour text is about
# walls and cushions ("sage green on accent walls"), so on its own it hardly
# moves which furniture CLIP finds; these name the colour on the piece itself.
COLOR_WORDS = {
    "warm_oat_cream": "warm oat and cream",
    "muted_sage_green": "muted sage green",
    "soft_terracotta": "soft terracotta",
    "raw_concrete_matte_black": "concrete grey and matte black",
    "rusty_brick_leather": "rusty brick red leather",
    "dark_navy_blue": "dark navy blue",
    "ivory_champagne_gold": "ivory and champagne gold",
    "emerald_green_brass": "emerald green and brass",
    "midnight_blue_silver": "midnight blue and silver",
}


def unit(vector: Any) -> Any:
    return vector / (float((vector**2).sum()) ** 0.5 or 1.0)


def target_vector(clip: Any, type_name: str, style: str, color: str, color_weight: float) -> list[float]:
    """What CLIP should look for: the style text, plus the colour on the piece
    weighted by `color_weight`. With unit vectors, ranking by this sum ranks by
    the weighted mean of the two similarities."""
    style_vector = unit(clip.encode([style_text(type_name, style, color)])[0])
    if not color_weight:
        return style_vector.tolist()
    color_vector = unit(clip.encode([f"A {COLOR_WORDS.get(color, color)} {type_name.replace('_', ' ')}"])[0])
    return (style_vector + color_weight * color_vector).tolist()


def prepare_room(
    room_id: str, kind: str, types: list[str], room_type: str, style: str, color: str, photo_name: str,
    source: Path, mime_type: str, catalog: dict[str, Any], args: argparse.Namespace, run: Path,
) -> dict[str, Any]:
    """One room's candidates (CLIP ranks them for the style and colour), the
    composer's picks among them, and the final prompt. `catalog` holds the
    usable products by type, their index vectors, the products already given
    out, the index and the CLIP model."""
    candidates = []
    for type_name in types:
        if type_name not in catalog["vectors"]:
            catalog["vectors"][type_name] = index_vectors(catalog["index"], catalog["by_type"][type_name])
        target = target_vector(catalog["clip"], type_name, style, color, args.color_weight)
        candidates += styled_candidates(
            catalog["by_type"][type_name], catalog["vectors"][type_name], target, args.per_type,
            type_name, catalog["index"], run / "refs", catalog["used"],
        )
    # Without --composer the photo goes to no model but Muse Image, and the
    # prompt is the app's static one (what the app sends when its composer fails).
    paragraph, composer_error, answer = None, "not used", None
    if args.composer:
        try:
            answer = asyncio.run(compose_with_products(room_type, style, color, source.read_bytes(), mime_type, candidates))
            paragraph, picked = parse_composer_choice(answer, len(candidates))
            composer_error = None
        except Exception as exc:  # the app falls back to its static prompt; so does this
            paragraph, composer_error = None, f"{type(exc).__name__}: {exc}"
    if paragraph is None:
        picked = list(range(len(candidates)))
    # One product per kind: the composer may pick two chairs; the first counts.
    products = one_per_kind([candidates[i] for i in picked])
    if paragraph is None:
        products = products[:FALLBACK_PRODUCTS]
    source_of_prompt = "composer" if paragraph else f"app static ({composer_error})"
    print(f"{room_id} {room_type} {style}/{color}: picked {', '.join(p['type'] for p in products)} of {len(candidates)}; prompt from {source_of_prompt}")
    return {
        "id": room_id,
        "kind": kind,
        "style": f"{style} · {color}",
        "candidate_ranking": {"per_type": args.per_type, "color_weight": args.color_weight},
        "candidates": candidates,
        "products": products,
        "app": {
            "room_type": room_type,
            "style": style,
            "color": color,
            "photo": photo_name,
            "source": source.name,
            "source_mime": mime_type,
            "composer_answer": answer,
            "composer_paragraph": paragraph,
            "composer_error": composer_error,
            "prompt": app_prompt(paragraph, room_type, style, color, products),
        },
    }


def load_catalog_for_prepare(exclude: set[str], seed: int) -> dict[str, Any]:
    """Usable products by type without the excluded ones, shuffled by `seed`,
    with the index and CLIP the ranking needs."""
    from app.services.embedding import get_model
    from app.services.furniture_types import image_group_key
    from app.services.pinecone_client import get_index

    by_type = usable_products_by_type(fetch_catalog())
    for type_name in by_type:
        by_type[type_name] = [r for r in by_type[type_name] if image_group_key(r["main_image"], r["sku"]) not in exclude]
    rng = random.Random(seed)
    for type_name in sorted(by_type):
        by_type[type_name].sort(key=lambda r: str(r["unique_id"]))
        rng.shuffle(by_type[type_name])
    return {"by_type": by_type, "vectors": {}, "used": set(), "index": get_index(), "clip": get_model(), "rng": rng}


def cmd_prepare(args: argparse.Namespace) -> None:
    """Everything before the image model: products, composer, final prompt.
    Writes the manifest; nothing is generated until the generate step."""
    require_v2_catalog()
    load_env()
    run = run_dir(args.run)
    if (run / "manifest.json").exists():
        sys.exit(f"{run} already has a manifest; the answers of a run never change. Use a new --run.")
    photos = sorted(p for p in Path(args.photos).iterdir() if p.suffix.lower() in PHOTO_SUFFIXES)
    room_types = [t.strip() for t in args.room_types.split(",")]
    if len(room_types) != len(photos):
        sys.exit(f"{len(photos)} photos but {len(room_types)} room types; give one type per photo, in file-name order.")
    unknown = sorted(set(room_types) - set(APP_ROOM_KINDS))
    if unknown:
        sys.exit(f"Unknown room types {unknown}; use {', '.join(APP_ROOM_KINDS)}.")

    catalog = load_catalog_for_prepare(used_groups(args.exclude_run), args.seed)
    available = [t for t, pool in catalog["by_type"].items() if len(pool) >= MIN_PRODUCTS_PER_TYPE]
    planned = plan_rooms(len(photos), sorted(available), catalog["rng"], size=args.types, kinds=[APP_ROOM_KINDS[t] for t in room_types])
    rooms = []
    for position, (photo, room_type, plan) in enumerate(zip(photos, room_types, planned)):
        style, color = APP_STYLES[position % len(APP_STYLES)]
        source = run / "sources" / f"{plan['id']}{photo.suffix.lower()}"
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_bytes(photo.read_bytes())
        mime_type = "image/png" if photo.suffix.lower() == ".png" else "image/webp" if photo.suffix.lower() == ".webp" else "image/jpeg"
        rooms.append(prepare_room(plan["id"], plan["kind"], plan["types"], room_type, style, color, photo.name, source, mime_type, catalog, args, run))

    write_json(
        run / "manifest.json",
        {
            "run": args.run,
            "seed": args.seed,
            "flow": "app",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "catalog_table": CATALOG_TABLE,
            "usable_products": {t: len(p) for t, p in catalog["by_type"].items()},
            "rooms": rooms,
        },
    )
    print(f"Wrote {run / 'manifest.json'}. Nothing was generated yet.")


def cmd_recompose(args: argparse.Namespace) -> None:
    """Ask the composer again for one room, with the same candidates and an
    extra instruction (`--note`), e.g. to furnish a sparse room more fully.
    Only before the room is generated. The note is kept in the manifest, as it
    is something the app does not say."""
    require_v2_catalog()
    load_env()
    run = run_dir(args.run)
    manifest = load_manifest(run)
    room = next((r for r in manifest["rooms"] if r["id"] == args.room), None)
    if room is None or "app" not in room:
        sys.exit(f"No app-flow room {args.room} in {run}.")
    if (run / "rooms" / f"{args.room}.png").exists():
        sys.exit(f"{args.room} is generated already; its answers can't change.")
    app = room["app"]
    source = run / "sources" / app["source"]
    answer = asyncio.run(
        compose_with_products(app["room_type"], app["style"], app["color"], source.read_bytes(), app["source_mime"], room["candidates"], args.note)
    )
    paragraph, picked = parse_composer_choice(answer, len(room["candidates"]))  # a doubtful pick stops here, nothing written
    room["products"] = one_per_kind([room["candidates"][i] for i in picked])
    app.update(
        composer_answer=answer,
        composer_paragraph=paragraph,
        composer_error=None,
        composer_note=args.note,
        prompt=app_prompt(paragraph, app["room_type"], app["style"], app["color"], room["products"]),
    )
    write_json(run / "manifest.json", manifest)
    print(f"{args.room}: picked {', '.join(p['type'] for p in room['products'])} of {len(room['candidates'])}. Nothing was generated yet.")


def cmd_reprepare(args: argparse.Namespace) -> None:
    """Prepare one room of an app-flow run again, e.g. with more weight on its
    colour, keeping the other rooms as they are. Only before it is generated:
    after that, its answers are fixed."""
    require_v2_catalog()
    load_env()
    run = run_dir(args.run)
    manifest = load_manifest(run)
    position = next((i for i, r in enumerate(manifest["rooms"]) if r["id"] == args.room), None)
    if position is None:
        sys.exit(f"No room {args.room} in {run}.")
    if (run / "rooms" / f"{args.room}.png").exists():
        sys.exit(f"{args.room} is generated already; its answers can't change.")
    old = manifest["rooms"][position]
    # Keep every other room's candidates (and earlier runs' products) out of this one.
    others = {c["group"] for r in manifest["rooms"] if r["id"] != args.room for c in r.get("candidates", r["products"])}
    catalog = load_catalog_for_prepare(used_groups(args.exclude_run) | others, manifest["seed"])
    types = list(dict.fromkeys(c["type"] for c in old["candidates"]))
    app = old["app"]
    manifest["rooms"][position] = prepare_room(
        old["id"], old["kind"], types, app["room_type"], app["style"], app["color"], app["photo"],
        run / "sources" / app["source"], app["source_mime"], catalog, args, run,
    )
    write_json(run / "manifest.json", manifest)
    print(f"Rewrote {args.room} in {run / 'manifest.json'}. Nothing was generated yet.")


# ── preflight ───────────────────────────────────────────────────────────────

def load_segmentation(run: Path):
    """Import segmentation with this run's settings: it reads them at import.
    Colours and materials are off because no filter here uses them."""
    os.chdir(BACKEND)  # its checkpoint path is relative
    os.environ["FEATURE_PROVIDER"] = "none"
    os.environ["SEGMENTATION_RESULTS_DIR"] = str(run / "segmentation_results")
    load_env()
    from segmentation import run_segmentation

    return run_segmentation


def cmd_preflight(args: argparse.Namespace) -> None:
    from app.services.pinecone_client import get_index
    from app.services.product_repo import fetch_products_by_ids

    require_v2_catalog()
    load_env()
    run = run_dir(args.run)
    manifest = load_manifest(run)
    index = get_index()
    failed = False

    def check(name: str, passed: bool, detail: Any = "") -> None:
        nonlocal failed
        failed = failed or not passed
        print(f"[{'ok' if passed else 'FAIL'}] {name} {detail}")

    probe = [0.01] * 512
    matches = index.query(vector=probe, top_k=50, filter={"furniture_type": {"$eq": "sofa"}}).matches
    rows = fetch_products_by_ids([m.id for m in matches])
    check("index ids find their catalog rows", len(matches) > 0 and len(rows) == len(matches), f"{len(rows)}/{len(matches)}")
    subtype = index.query(
        vector=probe, top_k=1, filter={"furniture_type": {"$eq": "lamp"}, "furniture_subtype": {"$eq": "floor"}}
    ).matches
    check("index metadata has the lamp subtype", bool(subtype))
    check("OpenRouter key and Muse Image model are set", bool(os.getenv("OPENROUTER_API_KEY")) and os.getenv("OPENROUTER_MODEL") == "meta/muse-image")

    picture = run / "refs" / manifest["rooms"][0]["products"][0]["ref"]
    run_segmentation = load_segmentation(run)
    result = asyncio.run(run_segmentation(picture.read_bytes(), "image/jpeg"))
    check("segmentation runs on SAM 2, not the fallback", result.method == SEGMENTATION_METHOD, f"{result.method} {result.fallback_reason or ''}")
    check("segmentation gives cutouts", any(i.cutout_image for i in result.items), f"{len(result.items)} items in {picture.name}")
    if failed:
        sys.exit("Preflight failed: nothing was generated.")
    print("Preflight passed.")


# ── generate ────────────────────────────────────────────────────────────────

def room_prompt(room: dict[str, Any]) -> str:
    """Every product of the room by position in the reference list, asked for
    exactly as pictured: the answer is only valid if the picture shows it."""
    lines = [
        f"Reference image {position}: {product['title'].replace(' - IKEA', '').strip()}"
        for position, product in enumerate(room["products"], start=1)
    ]
    return (
        f"Create a photorealistic interior photograph of a furnished {room['kind']} in a {room['style']} style.\n"
        f"The room must contain exactly these {len(lines)} products, one of each:\n"
        + "\n".join(lines)
        + "\nReproduce each product exactly as in its reference image: the same shape, proportions, colour, "
        "material and details. Do not restyle, recolour or simplify them.\n"
        "Show every product fully in frame, from a wide angle, with clear space around it and not hidden by "
        "anything. Do not add any other furniture, lamps, rugs, curtains, mirrors, plants or wall art. "
        "No people, text or logos."
    )


def reference_board(images: list[Image.Image], cell: int = 512) -> bytes:
    """The products side by side as one picture, for a model that takes a single reference."""
    columns = 2
    rows = (len(images) + 1) // columns
    board = Image.new("RGB", (cell * columns, cell * rows), "white")
    for position, image in enumerate(images):
        fitted = image.convert("RGB")
        fitted.thumbnail((cell, cell))
        x = position % columns * cell + (cell - fitted.width) // 2
        y = position // columns * cell + (cell - fitted.height) // 2
        board.paste(fitted, (x, y))
    buffer = BytesIO()
    board.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


def image_mime(image_bytes: bytes) -> str:
    if image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/webp" if image_bytes[8:12] == b"WEBP" else "image/jpeg"


def data_url(image_bytes: bytes) -> str:
    return f"data:{image_mime(image_bytes)};base64," + base64.b64encode(image_bytes).decode("utf-8")


def critic_loop(
    prompt: str,
    generate: Any,
    critique: Any,
    max_retries: int,
    reminder: Any,
) -> list[dict[str, Any]]:
    """run_generate_room's critic retries: generate, have the critic check it,
    and on a FAIL generate again with the critic's reason added to the prompt,
    up to max_retries times. The last attempt is kept, passed or not, as in the
    app. `generate(prompt)` gives (image, cost); `critique(image)` gives (passed, verdict)."""
    attempts = []
    while True:
        image, cost = generate(prompt)
        passed, verdict = critique(image)
        attempts.append({"prompt": prompt, "image": image, "cost": cost, "passed": passed, "verdict": verdict})
        if passed or len(attempts) > max_retries:
            return attempts
        prompt = f"{prompt}{reminder(verdict)}"


def request_room(prompt: str, references: list[bytes], model: str, api_key: str) -> tuple[bytes, float | None]:
    """Same call as generation.generate_with_openrouter, with several references.
    A rate limit, server error or dropped connection is tried again; a rejected
    request stops the run, because every other room would be rejected as well."""
    payload = {
        "model": model,
        "prompt": prompt,
        "input_references": [{"type": "image_url", "image_url": {"url": data_url(r)}} for r in references],
    }
    for attempt in range(1, GENERATE_ATTEMPTS + 1):
        try:
            response = httpx.post(
                OPENROUTER_URL,
                headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
                json=payload,
                timeout=180,
            )
            if response.status_code == 429 or response.status_code >= 500:
                raise httpx.HTTPStatusError(response.text[:300], request=response.request, response=response)
            if response.status_code >= 400:
                sys.exit(f"OpenRouter rejected the request ({response.status_code}): {response.text[:500]}")
            body = response.json()
            data = body.get("data") or []
            if not data or not data[0].get("b64_json"):
                raise ValueError(f"no image in the answer: {str(body)[:300]}")
            return base64.b64decode(data[0]["b64_json"]), (body.get("usage") or {}).get("cost")
        except (httpx.HTTPError, ValueError) as exc:
            print(f"  attempt {attempt} failed: {exc}")
            if attempt == GENERATE_ATTEMPTS:
                raise
            time.sleep(5 * attempt)
    raise AssertionError("unreachable")


def generate_app_room(
    run: Path, room: dict[str, Any], product_refs: list[bytes], model: str, api_key: str, log: dict[str, Any], log_path: Path
) -> None:
    """The app's flow for one room: its room photo first, as the image it edits,
    then the products; the critic and its retries from generation.py."""
    import generation

    app = room["app"]
    source = (run / "sources" / app["source"]).read_bytes()
    references = [source] + product_refs
    started = time.monotonic()

    def generate(prompt: str) -> tuple[bytes, float | None]:
        return request_room(prompt, references, model, api_key)

    def critique(image: bytes) -> tuple[bool, str]:
        return asyncio.run(
            generation.critique_generated_image(
                source, app["source_mime"], image, image_mime(image), app["room_type"], app["style"], app["color"]
            )
        )

    attempts = critic_loop(app["prompt"], generate, critique, generation.CRITIC_MAX_RETRIES, generation.critic_retry_reminder)
    folder = run / "rooms"
    folder.mkdir(parents=True, exist_ok=True)
    for number, attempt in enumerate(attempts, start=1):
        Image.open(BytesIO(attempt["image"])).convert("RGB").save(folder / f"{room['id']}_attempt{number}.png")
    # Only the last attempt is a test picture, as only it reaches the user in the app.
    Image.open(BytesIO(attempts[-1]["image"])).convert("RGB").save(folder / f"{room['id']}.png")
    costs = [a["cost"] or 0 for a in attempts]
    log[room["id"]] = {
        "model": model,
        "references": "app",
        "prompt": attempts[-1]["prompt"],
        "cost": sum(costs),
        "seconds": round(time.monotonic() - started, 1),
        "critic_passed": attempts[-1]["passed"],
        "attempts": [
            {"file": f"{room['id']}_attempt{n}.png", "cost": a["cost"], "passed": a["passed"], "verdict": a["verdict"]}
            for n, a in enumerate(attempts, start=1)
        ],
    }
    write_json(log_path, log)
    verdicts = " | ".join(a["verdict"] or "PASS (no verdict)" for a in attempts)
    print(f"{room['id']}: {len(attempts)} attempt(s), cost=${sum(costs):.4f}, critic: {verdicts}")


def cmd_generate(args: argparse.Namespace) -> None:
    load_env()
    run = run_dir(args.run)
    manifest = load_manifest(run)
    api_key, model = os.getenv("OPENROUTER_API_KEY"), os.getenv("OPENROUTER_MODEL")
    if not api_key or model != "meta/muse-image":
        sys.exit("OPENROUTER_API_KEY must be set and OPENROUTER_MODEL must be meta/muse-image.")

    log_path = run / "generation.json"
    log = read_json(log_path, {})
    spent = sum(entry.get("cost") or 0 for entry in log.values())
    done = 0
    for room in manifest["rooms"]:
        picture = run / "rooms" / f"{room['id']}.png"
        if picture.exists():
            continue
        if args.limit is not None and done >= args.limit:
            break
        if spent >= args.max_usd:
            print(f"Stopped at the budget: ${spent:.4f} of ${args.max_usd}.")
            break

        paths = [run / "refs" / p["ref"] for p in room["products"]]
        if "app" in room:
            generate_app_room(run, room, [p.read_bytes() for p in paths], model, api_key, log, log_path)
            spent += log[room["id"]]["cost"]
            done += 1
            continue
        if args.references == "board":
            references = [reference_board([Image.open(p) for p in paths])]
            prompt = room_prompt(room)
        else:
            references = [p.read_bytes() for p in paths]
            prompt = room_prompt(room)
        started = time.monotonic()
        image_bytes, cost = request_room(prompt, references, model, api_key)

        picture.parent.mkdir(parents=True, exist_ok=True)
        Image.open(BytesIO(image_bytes)).convert("RGB").save(picture)
        log[room["id"]] = {
            "model": model,
            "references": args.references,
            "prompt": prompt,
            "cost": cost,
            "seconds": round(time.monotonic() - started, 1),
        }
        write_json(log_path, log)
        spent += cost or 0
        done += 1
        print(f"{room['id']}: {picture.name} cost={cost} total=${spent:.4f}")
    print(f"{sum(1 for r in manifest['rooms'] if (run / 'rooms' / (r['id'] + '.png')).exists())}/{len(manifest['rooms'])} rooms generated.")


# ── segment ─────────────────────────────────────────────────────────────────

def save_data_url(url: str | None, path: Path) -> str | None:
    if not url:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(base64.b64decode(url.split(",", 1)[1]))
    return path.name


def cmd_segment(args: argparse.Namespace) -> None:
    from app.services.furniture_types import label_to_type

    run = run_dir(args.run)
    manifest = load_manifest(run)
    run_segmentation = load_segmentation(run)
    for room in manifest["rooms"]:
        picture = run / "rooms" / f"{room['id']}.png"
        out = run / "segments" / f"{room['id']}.json"
        if out.exists() or not picture.exists():
            continue
        started = time.monotonic()
        result = asyncio.run(run_segmentation(picture.read_bytes(), "image/png"))
        # A Gemini box isn't what the app searches with, so it can't be scored with the rest.
        if result.method != SEGMENTATION_METHOD:
            sys.exit(f"{room['id']}: segmentation fell back to {result.method} ({result.fallback_reason}). Nothing was saved for this room.")
        detections = []
        for position, item in enumerate(result.items):
            stem = f"{room['id']}_{position:02d}"
            detections.append(
                {
                    "id": stem,
                    "label": item.label,
                    "type": label_to_type(item.label),
                    "confidence": item.confidence,
                    "bbox": item.bbox.model_dump(),
                    "mask_precise": item.mask_precise,
                    "cutout": save_data_url(item.cutout_image, run / "cutouts" / f"{stem}.png"),
                    "crop": save_data_url(item.crop_image, run / "cutouts" / f"{stem}.jpg"),
                }
            )
        write_json(out, {"method": result.method, "seconds": round(time.monotonic() - started, 1), "detections": detections})
        found = {d["type"] for d in detections}
        wanted = [p["type"] for p in room["products"]]
        print(f"{room['id']}: {len(detections)} detections in {time.monotonic() - started:.0f}s; answers found {sum(t in found for t in wanted)}/{len(wanted)}")


# ── review ──────────────────────────────────────────────────────────────────

def review_items(run: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every detection with a picture, beside the products of its room, in an
    order that hides the room. Holds only pictures; search results are never
    read here. The reviewer, not the detector's label, says which product it is."""
    items = []
    for room in manifest["rooms"]:
        segments = read_json(run / "segments" / f"{room['id']}.json")
        if segments is None:
            continue
        for detection in segments["detections"]:
            if detection["cutout"] or detection["crop"]:
                items.append(
                    {
                        "id": detection["id"],
                        "cutout": detection["cutout"] or detection["crop"],
                        "products": [{"group": p["group"], "ref": p["ref"]} for p in room["products"]],
                    }
                )
    rng = random.Random(manifest["seed"])
    rng.shuffle(items)
    for item in items:
        rng.shuffle(item["products"])  # so the answer isn't always in one place
    return items


def valid_review(answer: Any, item: dict[str, Any]) -> bool:
    """{"group": null} for a piece that is none of the products, else a product
    of its room with a verdict on how faithfully it was drawn."""
    if not isinstance(answer, dict):
        return False
    if answer.get("group") is None:
        return answer.get("verdict") is None
    return answer["group"] in {p["group"] for p in item["products"]} and answer.get("verdict") in VERDICTS


REVIEW_PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8"><title>Room review</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font:16px system-ui,sans-serif;margin:0;padding:16px;background:#f4f4f2;color:#222;text-align:center}
.cut{display:inline-block;background:#fff;border-radius:8px;padding:12px;margin:8px 0}
.cut img{width:min(360px,88vw);height:min(360px,88vw);object-fit:contain;display:block}
.refs{display:flex;gap:10px;justify-content:center;flex-wrap:wrap;margin:8px 0}
.refs button{padding:6px;background:#fff}
.refs img{width:min(170px,40vw);height:min(170px,40vw);object-fit:contain;display:block}
button{font:inherit;padding:10px 18px;margin:4px;border:2px solid #aaa;border-radius:8px;background:#fff;cursor:pointer}
button.on{border-color:#222;background:#222;color:#fff}
.refs button.on{background:#fff;border-color:#e0531f;box-shadow:0 0 0 3px #e0531f}
</style></head><body><h2 id="head"></h2>
<div class="cut"><img id="cut" alt="ชิ้นที่ตัดจากห้อง"></div>
<p><b>1.</b> ชิ้นนี้คือสินค้าตัวไหน (กด a b c d หรือ 0 ถ้าไม่ใช่ตัวไหนเลย)</p>
<div class="refs" id="refs"></div>
<p><button id="none">0 ไม่ใช่ตัวไหนเลย</button></p>
<div id="verdict"><p><b>2.</b> เหมือนสินค้าต้นฉบับไหม (ดูรูปทรง สี และรายละเอียด)</p>
<button data-v="same">1 เหมือน</button><button data-v="different">2 เพี้ยน</button><button data-v="unsure">3 ไม่แน่ใจ</button></div>
<p><button id="back">ย้อนกลับ</button></p>
<script>
const items=__ITEMS__;let reviews=__REVIEWS__;
let at=items.findIndex(p=>!reviews[p.id]);if(at<0)at=items.length;let chosen;
const q=s=>document.querySelector(s);
const letters="abcd";
function show(){
  const done=items.filter(p=>reviews[p.id]).length;
  const end=at>=items.length;
  ["#refs",".cut","#none","#verdict"].forEach(s=>q(s).style.display=end?"none":"");
  if(end){q("#head").textContent="ตรวจแล้ว "+done+"/"+items.length+" ชิ้น"+(done===items.length?" ครบแล้ว ปิดหน้านี้ได้":"");return}
  const p=items[at];const saved=reviews[p.id];chosen=saved?saved.group:undefined;
  q("#head").textContent="ชิ้นที่ "+(at+1)+"/"+items.length+" (ตรวจแล้ว "+done+")";
  q("#cut").src="/image/"+at+"/cutout";
  q("#refs").innerHTML=p.products.map((r,i)=>'<button data-g="'+i+'"><img src="/image/'+at+'/ref'+i+'" alt="สินค้า '+letters[i]+'">'+letters[i]+'</button>').join("");
  q("#refs").querySelectorAll("[data-g]").forEach(b=>b.onclick=()=>choose(p.products[+b.dataset.g].group));
  mark();
}
function mark(){
  const p=items[at];const saved=reviews[p.id];
  q("#refs").querySelectorAll("[data-g]").forEach(b=>b.classList.toggle("on",chosen===p.products[+b.dataset.g].group));
  q("#none").classList.toggle("on",chosen===null);
  q("#verdict").style.visibility=chosen?"visible":"hidden";
  document.querySelectorAll("[data-v]").forEach(b=>b.classList.toggle("on",!!saved&&saved.group===chosen&&saved.verdict===b.dataset.v));
}
async function save(answer){
  const p=items[at];
  const r=await fetch("/answer",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:p.id,answer:answer})});
  if(!r.ok){alert("บันทึกไม่สำเร็จ");return}
  reviews[p.id]=answer;at++;show();
}
function choose(group){chosen=group;mark();if(group===null)save({group:null,verdict:null})}
function verdict(v){if(chosen)save({group:chosen,verdict:v})}
q("#none").onclick=()=>choose(null);
document.querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>verdict(b.dataset.v));
q("#back").onclick=()=>{if(at>0){at--;show()}};
addEventListener("keydown",e=>{
  if(at>=items.length)return;const p=items[at];
  const i=letters.indexOf(e.key);if(i>=0&&i<p.products.length)choose(p.products[i].group);
  if(e.key==="0")choose(null);
  const v={"1":"same","2":"different","3":"unsure"}[e.key];if(v)verdict(v);
});
show();
</script></body></html>"""


def cmd_review(args: argparse.Namespace) -> None:
    run = run_dir(args.run)
    manifest = load_manifest(run)
    items = review_items(run, manifest)
    if not items:
        sys.exit("Nothing to review. Run the segment step first.")
    by_id = {item["id"]: item for item in items}
    reviews_path = run / "reviews.json"
    reviews = read_json(reviews_path, {})
    public = [{"id": item["id"], "products": [{"group": p["group"]} for p in item["products"]]} for item in items]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                page = REVIEW_PAGE.replace("__ITEMS__", json.dumps(public)).replace("__REVIEWS__", json.dumps(reviews))
                self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
                return
            # Images go by position, so no path from the browser ever reaches the disk.
            match = re.fullmatch(r"/image/(\d+)/(cutout|ref(\d))", self.path)
            if not match or int(match[1]) >= len(items):
                self._send(404, "text/plain", b"not found")
                return
            item = items[int(match[1])]
            if match[2] == "cutout":
                path = run / "cutouts" / item["cutout"]
            elif int(match[3]) < len(item["products"]):
                path = run / "refs" / item["products"][int(match[3])]["ref"]
            else:
                self._send(404, "text/plain", b"not found")
                return
            self._send(200, "image/png" if path.suffix == ".png" else "image/jpeg", path.read_bytes())

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            item = by_id.get(body.get("id"))
            if self.path != "/answer" or item is None or not valid_review(body.get("answer"), item):
                self._send(400, "text/plain", b"bad request")
                return
            reviews[item["id"]] = body["answer"]
            write_json(reviews_path, reviews)
            self._send(204, "text/plain", b"")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"{sum(i['id'] in reviews for i in items)}/{len(items)} reviewed. Open http://127.0.0.1:{args.port}/ (Ctrl+C to stop; answers are saved as you go)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ── judge ───────────────────────────────────────────────────────────────────

def judge_items(run: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every search the app ran, as a cutout and its top results, for a person
    to pick the one that matches best by eye. Nothing else is shown: not the
    answer, the rank, the score or the title, and the results are shuffled so
    their order says nothing."""
    rng = random.Random(manifest["seed"] + 1)
    items = []
    for room in manifest["rooms"]:
        segments = read_json(run / "segments" / f"{room['id']}.json")
        matches = read_json(run / "matches" / f"{room['id']}.json")
        if segments is None or matches is None:
            continue
        for detection in segments["detections"]:
            found = matches.get(detection["id"])
            if not found or not found["results"]:
                continue
            results = [{"group": r["group"], "image_url": r["image_url"]} for r in found["results"][:JUDGE_TOP]]
            rng.shuffle(results)
            items.append({"id": detection["id"], "cutout": detection["cutout"] or detection["crop"], "results": results})
    rng.shuffle(items)
    return items


def valid_judgment(answer: Any, item: dict[str, Any]) -> bool:
    """The result that matches the piece best, with how well it matches, or
    {"best": null} when none of them matches or looks alike."""
    if not isinstance(answer, dict) or set(answer) != {"best", "grade"}:
        return False
    if answer["best"] is None:
        return answer["grade"] is None
    return answer["best"] in {r["group"] for r in item["results"]} and answer["grade"] in JUDGE_GRADES


JUDGE_PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8"><title>Match judging</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font:16px system-ui,sans-serif;margin:0;padding:16px;background:#f4f4f2;color:#222;text-align:center}
.cut{display:inline-block;background:#fff;border-radius:8px;padding:10px}
.cut img{width:min(300px,80vw);height:min(300px,80vw);object-fit:contain;display:block}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;max-width:1000px;margin:12px auto}
.card{background:#fff;border-radius:8px;padding:8px;border:4px solid transparent;cursor:pointer;font:inherit}
.card img{width:100%;aspect-ratio:1;object-fit:contain;display:block}
.card.on{border-color:#1d7a48}
.card:focus-visible{outline:3px solid #222}
button{font:inherit;padding:10px 18px;margin:4px;border:2px solid #aaa;border-radius:8px;background:#fff;cursor:pointer}
button.on{background:#222;color:#fff;border-color:#222}
#grade{min-height:60px}
.help{color:#555;max-width:760px;margin:6px auto}
</style></head><body><h2 id="head"></h2>
<div class="cut"><img id="cut" alt="ชิ้นที่ตัดจากห้อง"></div>
<p class="help"><b>1.</b> คลิกใบที่ตรงหรือคล้ายชิ้นด้านบน<b>ที่สุด</b> (กด 1–5) หรือกด 0 ถ้าไม่มีใบไหนตรงหรือคล้ายเลย</p>
<div class="grid" id="grid"></div>
<p><button id="none">0 ไม่มีใบไหนตรงหรือคล้ายเลย</button></p>
<div id="grade"><p class="help"><b>2.</b> ใบที่เลือกตรงแค่ไหน</p>
<button data-v="exact">e ตรงเป๊ะ (สินค้าตัวเดียวกัน)</button><button data-v="similar">s คล้าย ใช้แทนได้</button></div>
<p><button id="back">ย้อนกลับ</button></p>
<script>
const items=__ITEMS__;let saved=__JUDGMENTS__;
let at=items.findIndex(p=>!saved[p.id]);if(at<0)at=items.length;let best;
const q=s=>document.querySelector(s);
function show(){
  const done=items.filter(p=>saved[p.id]).length;
  const end=at>=items.length;
  ["#grid",".cut","#none","#grade",".help"].forEach(s=>document.querySelectorAll(s).forEach(e=>e.style.display=end?"none":""));
  if(end){q("#head").textContent="ประเมินแล้ว "+done+"/"+items.length+(done===items.length?" ครบแล้ว ปิดหน้านี้ได้":"");return}
  const p=items[at];const old=saved[p.id];best=old?old.best:undefined;
  q("#head").textContent="ชิ้นที่ "+(at+1)+"/"+items.length+" (ประเมินแล้ว "+done+")";
  q("#cut").src="/image/"+at+"/cutout";
  q("#grid").innerHTML=p.results.map((r,i)=>'<button class="card" data-i="'+i+'"><img src="'+r.image_url+'" alt="ผลค้นหาใบที่ '+(i+1)+'" referrerpolicy="no-referrer">'+(i+1)+'</button>').join("");
  q("#grid").querySelectorAll(".card").forEach(c=>c.onclick=()=>choose(p.results[+c.dataset.i].group));
  mark();
}
function mark(){
  const p=items[at];const old=saved[p.id];
  q("#grid").querySelectorAll(".card").forEach(c=>c.classList.toggle("on",best===p.results[+c.dataset.i].group));
  q("#none").classList.toggle("on",best===null);
  q("#grade").style.visibility=best?"visible":"hidden";
  document.querySelectorAll("[data-v]").forEach(b=>b.classList.toggle("on",!!old&&old.best===best&&old.grade===b.dataset.v));
}
async function save(answer){
  const p=items[at];
  const r=await fetch("/judge",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({id:p.id,answer:answer})});
  if(!r.ok){q("#head").textContent="บันทึกไม่สำเร็จ ลองอีกครั้ง";return}
  saved[p.id]=answer;at++;show();
}
function choose(group){best=group;mark();if(group===null)save({best:null,grade:null})}
function grade(v){if(best)save({best:best,grade:v})}
q("#none").onclick=()=>choose(null);
document.querySelectorAll("[data-v]").forEach(b=>b.onclick=()=>grade(b.dataset.v));
q("#back").onclick=()=>{if(at>0){at--;show()}};
addEventListener("keydown",e=>{
  if(at>=items.length)return;const p=items[at];
  const i="12345".indexOf(e.key);if(i>=0&&i<p.results.length)choose(p.results[i].group);
  if(e.key==="0")choose(null);
  if(e.key==="e")grade("exact");if(e.key==="s")grade("similar");
});
show();
</script></body></html>"""


def cmd_judge(args: argparse.Namespace) -> None:
    run = run_dir(args.run)
    manifest = load_manifest(run)
    items = judge_items(run, manifest)
    if not items:
        sys.exit("Nothing to judge. Run the match step first.")
    by_id = {item["id"]: item for item in items}
    judgments_path = run / "judgments.json"
    judgments = read_json(judgments_path, {})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                page = JUDGE_PAGE.replace("__ITEMS__", json.dumps(items)).replace("__JUDGMENTS__", json.dumps(judgments))
                self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
                return
            match = re.fullmatch(r"/image/(\d+)/cutout", self.path)
            if not match or int(match[1]) >= len(items):
                self._send(404, "text/plain", b"not found")
                return
            path = run / "cutouts" / items[int(match[1])]["cutout"]
            self._send(200, "image/png" if path.suffix == ".png" else "image/jpeg", path.read_bytes())

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            item = by_id.get(body.get("id"))
            if self.path != "/judge" or item is None or not valid_judgment(body.get("answer"), item):
                self._send(400, "text/plain", b"bad request")
                return
            judgments[item["id"]] = body["answer"]
            write_json(judgments_path, judgments)
            self._send(204, "text/plain", b"")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"{sum(i['id'] in judgments for i in items)}/{len(items)} judged. Open http://127.0.0.1:{args.port}/ (Ctrl+C to stop; answers are saved as you go)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ── typecheck ───────────────────────────────────────────────────────────────

def typecheck_items(run: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every answer of the run with what a person needs to say which catalog
    section it really belongs to: its picture, title, the catalog's section,
    the cutouts tied to it and what the detector called them."""
    reviews = read_json(run / "reviews.json", {})
    items = []
    for room in manifest["rooms"]:
        segments = read_json(run / "segments" / f"{room['id']}.json") or {"detections": []}
        for product in room["products"]:
            tied = [d for d in segments["detections"] if (reviews.get(d["id"]) or {}).get("group") == product["group"]]
            items.append(
                {
                    "group": product["group"],
                    "room": room["id"],
                    "title": product["title"],
                    "ref": product["ref"],
                    "catalog_type": product.get("catalog_type", product["type"]),
                    "cutouts": [{"file": d["cutout"] or d["crop"], "label": d["label"]} for d in tied if d["cutout"] or d["crop"]],
                }
            )
    return items


def valid_type_check(answer: Any) -> bool:
    """A furniture type the matcher knows, and a lamp subtype only for a lamp."""
    from app.services.furniture_types import FURNITURE_TYPES

    if not isinstance(answer, dict) or set(answer) != {"type", "subtype"} or answer["type"] not in FURNITURE_TYPES:
        return False
    return answer["subtype"] is None or (answer["type"] == "lamp" and answer["subtype"] in LAMP_SUBTYPES)


LAMP_SUBTYPES = ("floor", "table", "pendant", "ceiling", "wall")

TYPECHECK_PAGE = """<!doctype html><html lang="th"><head><meta charset="utf-8"><title>Type check</title>
<meta name="viewport" content="width=device-width, initial-scale=1"><style>
body{font:16px system-ui,sans-serif;margin:0;padding:16px;background:#f4f4f2;color:#222}
.wrap{max-width:980px;margin:0 auto;display:grid;gap:14px}
.row{display:flex;gap:14px;flex-wrap:wrap;align-items:flex-start}
figure{margin:0;background:#fff;border-radius:8px;padding:8px;text-align:center}
figure img{width:220px;height:220px;object-fit:contain;display:block}
figcaption{font-size:13px;color:#555;margin-top:4px;max-width:220px}
code{font:14px ui-monospace,monospace;background:#e8ebee;border-radius:4px;padding:1px 5px}
select,button{font:inherit;padding:8px 12px;border-radius:8px;border:1px solid #888;background:#fff}
button{cursor:pointer}
.panel{background:#fff;border-radius:8px;padding:12px;display:grid;gap:10px}
.title{font-weight:600}
</style></head><body><div class="wrap">
<h2 id="head"></h2>
<div class="panel"><div class="title" id="title"></div><div>หมวดใน catalog: <code id="catalog"></code></div></div>
<div class="row" id="pics"></div>
<div class="panel">
  <label>หมวดที่ถูกต้อง (furniture_type): <select id="type"></select></label>
  <label id="subrow">ชนิดย่อยของโคมไฟ: <select id="sub"><option value="">ไม่ระบุ</option></select></label>
  <div><button id="save">บันทึกและไปต่อ</button> <button id="back">ย้อนกลับ</button></div>
</div></div>
<script>
const items=__ITEMS__;let saved=__CHECKS__;const types=__TYPES__;const subs=__SUBS__;
let at=items.findIndex(p=>!saved[p.group]);if(at<0)at=items.length;
const q=s=>document.querySelector(s);
q("#type").innerHTML=types.map(t=>'<option value="'+t[0]+'">'+t[0]+' ('+t[1]+')</option>').join("");
q("#sub").innerHTML+=subs.map(s=>'<option value="'+s[0]+'">'+s[0]+' ('+s[1]+')</option>').join("");
q("#type").onchange=()=>{q("#subrow").style.display=q("#type").value==="lamp"?"":"none"};
function show(){
  const done=items.filter(p=>saved[p.group]).length;
  if(at>=items.length){q("#head").textContent="ตรวจแล้ว "+done+"/"+items.length+(done===items.length?" ครบแล้ว ปิดหน้านี้ได้":"");document.querySelectorAll(".panel,.row").forEach(e=>e.style.display="none");return}
  document.querySelectorAll(".panel,.row").forEach(e=>e.style.display="");
  const p=items[at];const old=saved[p.group];
  q("#head").textContent="สินค้าที่ "+(at+1)+"/"+items.length+" · "+p.room+" (ตรวจแล้ว "+done+")";
  q("#title").textContent=p.title;q("#catalog").textContent=p.catalog_type;
  let pics='<figure><img src="/image/'+at+'/ref" alt=""><figcaption>ภาพสินค้าใน catalog</figcaption></figure>';
  p.cutouts.forEach((c,i)=>{pics+='<figure><img src="/image/'+at+'/cut'+i+'" alt=""><figcaption>cutout จากห้อง · detector เห็นว่าเป็น <code>"'+c.label+'"</code></figcaption></figure>'});
  if(!p.cutouts.length)pics+='<figure><figcaption>ไม่มี cutout ของสินค้านี้</figcaption></figure>';
  q("#pics").innerHTML=pics;
  q("#type").value=old?old.type:p.catalog_type;q("#sub").value=old&&old.subtype?old.subtype:"";q("#type").onchange();
}
q("#save").onclick=async()=>{
  const p=items[at];const t=q("#type").value;const s=t==="lamp"&&q("#sub").value?q("#sub").value:null;
  const r=await fetch("/check",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({group:p.group,answer:{type:t,subtype:s}})});
  if(!r.ok){q("#head").textContent="บันทึกไม่สำเร็จ ลองอีกครั้ง";return}
  saved[p.group]={type:t,subtype:s};at++;show();
};
q("#back").onclick=()=>{if(at>0){at--;show()}};
show();
</script></body></html>"""

TYPE_NAMES_TH = {
    "sofa": "โซฟา", "armchair": "อาร์มแชร์", "chair": "เก้าอี้", "stool": "สตูล", "bench": "ม้านั่ง",
    "ottoman": "ออตโตมัน", "coffee_table": "โต๊ะกลาง", "dining_table": "โต๊ะทานข้าว", "side_table": "โต๊ะข้าง",
    "nightstand": "โต๊ะข้างเตียง", "desk": "โต๊ะทำงาน", "bed": "เตียง", "dresser": "ตู้ลิ้นชัก",
    "wardrobe": "ตู้เสื้อผ้า", "bookshelf": "ชั้นวางของตั้งพื้น (รวมชั้นหนังสือ)", "cabinet": "ตู้เก็บของ", "tv_bench": "ชั้นวางทีวี",
    "lamp": "โคมไฟ", "rug": "พรม", "curtain": "ม่าน", "mirror": "กระจก", "wall_shelf": "ชั้นติดผนัง",
    "plant_pot": "กระถางต้นไม้", "vase": "แจกัน", "artwork": "ภาพติดผนัง",
}
SUBTYPE_NAMES_TH = {"floor": "โคมตั้งพื้น", "table": "โคมตั้งโต๊ะ", "pendant": "โคมแขวน", "ceiling": "โคมติดเพดาน", "wall": "โคมติดผนัง"}


def cmd_typecheck(args: argparse.Namespace) -> None:
    """A person says which section each answer really belongs to; the
    catalog's own furniture_type is sometimes wrong. Saved to type_checks.json."""
    from app.services.furniture_types import FURNITURE_TYPES

    run = run_dir(args.run)
    manifest = load_manifest(run)
    items = typecheck_items(run, manifest)
    groups = {item["group"] for item in items}
    checks_path = run / "type_checks.json"
    checks = read_json(checks_path, {})
    page_data = {
        "__ITEMS__": json.dumps([{k: v for k, v in i.items() if k != "ref"} | {"cutouts": [{"label": c["label"]} for c in i["cutouts"]]} for i in items]),
        "__CHECKS__": json.dumps(checks),
        "__TYPES__": json.dumps([[t, TYPE_NAMES_TH.get(t, t)] for t in FURNITURE_TYPES]),
        "__SUBS__": json.dumps([[s, SUBTYPE_NAMES_TH[s]] for s in LAMP_SUBTYPES]),
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_: Any) -> None:
            pass

        def _send(self, status: int, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path == "/":
                page = TYPECHECK_PAGE
                for key, value in page_data.items():
                    page = page.replace(key, value)
                page = page.replace(json.dumps(checks), json.dumps(checks))
                self._send(200, "text/html; charset=utf-8", page.encode("utf-8"))
                return
            # Images go by position, so no path from the browser ever reaches the disk.
            match = re.fullmatch(r"/image/(\d+)/(ref|cut(\d+))", self.path)
            if not match or int(match[1]) >= len(items):
                self._send(404, "text/plain", b"not found")
                return
            item = items[int(match[1])]
            if match[2] == "ref":
                path = run / "refs" / item["ref"]
            elif int(match[3]) < len(item["cutouts"]):
                path = run / "cutouts" / item["cutouts"][int(match[3])]["file"]
            else:
                self._send(404, "text/plain", b"not found")
                return
            self._send(200, "image/png" if path.suffix == ".png" else "image/jpeg", path.read_bytes())

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path != "/check" or body.get("group") not in groups or not valid_type_check(body.get("answer")):
                self._send(400, "text/plain", b"bad request")
                return
            checks[body["group"]] = body["answer"]
            write_json(checks_path, checks)
            self._send(204, "text/plain", b"")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"{sum(i['group'] in checks for i in items)}/{len(items)} checked. Open http://127.0.0.1:{args.port}/ (Ctrl+C to stop; answers are saved as you go)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


# ── match ───────────────────────────────────────────────────────────────────

def search_record(result: Any) -> dict[str, Any]:
    from app.services.furniture_types import image_group_key

    return {
        "filter_level": result.filter_level,
        "results": [
            {
                "group": image_group_key(c.row.get("main_image"), c.row.get("sku")),
                "score": round(c.score, 4),
                "unique_id": str(c.row.get("unique_id")),
                "title": c.row.get("main_title") or "",
                "image_url": c.row.get("main_image"),
            }
            for c in result.candidates
        ],
    }


def cmd_match(args: argparse.Namespace) -> None:
    from product_matching import find_candidates

    require_v2_catalog()
    run = run_dir(args.run)
    manifest = load_manifest(run)
    reviews = read_json(run / "reviews.json", {})
    items = review_items(run, manifest)
    pending = [i["id"] for i in items if i["id"] not in reviews]
    if pending:
        sys.exit(f"{len(pending)}/{len(items)} pieces aren't reviewed yet. Finish the review step first: searching first would spoil it.")

    for room in manifest["rooms"]:
        segments = read_json(run / "segments" / f"{room['id']}.json")
        out = run / "matches" / f"{room['id']}.json"
        if segments is None or out.exists():
            continue
        types = {p["group"]: p["type"] for p in room["products"]}
        matched = {}
        for detection in segments["detections"]:
            group = (reviews.get(detection["id"]) or {}).get("group")
            if group is None:
                continue  # none of the products: there is no right answer to find
            # The app's order: the cutout, or the crop when there is no real mask.
            name = detection["cutout"] or detection["crop"]
            image = Image.open(run / "cutouts" / name)
            image.load()
            # As the app searches: with the detector's label. Then with the
            # right type, which leaves the label out and measures the search alone.
            system = search_record(find_candidates(image, detection["label"], min_score=0.0))
            oracle = search_record(find_candidates(image, detection["label"], min_score=0.0, furniture_type=types[group]))
            matched[detection["id"]] = {
                "query": "cutout" if detection["cutout"] else "crop",
                "filter_level": system["filter_level"],
                "results": system["results"],
                "oracle_filter_level": oracle["filter_level"],
                "oracle_results": oracle["results"],
            }
        write_json(out, matched)
        print(f"{room['id']}: searched {len(matched)} pieces")


# ── equivalents ─────────────────────────────────────────────────────────────

def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return dot / norm if norm else 0.0


def cmd_equivalents(args: argparse.Namespace) -> None:
    """How alike each search result is to the answer, as the catalog sees them:
    the same SKU, and the cosine of their two vectors in the index. The catalog
    keeps one product under several listings (another country's, a combination
    with its own SKU), so a result can be the answer under another name."""
    from app.services.pinecone_client import get_index
    from app.services.product_repo import fetch_products_by_ids

    require_v2_catalog()
    run = run_dir(args.run)
    manifest = load_manifest(run)
    reviews = read_json(run / "reviews.json", {})
    index = get_index()
    out_path = run / "equivalents.json"
    equivalents = read_json(out_path, {})

    def vectors(ids: list[str]) -> dict[str, list[float]]:
        found: dict[str, list[float]] = {}
        for start in range(0, len(ids), 100):
            for vector_id, vector in index.fetch(ids=ids[start : start + 100]).vectors.items():
                found[vector_id] = list(vector.values)
        return found

    for room in manifest["rooms"]:
        matches = read_json(run / "matches" / f"{room['id']}.json")
        if matches is None:
            continue
        products = {p["group"]: p for p in room["products"]}
        for detection_id, found in matches.items():
            if detection_id in equivalents:
                continue
            product = products[reviews[detection_id]["group"]]
            entry = {}
            for key in ("results", "oracle_results"):
                results = found[key][:EQUIVALENT_DEPTH]
                ids = [r["unique_id"] for r in results]
                rows = fetch_products_by_ids(ids)
                found_vectors = vectors(ids + [product["unique_id"]])
                answer = found_vectors[product["unique_id"]]
                entry[key.replace("results", "equivalence")] = [
                    {
                        "same_sku": canonical_sku(rows.get(r["unique_id"], {}).get("sku")) == canonical_sku(product["sku"]),
                        "similarity": round(cosine(answer, found_vectors[r["unique_id"]]), 4) if r["unique_id"] in found_vectors else None,
                    }
                    for r in results
                ]
            equivalents[detection_id] = entry
        write_json(out_path, equivalents)
        print(f"{room['id']}: done")


# ── report ──────────────────────────────────────────────────────────────────

def load_results(run: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """The manifest's rooms with their detections, reviews and search results."""
    reviews = read_json(run / "reviews.json", {})
    equivalents = read_json(run / "equivalents.json", {})
    rooms = []
    for room in manifest["rooms"]:
        segments = read_json(run / "segments" / f"{room['id']}.json")
        matches = read_json(run / "matches" / f"{room['id']}.json")
        if segments is None or matches is None:
            continue
        detections = []
        for detection in segments["detections"]:
            review = reviews.get(detection["id"]) or {}
            detections.append(
                {
                    **detection,
                    **matches.get(detection["id"], {}),
                    **equivalents.get(detection["id"], {}),
                    "group": review.get("group"),
                    "verdict": review.get("verdict"),
                }
            )
        rooms.append({**room, "detections": detections, "segment_seconds": segments.get("seconds")})
    return rooms


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def num(value: float | None) -> str:
    return "-" if value is None else f"{value:.3f}"


def with_ci(items: list[dict[str, Any]], key: str, rng: random.Random, rank_key: str = "rank") -> dict[str, Any]:
    interval = bootstrap_ci(items, lambda sample: summarize(sample, rank_key)[key], rng)
    return {"value": summarize(items, rank_key)[key], "ci95": interval}


def pct_ci(entry: dict[str, Any]) -> str:
    interval = entry["ci95"]
    return pct(entry["value"]) + (f" ({pct(interval[0])}–{pct(interval[1])})" if interval else "")


def label_confusions(items: list[dict[str, Any]]) -> Counter:
    """(product's type, label the detector gave it) for pieces labelled as
    another type: each sends the search to the wrong part of the catalog."""
    return Counter((item["type"], d["label"]) for item in items for d in item["detections"] if not d["label_ok"])


def compute_metrics(
    rooms: list[dict[str, Any]],
    manifest: dict[str, Any],
    generation: dict[str, Any],
    floor: float,
    judgments: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = random.Random(manifest["seed"])
    items = item_outcomes(rooms, floor)
    detected = detected_only(items)
    faithful = faithful_only(items)
    # The same pieces searched with their right type: the search on its own.
    search_only = faithful_only(item_outcomes(rooms, floor, "oracle_results"))
    # The labels as the user sees them, so on the app's own searches.
    points = calibration_points(items)

    # Right means the answer or the same product under another listing; the
    # strict figures (the answer's own listing only) are kept beside them.
    def block(population: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "n": len(population),
            "hit1": with_ci(population, "hit1", rng, EQUIVALENT),
            "hit5": with_ci(population, "hit5", rng, EQUIVALENT),
            "mrr": with_ci(population, "mrr", rng, EQUIVALENT),
            "strict_hit1": with_ci(population, "hit1", rng),
            "strict_hit5": with_ci(population, "hit5", rng),
            "not_in_pool": summarize(population, EQUIVALENT)["not_in_pool"],
            "line_hit5": summarize(population)["line_hit5"],
            "shown_hit1": with_ci(population, "hit1", rng, EQUIVALENT + "_with_floor"),
            "shown_hit5": with_ci(population, "hit5", rng, EQUIVALENT + "_with_floor"),
        }

    types = sorted({i["type"] for i in items})
    per_type = {}
    for type_name in types:
        of_type = [i for i in items if i["type"] == type_name]
        faithful_of_type = [i for i in faithful if i["type"] == type_name]
        tied = [d for i in of_type for d in i["detections"]]
        per_type[type_name] = {
            "n": len(of_type),
            "detected": len(detected_only(of_type)) / len(of_type),
            "label_ok": sum(d["label_ok"] for d in tied) / len(tied) if tied else None,
            "system": summarize(of_type, EQUIVALENT),
            "faithful": summarize(faithful_of_type, EQUIVALENT),
            "search_only": summarize([i for i in search_only if i["type"] == type_name], EQUIVALENT),
            "search_only_strict": summarize([i for i in search_only if i["type"] == type_name]),
        }

    tied = [d for i in items for d in i["detections"]]
    verdicts = Counter(d["verdict"] for d in tied)
    verdicts["none_of_them"] = sum(1 for r in rooms for d in r["detections"] if d.get("group") is None)
    curve = precision_curve(points, threshold_grid())
    thresholds: dict[str, Any] = {
        "current": {"high": 0.75, "medium": 0.65, "labels": label_precision(points, 0.75, 0.65)},
        "curve": curve,
        "by_type": scores_by_type(points),
        "calibrated": None,
    }
    if len(rooms) >= MIN_ROOMS_TO_CALIBRATE:
        fit_rooms, check_rooms = split_rooms((r["id"] for r in rooms), random.Random(manifest["seed"]))
        fit = [p for p in points if p["room"] in fit_rooms]
        check = [p for p in points if p["room"] in check_rooms]
        fit_curve = precision_curve(fit, threshold_grid())
        high = choose_threshold(fit_curve, TARGET_HIGH)
        medium = choose_threshold(fit_curve, TARGET_MEDIUM)
        thresholds["calibrated"] = {
            "targets": {"high": TARGET_HIGH, "medium": TARGET_MEDIUM},
            "high": high,
            "medium": medium,
            "fit_rooms": len(fit_rooms),
            "check_rooms": len(check_rooms),
            "fit_labels": label_precision(fit, high, medium),
            "check_labels": label_precision(check, high, medium),
            "current_on_check": label_precision(check, 0.75, 0.65),
        }

    seconds = [r["segment_seconds"] for r in rooms if r.get("segment_seconds") is not None]
    costs = [e.get("cost") for e in generation.values() if e.get("cost") is not None]
    return {
        "rooms": len(rooms),
        "items": len(items),
        "floor": floor,
        "detection_rate": len(detected) / len(items) if items else None,
        "items_detected_more_than_once": sum(1 for i in items if len(i["detections"]) > 1),
        "label_accuracy": sum(d["label_ok"] for d in tied) / len(tied) if tied else None,
        "verdicts": dict(verdicts),
        "filter_levels": dict(Counter(d["filter_level"] for d in tied)),
        "system": block(items),
        "detected": block(detected),
        "faithful": block(faithful),
        "search_only": block(search_only),
        "per_type": per_type,
        "by_reference": {
            kind: {
                "system": summarize([i for i in items if i["ref_kind"] == kind], EQUIVALENT),
                "search_only": summarize([i for i in search_only if i["ref_kind"] == kind], EQUIVALENT),
            }
            for kind in ("studio", "scene")
        },
        "label_confusions": [
            {"type": t, "label": l, "count": c} for (t, l), c in label_confusions(items).most_common(25)
        ],
        "thresholds": thresholds,
        "human": human_judgments(rooms, judgments or {}, threshold_grid()) if judgments else None,
        "cost": {"usd_total": sum(costs), "rooms_with_cost": len(costs), "rooms_generated": len(generation)},
        "segment_seconds_per_room": sum(seconds) / len(seconds) if seconds else None,
    }


def block_rows(metrics: dict[str, Any]) -> list[str]:
    rows = [
        "| กลุ่ม | n | Hit@1 (95% CI) | Hit@5 (95% CI) | MRR | Hit@1 เข้ม | Hit@5 เข้ม | ไม่อยู่ใน pool | Hit@5 รุ่นเดียวกัน |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    names = (
        ("system", "ทั้งระบบ: ทุกชิ้นที่วางในห้อง"),
        ("detected", "เฉพาะชิ้นที่ตรวจจับเจอ"),
        ("faithful", "เฉพาะชิ้นที่ตรวจว่าเหมือนต้นฉบับ"),
        ("search_only", "การค้นหาอย่างเดียว: ชิ้นที่เหมือน + ประเภทเฉลย"),
    )
    for key, name in names:
        b = metrics[key]
        rows.append(
            f"| {name} | {b['n']} | {pct_ci(b['hit1'])} | {pct_ci(b['hit5'])} | {num(b['mrr']['value'])} | "
            f"{pct(b['strict_hit1']['value'])} | {pct(b['strict_hit5']['value'])} | {pct(b['not_in_pool'])} | {pct(b['line_hit5'])} |"
        )
    return rows


def write_report(path: Path, name: str, metrics: dict[str, Any]) -> None:
    m = metrics
    verdicts = m["verdicts"]
    lines = [
        f"# ผลทดสอบ product matching ด้วยห้องที่ generate จากสินค้า IKEA: {name}",
        "",
        f"{m['rooms']} ห้อง, สินค้า {m['items']} ชิ้น, generate ด้วย Muse Image (OpenRouter), แยกเฟอร์นิเจอร์ด้วย Grounding DINO + SAM 2, catalog v2",
        "",
        "## ผลรวม",
        "",
        *block_rows(m),
        "",
        "Hit@1 และ Hit@5 คือสินค้าที่ถูกต้องอยู่อันดับ 1 และอยู่ใน 5 อันดับแรก ช่วง CI ได้จากการ bootstrap ตามห้อง",
        "",
        "**สินค้าที่ถูกต้อง** หมายถึงตัวเฉลย หรือตัวเดียวกันที่อยู่คนละ listing (SKU เดียวกัน หรือภาพใน index เหมือนกันจริง cosine ≥ 0.999) เพราะ catalog เก็บสินค้าเดียวกันซ้ำหลาย listing เช่น ของประเทศอื่น ส่วน **เข้ม** นับเฉพาะ listing ของเฉลยเอง และ **รุ่นเดียวกัน** นับสีหรือขนาดอื่นของรุ่นเดียวกันด้วย",
        "",
        f"ถ้าตัดผลที่ score ต่ำกว่า {m['floor']} ออกแบบที่แอปแสดงจริง ทั้งระบบได้ Hit@1 {pct_ci(m['system']['shown_hit1'])} และ Hit@5 {pct_ci(m['system']['shown_hit5'])}",
        "",
        "## พลาดตรงไหน",
        "",
        f"- **ตัดชิ้นออกมาไม่ได้:** สินค้าที่มี cutout ของตัวเองอย่างน้อยหนึ่งชิ้นคิดเป็น {pct(m['detection_rate'])} ของทั้งหมด ({m['items_detected_more_than_once']} ชิ้นถูกตัดออกมามากกว่าหนึ่งครั้ง)",
        f"- **ตั้งชื่อผิดประเภท:** ป้ายของ detector ตรงประเภทสินค้า {pct(m['label_accuracy'])} ของ cutout ที่ผูกกับสินค้า ป้ายที่ผิดทำให้ค้นในประเภทผิดของ catalog",
        f"- **ภาพเพี้ยน:** cutout ที่ผูกกับสินค้าตรวจว่าเหมือน {verdicts.get('same', 0)}, เพี้ยน {verdicts.get('different', 0)}, ไม่แน่ใจ {verdicts.get('unsure', 0)} และมีอีก {verdicts.get('none_of_them', 0)} cutout ที่ไม่ใช่สินค้าตัวไหนเลย",
        f"- **ระดับ filter ที่ค้นเจอผล:** {', '.join(f'{k}: {v}' for k, v in sorted(m['filter_levels'].items(), key=lambda kv: str(kv[0])))}",
        "",
        "ป้ายที่ detector ให้ผิดประเภท:",
        "",
        "| ประเภทสินค้า | ป้ายที่ได้ | จำนวน cutout |",
        "|---|---|---|",
        *[f"| {row['type']} | {row['label']} | {row['count']} |" for row in m["label_confusions"][:15]],
        "",
        "## แยกตามประเภทสินค้า",
        "",
        f"ทั้งระบบนับทุกชิ้น ส่วนการค้นหาอย่างเดียวนับเฉพาะชิ้นที่เหมือนและค้นด้วยประเภทเฉลย ประเภทที่มีน้อยกว่า {MIN_N_PER_TYPE} ชิ้นมีเครื่องหมาย * ตัวเลขยังไม่นิ่ง",
        "",
        "| ประเภท | n | ตัดออกมาได้ | ป้ายถูก | ทั้งระบบ Hit@1 | ทั้งระบบ Hit@5 | n ค้นหาอย่างเดียว | Hit@1 | Hit@5 | Hit@5 เข้ม | Hit@5 รุ่นเดียวกัน |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(m["per_type"].items(), key=lambda kv: (kv[1]["system"]["hit5"] or 0, kv[0]))
    for type_name, row in ordered:
        mark = "" if row["n"] >= MIN_N_PER_TYPE else "*"
        alone = row["search_only"]
        lines.append(
            f"| {type_name}{mark} | {row['n']} | {pct(row['detected'])} | {pct(row['label_ok'])} | {pct(row['system']['hit1'])} | {pct(row['system']['hit5'])} | "
            f"{alone['n']} | {pct(alone['hit1'])} | {pct(alone['hit5'])} | {pct(row['search_only_strict']['hit5'])} | {pct(alone['line_hit5'])} |"
        )

    lines += [
        "",
        "## แยกตามชนิดภาพใน catalog",
        "",
        "ภาพสินค้าเดี่ยวคือสินค้าบนพื้นหลังเรียบ ส่วนภาพทั้งห้องมีของอื่นร่วมด้วย จึงอาจทั้งทำให้ generator คัดลอกของอื่น และทำให้ vector ใน index ไม่ได้แทนตัวสินค้าโดยตรง จำแนกอัตโนมัติจากสีขอบภาพ",
        "",
        "| ชนิดภาพ | n | ทั้งระบบ Hit@1 | ทั้งระบบ Hit@5 | n ค้นหาอย่างเดียว | Hit@1 | Hit@5 |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind, name in (("studio", "สินค้าเดี่ยว"), ("scene", "ทั้งห้อง")):
        row = m["by_reference"][kind]
        lines.append(
            f"| {name} | {row['system']['n']} | {pct(row['system']['hit1'])} | {pct(row['system']['hit5'])} | "
            f"{row['search_only']['n']} | {pct(row['search_only']['hit1'])} | {pct(row['search_only']['hit5'])} |"
        )

    t = m["thresholds"]
    lines += [
        "",
        "## เกณฑ์ป้ายความมั่นใจ",
        "",
        "วัดจาก score ของผลอันดับ 1 ของการค้นหาแบบที่แอปทำ (ใช้ป้ายจาก detector) เฉพาะ cutout ที่ตรวจว่าเหมือนต้นฉบับ เพราะป้ายความมั่นใจแสดงบนผลแบบนี้",
        "",
        "ความแม่นยำของอันดับ 1 ตามป้ายด้วยเกณฑ์ปัจจุบัน (0.75 / 0.65):",
        "",
        "| ป้าย | n | แม่นยำ |",
        "|---|---|---|",
        *[f"| {k} | {v['n']} | {pct(v['precision'])} |" for k, v in t["current"]["labels"].items()],
        "",
        "ความแม่นยำเมื่อ score ถึงเกณฑ์:",
        "",
        "| เกณฑ์ | n | แม่นยำ |",
        "|---|---|---|",
        *[f"| {row['threshold']:.2f} | {row['n']} | {pct(row['precision'])} |" for row in t["curve"] if round(row["threshold"] * 100) % 5 == 0],
        "",
    ]
    calibrated = t["calibrated"]
    if calibrated is None:
        lines += [f"**ยังไม่เสนอเกณฑ์ใหม่:** มี {m['rooms']} ห้อง น้อยกว่า {MIN_ROOMS_TO_CALIBRATE} ห้องที่กำหนดไว้สำหรับการปรับเกณฑ์", ""]
    else:
        lines += [
            f"**เกณฑ์ที่เสนอ:** high = {calibrated['high']}, medium = {calibrated['medium']} (เป้า {pct(calibrated['targets']['high'])} และ {pct(calibrated['targets']['medium'])}) เลือกจาก {calibrated['fit_rooms']} ห้อง และยืนยันกับอีก {calibrated['check_rooms']} ห้อง",
            "",
            "| ป้าย | ชุดเลือก n | แม่นยำ | ชุดยืนยัน n | แม่นยำ | เกณฑ์เดิมบนชุดยืนยัน n | แม่นยำ |",
            "|---|---|---|---|---|---|---|",
        ]
        for label in ("high", "medium", "low"):
            fit, check, current = calibrated["fit_labels"][label], calibrated["check_labels"][label], calibrated["current_on_check"][label]
            lines.append(
                f"| {label} | {fit['n']} | {pct(fit['precision'])} | {check['n']} | {pct(check['precision'])} | {current['n']} | {pct(current['precision'])} |"
            )
        lines.append("")
    human = m.get("human")
    if human and human["n"]:
        ranks = human["best_rank"]
        lines += [
            "## ประเมินด้วยสายตา",
            "",
            f"ผู้ตรวจดูผล {JUDGE_TOP} อันดับแรกของการค้นหาแบบที่แอปทำ {human['n']} ครั้ง แล้วเลือกใบที่ตรงหรือคล้ายที่สุด โดยไม่เห็นเฉลย อันดับ score หรือชื่อสินค้า และสลับลำดับผลไว้",
            "",
            "| วัดอะไร | ผล |",
            "|---|---|",
            f"| มีใบที่ตรงหรือคล้ายใน {JUDGE_TOP} อันดับ (Hit@{JUDGE_TOP} ด้วยสายตา) | {pct(human['hit_useful'])} |",
            f"| มีใบที่ตรงเป๊ะใน {JUDGE_TOP} อันดับ | {pct(human['hit_exact'])} |",
            f"| ใบที่ดีที่สุดคืออันดับ 1 ของแอป | {pct(human['best_is_top1'])} |",
            f"| ใบที่ดีที่สุดคือตัวเฉลย (เฉพาะการค้นหาที่มีเฉลย) | {pct(human['best_is_answer'])} |",
            "",
            "อันดับในแอปของใบที่ผู้ตรวจเลือกว่าดีที่สุด: "
            + ", ".join(f"อันดับ {k} = {v}" for k, v in sorted(ranks.items(), key=lambda kv: int(kv[0])))
            + f", ไม่มีใบไหนตรงหรือคล้าย = {human['none']}",
            "",
            "ป้ายความมั่นใจตามเกณฑ์ปัจจุบัน เทียบกับว่าอันดับ 1 ของแอปคือใบที่ผู้ตรวจเลือก:",
            "",
            "| ป้าย | n | อันดับ 1 คือใบที่เลือก | และตรงเป๊ะ |",
            "|---|---|---|---|",
            *[f"| {k} | {v['n']} | {pct(v['useful'])} | {pct(v['exact'])} |" for k, v in human["labels"].items()],
            "",
            "อันดับ 1 คือใบที่ผู้ตรวจเลือกบ่อยแค่ไหน เมื่อ score ถึงเกณฑ์:",
            "",
            "| เกณฑ์ | n | อันดับ 1 คือใบที่เลือก | และตรงเป๊ะ |",
            "|---|---|---|---|",
            *[
                f"| {row['threshold']:.2f} | {row['n']} | {pct(row['useful'])} | {pct(row['exact'])} |"
                for row in human["curve"]
                if round(row["threshold"] * 100) % 5 == 0
            ],
            "",
        ]
    lines += [
        "score แยกตามประเภท ถ้าค่ากลางของผลถูกและผลผิดต่างกันมากในแต่ละประเภท เกณฑ์เดียวจะใช้ได้ไม่ดีเท่ากัน:",
        "",
        "| ประเภท | n | แม่นยำอันดับ 1 | ค่ากลาง score ผลถูก | ค่ากลาง score ผลผิด |",
        "|---|---|---|---|---|",
        *[
            f"| {k} | {v['n']} | {pct(v['precision'])} | {num(v['median_right'])} | {num(v['median_wrong'])} |"
            for k, v in t["by_type"].items()
        ],
        "",
        "## ค่าใช้จ่ายและเวลา",
        "",
        f"- ค่า generate: ${m['cost']['usd_total']:.4f} จาก {m['cost']['rooms_with_cost']}/{m['cost']['rooms_generated']} ห้องที่ API รายงานค่าใช้จ่าย",
        f"- แยกเฟอร์นิเจอร์บน CPU: เฉลี่ย {m['segment_seconds_per_room'] or 0:.0f} วินาทีต่อห้อง",
        "",
        "## ข้อจำกัด",
        "",
        "- **ผลมีแนวโน้มดีเกินจริง:** ภาพอ้างอิงที่ส่งให้ generator เป็นภาพเดียวกับที่ embed ไว้ใน index",
        "- **ขอบเขตของเกณฑ์:** เกณฑ์สะท้อนการหาสินค้าตัวตรงในภาพ AI ส่วนห้องจริงที่ไม่มีสินค้า IKEA ไม่มีคำตอบตรงตัว ความหมายของป้ายจึงอาจต่างกัน",
        "- **ผู้ตรวจคนเดียว:** การตัดสินว่าเหมือนหรือเพี้ยนมาจากสายตาของคนเดียว",
        "- **การผูก cutout กับสินค้า:** ผู้ตรวจเป็นคนเลือกว่า cutout เป็นสินค้าชิ้นไหน ไม่ได้ใช้ป้ายของ detector",
        "- **กลุ่มสินค้าที่สุ่ม:** ใช้เฉพาะสินค้าที่มี listing ภาษาอังกฤษ ได้ประเภทจากชื่อ และไม่ใช่อะไหล่ ชุดรวม หรือของกลางแจ้ง",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def cmd_report(args: argparse.Namespace) -> None:
    from app.core.config import settings

    run = run_dir(args.run)
    manifest = load_manifest(run)
    rooms = load_results(run, manifest)
    if not rooms:
        sys.exit("No matched rooms yet. Run the match step first.")
    generation = read_json(run / "generation.json", {})
    judgments = read_json(run / "judgments.json", {})
    metrics = compute_metrics(rooms, manifest, generation, settings.match_min_score, judgments)
    write_json(run / "metrics.json", metrics)
    write_json(run / "results.json", item_outcomes(rooms, settings.match_min_score))
    write_report(run / "report.md", args.run, metrics)
    system, alone = metrics["system"], metrics["search_only"]
    print(f"{metrics['rooms']} rooms, {metrics['items']} products, cut out {pct(metrics['detection_rate'])}, labels right {pct(metrics['label_accuracy'])}")
    print(f"System:      Hit@1 {pct_ci(system['hit1'])}  Hit@5 {pct_ci(system['hit5'])}")
    print(f"Search only: Hit@1 {pct_ci(alone['hit1'])}  Hit@5 {pct_ci(alone['hit5'])}  (n={alone['n']})")
    print(f"Wrote {run / 'report.md'}")


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    steps = parser.add_subparsers(dest="step", required=True)

    def step(name: str, handler: Any) -> argparse.ArgumentParser:
        sub = steps.add_parser(name)
        sub.add_argument("--run", required=True)
        sub.set_defaults(handler=handler)
        return sub

    sample = step("sample", cmd_sample)
    sample.add_argument("--rooms", type=int, required=True)
    sample.add_argument("--seed", type=int, required=True)
    sample.add_argument("--exclude-run", action="append", help="don't reuse the products of this run")
    prepare = step("prepare", cmd_prepare)
    prepare.add_argument("--photos", required=True, help="folder of room photos, as a user would upload them")
    prepare.add_argument("--room-types", required=True, help="comma-separated app room ids, one per photo in file-name order")
    prepare.add_argument("--seed", type=int, required=True)
    prepare.add_argument("--types", type=int, default=6, help="kinds of furniture offered per room")
    prepare.add_argument("--per-type", type=int, default=2, help="products offered per kind, the ones that suit the style best")
    prepare.add_argument("--color-weight", type=float, default=0.0, help="weight of the colour, beside the style, when ranking products")
    recompose = step("recompose", cmd_recompose)
    recompose.add_argument("--room", required=True)
    recompose.add_argument("--note", required=True, help="extra instruction for the composer, kept in the manifest")
    reprepare = step("reprepare", cmd_reprepare)
    reprepare.add_argument("--room", required=True)
    reprepare.add_argument("--per-type", type=int, default=2)
    reprepare.add_argument("--color-weight", type=float, default=1.0)
    reprepare.add_argument("--composer", action="store_true")
    reprepare.add_argument("--exclude-run", action="append")
    prepare.add_argument("--composer", action="store_true", help="also ask the app's Gemini prompt composer (sends the photo to Gemini)")
    prepare.add_argument("--exclude-run", action="append", help="don't reuse the products of this run")
    step("preflight", cmd_preflight)
    generate = step("generate", cmd_generate)
    generate.add_argument("--max-usd", type=float, required=True, help="stop once this much was spent")
    generate.add_argument("--limit", type=int, help="generate at most this many rooms now")
    generate.add_argument("--references", choices=("multi", "board"), default="multi", help="one reference per product, or all on one picture")
    step("segment", cmd_segment)
    review = step("review", cmd_review)
    review.add_argument("--port", type=int, default=8765)
    step("match", cmd_match)
    step("equivalents", cmd_equivalents)
    typecheck = step("typecheck", cmd_typecheck)
    typecheck.add_argument("--port", type=int, default=8767)
    judge = step("judge", cmd_judge)
    judge.add_argument("--port", type=int, default=8766)
    step("report", cmd_report)

    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
