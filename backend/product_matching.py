"""
POST /match-products
  Finds real IKEA products for one piece of furniture detected in a generated
  room (a `FurnitureItem` from /segment-furniture).

Pipeline (choices follow pinecone-evaluation/EVALUATION_REPORT_EXPANDED_TH.md):

    cutout image ─► CLIP embedding (app.services.embedding, same model and
                    white-background preprocessing the catalog was embedded with)
                 ─► Pinecone query, filtered by furniture_type (+ price range)
                 ─► catalog rows from Supabase ─► one card per product
                    (collapse rows that share an image id)
                 ─► size as a soft re-rank, never a hard filter
                 ─► drop matches below MATCH_MIN_SCORE ─► drop dead links

- Filtering by type raised Hit@5 from 53.5% to 68.0% on room-like queries, so
  a label that maps to a furniture type is searched only within that type; an
  empty answer means the catalog has nothing close (it has no rugs at all).
  Image-only search is used only for labels with no type. `filter_level` in
  the response says which was used. A user's price range is never relaxed.
- A width filter helped only while the width was exact (84.5%) and fell to
  ~24% (worse than no filter) with a 10% error, so width only nudges ranking.
- Product photos are IKEA's own URLs and the cutout arrives in the request:
  nothing here reads or writes object storage.
"""

import asyncio
import time
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import httpx
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from app.core.config import settings
from app.services.furniture_types import FURNITURE_TYPES, image_group_key, label_to_type

router = APIRouter(tags=["product-matching"])

# Candidates fetched from Pinecone per filter level. One product often has many
# rows (one per country/language/size), so this is several times top_k to
# leave enough distinct products after collapsing them.
_CANDIDATE_POOL = 60

# Width within ±20% of the target costs nothing; further off costs up to
# _SIZE_PENALTY_MAX of the ranking score (cosine scores are ~0.6–1.0).
_SIZE_TOLERANCE = 0.20
_SIZE_PENALTY_MAX = 0.10

# The same IKEA product appears once per country. Show the English listing
# when there is one with (practically) the same score.
_PREFERRED_COUNTRIES = ("gb", "us")


# ── Filters ─────────────────────────────────────────────────────────────────

@dataclass
class FilterLevel:
    name: str
    filter: dict[str, Any] | None


def build_filter_ladder(
    furniture_type: str | None,
    min_price: float | None = None,
    max_price: float | None = None,
    in_stock_only: bool = False,
) -> list[FilterLevel]:
    """Filters to try, strictest first. The price range is part of every
    level: it is the user's budget, not a hint.

    A label with a furniture type is only ever searched within that type. An
    image-only search is a fallback for labels the catalog can't be filtered by
    ("decoration"): for a typed label it returned products of other types with
    high scores (a lamp matched drawer fronts at 0.86), which is worse than
    saying nothing was found."""
    price: dict[str, float] = {}
    if min_price is not None:
        price["$gte"] = float(min_price)
    if max_price is not None:
        price["$lte"] = float(max_price)
    base: dict[str, Any] = {"thb_price": price} if price else {}

    levels: list[FilterLevel] = []
    if furniture_type:
        typed = {**base, "furniture_type": {"$eq": furniture_type}}
        if in_stock_only:
            levels.append(FilterLevel("type+stock", {**typed, "in_stock": {"$eq": True}}))
        levels.append(FilterLevel("type", typed))
        return levels
    if in_stock_only:
        levels.append(FilterLevel("stock", {**base, "in_stock": {"$eq": True}}))
    levels.append(FilterLevel("image_only", base or None))
    return levels


# ── Ranking ─────────────────────────────────────────────────────────────────

def _as_float(value: Any) -> float | None:
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


def size_penalty(target_width_cm: float | None, product_width_cm: Any) -> float:
    width = _as_float(product_width_cm)
    if not target_width_cm or not width:
        return 0.0
    relative_error = abs(width - target_width_cm) / target_width_cm
    excess = max(relative_error - _SIZE_TOLERANCE, 0.0) / _SIZE_TOLERANCE
    return _SIZE_PENALTY_MAX * min(excess, 1.0)


@dataclass
class Candidate:
    row: dict[str, Any]
    score: float  # cosine similarity of the images, what the threshold applies to
    rank_score: float = field(init=False)
    target_width_cm: float | None = None

    def __post_init__(self) -> None:
        self.rank_score = self.score - size_penalty(self.target_width_cm, self.row.get("width_cm"))


def collapse_products(
    matches: list[dict[str, Any]],
    rows: dict[str, dict[str, Any]],
    target_width_cm: float | None = None,
) -> list[Candidate]:
    """One candidate per product (image group), best ranking first.

    `matches` are Pinecone matches ({"id", "score"}); ids without a catalog
    row are skipped."""
    best: dict[str, Candidate] = {}
    for match in matches:
        row = rows.get(str(match["id"]))
        if row is None:
            continue
        candidate = Candidate(row=row, score=float(match["score"]), target_width_cm=target_width_cm)
        key = image_group_key(row.get("main_image"), row.get("sku"))
        current = best.get(key)
        if current is None or _better(candidate, current):
            best[key] = candidate
    return sorted(best.values(), key=lambda c: c.rank_score, reverse=True)


def _better(a: Candidate, b: Candidate) -> bool:
    if round(a.score, 3) != round(b.score, 3):
        return a.score > b.score
    return _country_rank(a.row) < _country_rank(b.row)


def _country_rank(row: dict[str, Any]) -> int:
    country = (row.get("country_code") or "").lower()
    return _PREFERRED_COUNTRIES.index(country) if country in _PREFERRED_COUNTRIES else len(_PREFERRED_COUNTRIES)


def to_product_match(candidate: Candidate, link_ok: bool | None) -> dict[str, Any]:
    """The JSON the app reads into its ProductMatch model."""
    row = candidate.row
    return {
        "id": str(row.get("unique_id")),
        "name": row.get("main_title") or "",
        "image_url": row.get("main_image") or None,
        "price": _as_float(row.get("thb_price")),
        "category": row.get("category") or "",
        "width_cm": _as_float(row.get("width_cm")),
        "depth_cm": _as_float(row.get("depth_cm")),
        "height_cm": _as_float(row.get("height_cm")),
        "color_name": row.get("color_extracted") or None,
        "material": row.get("material_tag") or None,
        "in_stock": bool(row.get("in_stock")),
        "url": row.get("url") or None,
        "country_code": row.get("country_code") or None,
        "score": round(candidate.score, 4),
        "link_ok": link_ok,
    }


# ── Reference-link check ────────────────────────────────────────────────────

_LINK_CACHE_TTL = 24 * 3600
_link_cache: dict[str, tuple[bool | None, float]] = {}


async def check_link(client: httpx.AsyncClient, url: str) -> bool | None:
    """True if the product page answers, False if it is gone (404/410), None
    when it can't be told (timeouts, or the shop refusing automated requests),
    in which case the product is still shown."""
    cached = _link_cache.get(url)
    if cached and time.monotonic() - cached[1] < _LINK_CACHE_TTL:
        return cached[0]
    try:
        response = await client.head(url, follow_redirects=True)
        if response.status_code in (404, 410):
            ok: bool | None = False
        elif response.status_code < 400:
            ok = True
        else:
            ok = None
    except httpx.HTTPError:
        ok = None
    _link_cache[url] = (ok, time.monotonic())
    return ok


async def pick_with_live_links(candidates: list[Candidate], top_k: int) -> list[dict[str, Any]]:
    """The first `top_k` candidates whose link isn't known to be dead, checking
    links a batch at a time so a dead one is replaced by the next candidate."""
    picked: list[dict[str, Any]] = []
    position = 0
    async with httpx.AsyncClient(timeout=4.0, headers={"User-Agent": "SmartDecorator/1.0"}) as client:
        while len(picked) < top_k and position < len(candidates):
            batch = candidates[position : position + (top_k - len(picked))]
            position += len(batch)
            results = await asyncio.gather(*(check_link(client, c.row.get("url") or "") for c in batch))
            for candidate, ok in zip(batch, results):
                if ok is not False:
                    picked.append(to_product_match(candidate, ok))
    return picked


# ── Matching ────────────────────────────────────────────────────────────────

@dataclass
class MatchResult:
    furniture_type: str | None
    filter_level: str
    candidates: list[Candidate]  # above the threshold, best first
    below_threshold: int


def find_candidates(
    image: Image.Image,
    label: str,
    min_price: float | None = None,
    max_price: float | None = None,
    target_width_cm: float | None = None,
    in_stock_only: bool = False,
    top_k: int = 5,
    min_score: float | None = None,
    furniture_type: str | None = None,
) -> MatchResult:
    """Blocking part of a match (CLIP, Pinecone, Supabase); run it in a thread."""
    # Imported here so the app starts (and other endpoints work) even before
    # the matching dependencies/keys are set up on a machine.
    from app.services.embedding import encode_image
    from app.services.pinecone_client import query_similar
    from app.services.product_repo import fetch_products_by_ids

    threshold = settings.match_min_score if min_score is None else min_score
    # A type the user chose beats the one guessed from the label.
    furniture_type = furniture_type or label_to_type(label)
    vector = encode_image(image)

    levels = build_filter_ladder(furniture_type, min_price, max_price, in_stock_only)
    result = MatchResult(furniture_type, levels[-1].name, [], 0)
    for level in levels:
        matches = query_similar(vector, top_k=_CANDIDATE_POOL, metadata_filter=level.filter, dedupe=False)
        rows = fetch_products_by_ids([str(m["id"]) for m in matches])
        collapsed = collapse_products(matches, rows, target_width_cm)
        passing = [c for c in collapsed if c.score >= threshold]
        result = MatchResult(furniture_type, level.name, passing, len(collapsed) - len(passing))
        # Relax only when a level found nothing: two products of the right
        # type are better than five padded out with other types.
        if passing:
            break
    return result


@router.post("/match-products")
async def match_products(
    image: UploadFile = File(..., description="The item's cutout (PNG with transparency) or its crop."),
    label: str = Form(..., description="Segmentation label, e.g. 'armchair'."),
    min_price: float | None = Form(None, ge=0),
    max_price: float | None = Form(None, ge=0),
    target_width_cm: float | None = Form(None, gt=0),
    in_stock_only: bool = Form(False),
    top_k: int = Form(5, ge=1, le=20),
    furniture_type: str | None = Form(None, description="A type the user picked; used instead of the one guessed from the label."),
):
    if min_price is not None and max_price is not None and min_price > max_price:
        raise HTTPException(status_code=422, detail="min_price is greater than max_price.")
    if furniture_type is not None and furniture_type not in FURNITURE_TYPES:
        raise HTTPException(status_code=422, detail=f"Unknown furniture_type. Choose one of: {', '.join(FURNITURE_TYPES)}.")
    # A label the catalog can't be filtered by ("window", "pillows") is not
    # searched by picture alone: that returned unrelated products (a window
    # matched kitchen cabinets at 0.78). The app asks the user for a type.
    if furniture_type is None and label_to_type(label) is None:
        return {
            "label": label,
            "furniture_type": None,
            "filter_level": "no_type",
            "threshold": settings.match_min_score,
            "below_threshold": 0,
            "matches": [],
            "needs_type": True,
            "available_types": list(FURNITURE_TYPES),
        }
    try:
        picture = Image.open(BytesIO(await image.read()))
        picture.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    try:
        result = await asyncio.to_thread(
            find_candidates,
            picture,
            label,
            min_price=min_price,
            max_price=max_price,
            target_width_cm=target_width_cm,
            in_stock_only=in_stock_only,
            top_k=top_k,
            furniture_type=furniture_type,
        )
    except RuntimeError as exc:  # missing PINECONE_API_KEY / SUPABASE_* on this machine
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "label": label,
        "furniture_type": result.furniture_type,
        "filter_level": result.filter_level,
        "threshold": settings.match_min_score,
        "below_threshold": result.below_threshold,
        "matches": await pick_with_live_links(result.candidates, top_k),
        "needs_type": False,
    }
