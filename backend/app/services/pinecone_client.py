"""Thin wrapper around the Pinecone index used for product matching."""

import re
from functools import lru_cache
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from app.core.config import settings


@lru_cache
def get_pinecone_client() -> Pinecone:
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is not configured.")
    return Pinecone(api_key=settings.pinecone_api_key)


@lru_cache
def get_index():
    pc = get_pinecone_client()
    if settings.pinecone_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
    return pc.Index(settings.pinecone_index_name)


def upsert_batch(vectors: list[dict[str, Any]]) -> None:
    if not vectors:
        return
    get_index().upsert(vectors=vectors)


def _canonical_sku(match_id: str) -> str:
    """Strip the photo-variant suffix and any non-digit noise (leading 'S',
    dots) from a Pinecone match id, e.g. "S299.316.00_3" -> "29931600".

    Same-SKU catalog photos (multiple angles, or the same product rescraped
    under a country-specific id variant) embed as near-identical vectors and
    can otherwise occupy several of the top-K slots for one physical product,
    crowding out genuinely different candidates. See
    pinecone-evaluation/evaluation_results_expanded.json for measured impact.
    """
    base = match_id.split("_")[0]
    return re.sub(r"\D", "", base)


def dedupe_by_sku(matches: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    """Collapse matches that resolve to the same canonical SKU, keeping the
    highest-scoring (first, since `matches` is assumed score-sorted) match
    per SKU, then return the first `top_k` of what remains."""
    seen_skus: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for match in matches:
        sku = _canonical_sku(str(match["id"]))
        if sku in seen_skus:
            continue
        seen_skus.add(sku)
        deduped.append(match)
        if len(deduped) == top_k:
            break
    return deduped


def query_similar(
    vector: list[float],
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
    dedupe: bool = True,
    overfetch_factor: int = 4,
) -> list[dict[str, Any]]:
    """Query Pinecone for the `top_k` nearest product vectors.

    When `dedupe` is True (the default), fetches `top_k * overfetch_factor`
    raw candidates and collapses same-SKU duplicates before truncating to
    `top_k`, so the returned list favors distinct products over repeated
    photos of one product. Pass `dedupe=False` to get Pinecone's raw ranking.
    """
    fetch_k = top_k * overfetch_factor if dedupe else top_k
    response = get_index().query(
        vector=vector,
        top_k=fetch_k,
        include_metadata=True,
        filter=metadata_filter,
    )
    matches = response.get("matches", [])
    if dedupe:
        return dedupe_by_sku(matches, top_k)
    return matches[:top_k]
