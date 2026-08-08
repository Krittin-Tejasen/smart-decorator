"""Thin wrapper around the Pinecone index used for product matching."""

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


def query_similar(
    vector: list[float],
    top_k: int = 5,
    metadata_filter: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    response = get_index().query(
        vector=vector,
        top_k=top_k,
        include_metadata=True,
        filter=metadata_filter,
    )
    return response.get("matches", [])
