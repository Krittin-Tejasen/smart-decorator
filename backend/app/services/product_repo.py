"""Access to the product catalog stored in Supabase (source table: ikea_furniture)."""

from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from app.core.config import settings

TABLE_NAME = "ikea_furniture"
PAGE_SIZE = 1000


@lru_cache
def get_supabase_client() -> Client:
    if not settings.supabase_url or not settings.supabase_key:
        raise RuntimeError("SUPABASE_URL / SUPABASE_KEY are not configured.")
    return create_client(settings.supabase_url, settings.supabase_key)


def fetch_all_products() -> list[dict[str, Any]]:
    """Fetch every row from the catalog table.

    Supabase caps a single response at PAGE_SIZE rows, so this pages through
    with .range() until a short page signals the end.
    """
    client = get_supabase_client()
    rows: list[dict[str, Any]] = []
    start = 0

    while True:
        end = start + PAGE_SIZE - 1
        response = (
            client.table(TABLE_NAME).select("*").order("unique_id").range(start, end).execute()
        )
        batch = response.data
        if not batch:
            break

        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
        start += PAGE_SIZE

    return rows


def fetch_product_by_id(unique_id: str) -> dict[str, Any] | None:
    client = get_supabase_client()
    response = (
        client.table(TABLE_NAME).select("*").eq("unique_id", unique_id).limit(1).execute()
    )
    return response.data[0] if response.data else None
