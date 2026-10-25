"""Access to the product catalog stored in Supabase (source table: ikea_furniture)."""

import time
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


# What a product card needs; the catalog's long text columns (description,
# measurements, search_context...) stay in the database.
CARD_COLUMNS = (
    "unique_id,sku,main_title,main_image,category,url,in_stock,thb_price,"
    "width_cm,height_cm,depth_cm,color_extracted,material_tag,material_tags,country_code"
)


def fetch_all_products_light(retries: int = 5) -> list[dict[str, Any]]:
    """Every row with only the card columns. `select *` pulls the long text
    columns and the connection drops part-way; this does not, and a page that
    still fails is retried."""
    client = get_supabase_client()
    rows: list[dict[str, Any]] = []
    start = 0
    while True:
        for attempt in range(retries):
            try:
                batch = (
                    client.table(TABLE_NAME)
                    .select(CARD_COLUMNS)
                    .order("unique_id")
                    .range(start, start + PAGE_SIZE - 1)
                    .execute()
                    .data
                )
                break
            except Exception:  # noqa: BLE001 - network drop, retried then raised
                if attempt == retries - 1:
                    raise
                time.sleep(2 * (attempt + 1))
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def fetch_products_by_ids(unique_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Card fields for many products in one request, keyed by unique_id.
    Ids that aren't in the table are simply absent from the result."""
    if not unique_ids:
        return {}
    client = get_supabase_client()
    response = (
        client.table(TABLE_NAME).select(CARD_COLUMNS).in_("unique_id", list(unique_ids)).execute()
    )
    return {str(row["unique_id"]): row for row in response.data or []}
