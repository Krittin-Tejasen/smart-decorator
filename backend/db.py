"""
db.py
─────
Supabase persistence for generated room designs.

Every successful /generate-room call inserts one row into `room_designs`
(room type/style/color + paths to the uploaded and generated images) plus
one row per detected furniture item into `design_furniture`, linked by
design_id. Every new row starts with is_saved=false — the Flutter client
flips it to true when the user taps the favorite/save button on the
results screen, and that flag is what the history screen actually reads.
Rows that are never favorited just sit there unused; nothing deletes them.

Uses the Supabase *service-role* key (SUPABASE_SECRET_KEY) since this only
ever runs server-side. Never send that key to the Flutter client — it
bypasses row-level security entirely.

Required Supabase schema (create/extend via the SQL editor):

    alter table room_designs add column if not exists style text;
    alter table room_designs add column if not exists color text;
    alter table room_designs add column if not exists room_size jsonb;
    alter table room_designs add column if not exists is_saved boolean not null default false;

    create table if not exists design_furniture (
        id uuid primary key default gen_random_uuid(),
        design_id uuid not null references room_designs(id) on delete cascade,
        label text,
        confidence float8,
        bbox jsonb,
        features jsonb,
        crop_image text,
        mask_image text,
        mask_precise boolean,
        created_at timestamptz not null default now()
    );

Env vars
────────
  SUPABASE_URL           Supabase project URL
  SUPABASE_SECRET_KEY    Service-role key (server-side only, never shipped to the app)
"""

import os
import time
from typing import Any

from supabase import Client, create_client

_STORAGE_BUCKET = "room-images"

_client: Client | None = None
_client_checked = False


def _get_client() -> Client | None:
    global _client, _client_checked
    if _client_checked:
        return _client

    _client_checked = True
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SECRET_KEY")
    if not url or not key:
        print("[db] SUPABASE_URL/SUPABASE_SECRET_KEY not configured — generations won't be persisted.")
        return None

    _client = create_client(url, key)
    return _client


def save_generated_design(
    room_type: str,
    style: str,
    color: str,
    source_bytes: bytes,
    source_mime: str,
    generated_bytes: bytes,
    furniture_items: list[dict[str, Any]],
) -> str | None:
    """
    Uploads the source/generated images to Storage, inserts one row into
    room_designs (is_saved=false) and one row per furniture item into
    design_furniture. Returns the new design's id, or None if Supabase
    isn't configured or the write fails — a persistence hiccup should
    never break the /generate-room response itself.

    Synchronous (blocking network calls) — call via asyncio.to_thread from
    the async route handler so it doesn't block the event loop.
    """
    client = _get_client()
    if client is None:
        return None

    try:
        ts = int(time.time() * 1000)
        source_ext = "png" if "png" in source_mime else "jpg"
        source_path = f"designs/{ts}/source.{source_ext}"
        generated_path = f"designs/{ts}/generated.png"

        client.storage.from_(_STORAGE_BUCKET).upload(
            source_path, source_bytes, {"content-type": source_mime}
        )
        client.storage.from_(_STORAGE_BUCKET).upload(
            generated_path, generated_bytes, {"content-type": "image/png"}
        )

        design_rows = (
            client.table("room_designs")
            .insert(
                {
                    "room_type": room_type,
                    "style": style,
                    "color": color,
                    "source_image_path": source_path,
                    "generated_image_path": generated_path,
                    "room_size": None,  # not wired up yet — LiDAR scan flow is separate for now
                    "is_saved": False,
                }
            )
            .execute()
            .data
        )
        design_id = design_rows[0]["id"]

        if furniture_items:
            furniture_rows = [
                {
                    "design_id": design_id,
                    "label": item.get("label"),
                    "confidence": item.get("confidence"),
                    "bbox": item.get("bbox"),
                    "features": item.get("features"),
                    "crop_image": item.get("crop_image"),
                    "mask_image": item.get("mask_image"),
                    "mask_precise": item.get("mask_precise"),
                }
                for item in furniture_items
            ]
            client.table("design_furniture").insert(furniture_rows).execute()

        return design_id
    except Exception as exc:
        print(f"[db] failed to save generated design: {exc}")
        return None
