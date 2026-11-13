-- ikea_furniture_v2: the rebuilt product catalog (backend/scripts/clean_catalog.py).
--
-- A NEW table next to ikea_furniture, which is not touched: the app and the
-- backend keep reading the old one until they are pointed at this one.
--
-- Run in: Dashboard -> SQL Editor. Safe to run more than once.
-- Then load it: python -m scripts.load_catalog <catalog_clean.csv> --apply
--
-- What differs from ikea_furniture
--   * every product of the scrape is here (100,000 rows, 38,712 products), not
--     only those with a readable width and height; `has_dims` says which have a size
--   * `furniture_type` is set on every row, and `type_source` says whether it came
--     from the title or from IKEA's category
--   * `group_key` is the same for the same product on every country's site
--     (image id; sku when the image has none): collapse or dedupe by it
--   * all sizes are numeric (depth_cm was text in ikea_furniture)
--   * unique_id is "<sku>_<n>" counted over ALL rows of a sku, so ids differ from
--     ikea_furniture's: do not mix the two tables' ids (Pinecone ids included)

create table if not exists public.ikea_furniture_v2 (
  unique_id        text primary key,
  sku              text not null,
  group_key        text not null,

  main_title       text,
  description      text,
  search_context   text,
  main_image       text,
  url              text,

  category         text,
  category_id      text,
  furniture_type   text not null,
  type_source      text check (type_source in ('title', 'category')),

  country_code     text,
  currency         text,
  final_price      numeric,
  thb_price        numeric,
  in_stock         boolean,

  width_cm         numeric,
  height_cm        numeric,
  depth_cm         numeric,
  length_cm        numeric,
  diameter_cm      numeric,
  has_dims         boolean not null default false,
  size_label       text,
  size             text,

  material_tags    text,
  material_tag     text,
  is_wood          boolean,
  is_metal         boolean,
  is_fabric        boolean,
  is_plastic       boolean,
  is_glass         boolean,

  color            text,
  color_extracted  text
);

create index if not exists ikea_furniture_v2_type_idx    on public.ikea_furniture_v2 (furniture_type);
create index if not exists ikea_furniture_v2_group_idx   on public.ikea_furniture_v2 (group_key);
create index if not exists ikea_furniture_v2_sku_idx     on public.ikea_furniture_v2 (sku);
create index if not exists ikea_furniture_v2_country_idx on public.ikea_furniture_v2 (country_code);

-- Row level security on, and no policy: only the backend (secret key, which
-- bypasses RLS) can read or write it. The app does not use this table yet; when
-- it does, add a select policy like the one ikea_furniture has (see
-- supabase/inspect_policies.sql) instead of leaving it open by default.
alter table public.ikea_furniture_v2 enable row level security;
