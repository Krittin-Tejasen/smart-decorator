"""Offline job: turn the raw IKEA scrape into the catalog CSV (replaces clean_data.ipynb).

Run from the backend/ directory:
    python -m scripts.clean_catalog "C:/path/Ikea-product-full.json" --out ../catalog_out

Differences from the notebook:
- no row is dropped for a missing size: the old `dropna(width, height)` removed
  40% of the products, among them every rug, curtain and most lamps. A product
  with no readable size is kept with `has_dims = False`;
- sizes are read by app.services.measurements (fallback chain, many languages);
- `furniture_type` is computed here, with the same code ingest_products.py uses;
- material and colour tags come from app.services.catalog_attributes.

Writes catalog_clean.csv plus three logs to --out. It does not touch Supabase or
Pinecone.
"""

import argparse
import csv
import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from app.services.catalog_attributes import extract_color_from_title, material_tags, materials_text
from app.services.category_types import fill_missing_types, one_type_per_product, types_by_category_id
from app.services.furniture_types import image_group_key, types_by_unique_id
from app.services.measurements import extract_dims

# Same symbols the notebook knew, for the currencies found in the data.
SYMBOL_TO_ISO = {
    '$': 'USD', 'USD': 'USD', '€': 'EUR', '£': 'GBP', 'CHF': 'CHF',
    'SEK': 'SEK', 'DKK': 'DKK', 'KR': 'SEK', 'THB': 'THB', '฿': 'THB',
    'CLP$': 'CLP', 'COL$': 'COP', 'KČ': 'CZK', '₪': 'ILS', '¥': 'JPY', 'RM': 'MYR',
    'ZŁ': 'PLN', '₱': 'PHP', 'LEI': 'RON', 'ДИН.': 'RSD', 'SR': 'SAR', 'د.إ': 'AED',
    '₹': 'INR', 'FT': 'HUF',
}
# Used when the currency is "undefined" (the page only gives a number).
COUNTRY_TO_ISO = {
    'au': 'AUD', 'bh': 'BHD', 'ca': 'CAD', 'eg': 'EGP', 'jo': 'JOD', 'kw': 'KWD', 'ma': 'MAD',
    'mx': 'MXN', 'no': 'NOK', 'om': 'OMR', 'qa': 'QAR', 'sg': 'SGD', 'th': 'THB', 'jp': 'JPY',
    'gb': 'GBP', 'us': 'USD', 'ch': 'CHF', 'cl': 'CLP', 'co': 'COP', 'cz': 'CZK', 'il': 'ILS',
    'my': 'MYR', 'pl': 'PLN', 'ph': 'PHP', 'ro': 'RON', 'rs': 'RSD', 'sa': 'SAR', 'ae': 'AED',
    'in': 'INR', 'hu': 'HUF', 'se': 'SEK',
}
# 1 THB buys this much of the currency. Only used when the live rates can't be fetched.
FALLBACK_RATES = {
    'USD': 0.027, 'EUR': 0.025, 'GBP': 0.021, 'SEK': 0.28, 'DKK': 0.19, 'CHF': 0.024, 'CLP': 26.5,
    'AED': 0.10, 'SAR': 0.10, 'RON': 0.125, 'COP': 116.0, 'RSD': 2.94, 'MYR': 0.125, 'CZK': 0.63,
    'PLN': 0.108, 'THB': 1.0, 'JPY': 4.2, 'ILS': 0.10, 'PHP': 1.55, 'AUD': 0.042, 'BHD': 0.0102,
    'CAD': 0.038, 'EGP': 1.36, 'JOD': 0.0192, 'KWD': 0.0083, 'MAD': 0.27, 'MXN': 0.55, 'NOK': 0.30,
    'OMR': 0.0104, 'QAR': 0.099, 'SGD': 0.036, 'INR': 2.87, 'HUF': 9.3,
}

# Country sites whose category names are English.
ENGLISH_SITES = {'gb', 'us', 'au', 'ca', 'sg', 'ie', 'in', 'my', 'ph', 'th', 'hk', 'nz'}

FIELDS = [
    'unique_id', 'sku', 'main_title', 'description', 'in_stock', 'color', 'size', 'main_image',
    'category', 'category_id', 'url', 'country_code', 'currency', 'final_price', 'thb_price',
    'width_cm', 'height_cm', 'depth_cm', 'length_cm', 'diameter_cm', 'has_dims', 'size_label',
    'material_tags', 'material_tag', 'is_wood', 'is_metal', 'is_fabric', 'is_plastic', 'is_glass',
    'color_extracted', 'furniture_type', 'type_source', 'group_key', 'search_context',
]


def fetch_rates() -> dict[str, float]:
    try:
        with urllib.request.urlopen('https://api.exchangerate-api.com/v4/latest/THB', timeout=10) as response:
            rates = json.load(response).get('rates') or {}
        if rates:
            print('Live exchange rates fetched.')
            return rates
    except Exception as exc:  # noqa: BLE001 - offline or API down: fall back, loudly
        print(f'Could not fetch live rates ({exc}); using the fallback table.')
    return FALLBACK_RATES


def to_thb(row: dict[str, Any], rates: dict[str, float], unresolved: list[dict[str, Any]]) -> float | None:
    price = row.get('final_price')
    if price is None:
        return None
    symbol = str(row.get('currency') or '').strip()
    iso = SYMBOL_TO_ISO.get(symbol) or SYMBOL_TO_ISO.get(symbol.upper())
    iso = iso or COUNTRY_TO_ISO.get(str(row.get('country_code') or '').lower())
    rate = rates.get(iso) if iso else None
    if not rate:
        unresolved.append({'sku': row.get('sku'), 'currency': symbol, 'country_code': row.get('country_code')})
        return None
    return round(float(price) / rate, 2)


def size_label(dims: dict[str, float | None]) -> str | None:
    """"80x202 cm" / "80x202x40 cm": width, height, depth in that order, leaving out
    the ones the product doesn't give (a rug is "80x150 cm": width x length).
    None without a width."""
    if dims['width_cm'] is None:
        return None
    def fmt(v: float) -> str:
        return str(int(v)) if v == int(v) else str(round(v, 1))
    parts = [fmt(dims[key]) for key in ('width_cm', 'height_cm', 'depth_cm') if dims[key] is not None]
    return 'x'.join(parts) + ' cm' if len(parts) > 1 else None


def clean_text(text: Any) -> str:
    if not isinstance(text, str):
        return ''
    return re.sub(r'\s+', ' ', re.sub(r'<.*?>', '', text)).strip()


def read_raw(path: Path) -> Iterator[dict[str, Any]]:
    """Stream the raw JSON array. The pure-python parser is used because the C one
    stops at `initial_price` values that overflow a 64-bit integer."""
    from ijson.backends import python as ijson_python

    with open(path, 'rb') as handle:
        yield from ijson_python.items(handle, 'item', use_float=True)


def clean_row(raw: dict[str, Any], unique_id: str, rates: dict[str, float], currency_log: list[dict[str, Any]]) -> dict[str, Any]:
    dims = extract_dims(raw.get('measurements'))
    title = raw.get('main_title') or ''
    description = clean_text(raw.get('description'))
    mat_text = materials_text(raw.get('materials_and_care'), raw.get('materials_description'))
    tags = material_tags(mat_text, title, description)
    color = extract_color_from_title(title)
    row = {
        'unique_id': unique_id,
        'sku': raw.get('sku'),
        'main_title': title,
        'description': description,
        'in_stock': bool(raw.get('in_stock')),
        'color': raw.get('color'),
        'size': raw.get('size'),
        'main_image': raw.get('main_image'),
        'category': raw.get('category'),
        'category_id': raw.get('category_id'),
        'url': raw.get('url'),
        'country_code': raw.get('country_code'),
        'currency': raw.get('currency'),
        'final_price': raw.get('final_price'),
        **dims,
        'has_dims': any(v is not None for v in dims.values()),
        'size_label': size_label(dims),
        'material_tags': tags,
        'material_tag': tags.split('|')[0],
        **{f'is_{name.lower()}': name in tags.split('|') for name in ('Wood', 'Metal', 'Fabric', 'Plastic', 'Glass')},
        'color_extracted': color,
    }
    row['thb_price'] = to_thb(raw, rates, currency_log)
    row['search_context'] = ' '.join(
        filter(None, [title, raw.get('category') or '', row['material_tag'], color or '', description])
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('raw_json', type=Path)
    parser.add_argument('--out', type=Path, default=Path('catalog_out'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rates = fetch_rates()
    seen: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    currency_log: list[dict[str, Any]] = []
    unread_names: Counter[str] = Counter()
    # category_id -> {country: path}. The id is the same on every country's site, so what
    # the paths of an id say names the category of its Arabic or Hebrew products too.
    category_paths: dict[str, dict[str, list[str]]] = {}

    for index, raw in enumerate(read_raw(args.raw_json), start=1):
        seen[str(raw.get('sku'))] += 1
        row = clean_row(raw, f"{raw.get('sku')}_{seen[str(raw.get('sku'))]}", rates, currency_log)
        if not row['has_dims']:
            # The names of the measurements of a product we can't size: what to teach the reader next.
            for item in raw.get('measurements') or []:
                if isinstance(item, dict) and item.get('name'):
                    unread_names[str(item['name']).strip().lower()] += 1
        if raw.get('category_tree'):
            by_country = category_paths.setdefault(str(raw.get('category_id')), {})
            by_country.setdefault(str(raw.get('country_code')).lower(), [c.get('name') for c in raw['category_tree']])
        rows.append(row)
        if index % 10000 == 0:
            print(f'{index} rows read...')

    # Title first (it reads the product's own words), then the category fills what
    # the title could not name, so no product is left without a type.
    title_types = types_by_unique_id(rows)
    for row in rows:
        # The same product on every country's site shares this key: dedupe or collapse by it.
        row['group_key'] = image_group_key(row['main_image'], row['sku'])
    types = fill_missing_types(title_types, rows, types_by_category_id(category_paths, ENGLISH_SITES))
    types = one_type_per_product(types, rows)
    # Every product says what it is. Fail loudly rather than write a catalog with a gap.
    unnamed = [r for r in rows if types.get(r['unique_id'], 'other') == 'other']
    if unnamed:
        raise SystemExit(
            f"{len(unnamed)} rows have no furniture_type, e.g. "
            + ', '.join(f"{r['unique_id']} ({r['main_title'][:40]})" for r in unnamed[:5])
        )
    for row in rows:
        row['furniture_type'] = types[row['unique_id']]
        row['type_source'] = 'title' if title_types.get(row['unique_id'], 'other') != 'other' else 'category'

    with open(args.out / 'catalog_clean.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    with open(args.out / 'error_currency_log.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=['sku', 'currency', 'country_code'])
        writer.writeheader()
        writer.writerows(currency_log)
    with open(args.out / 'unread_measurement_names.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['name', 'rows_without_size'])
        writer.writerows(unread_names.most_common())

    sized = sum(1 for r in rows if r['has_dims'])
    typed = sum(1 for r in rows if r['furniture_type'] != 'household')
    print(f'{len(rows)} rows kept (none dropped). With any size: {sized} ({sized / len(rows):.1%}).')
    print(f'furniture_type set on every row; {len(rows) - typed} are the general "household" type. Unresolved currency: {len(currency_log)} rows.')
    print(f'Output in {args.out.resolve()}')


if __name__ == '__main__':
    main()
