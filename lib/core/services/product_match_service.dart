import '../../shared/models/furniture_item.dart';
import '../../shared/models/product_match.dart';

/// Finds catalogue products for a piece of furniture detected in a room.
///
/// Real matching (CLIP embedding -> Pinecone -> `ikea_furniture`) isn't merged
/// yet, so the app uses [SampleProductMatchService]; when it exists, only
/// [productMatchServiceProvider] has to change.
abstract class ProductMatchService {
  Future<List<ProductMatch>> matchesFor(FurnitureItem item);
}

/// Placeholder matches: plausible, clearly-marked (`isSample`) products so the
/// advisor can see the intended result before the real matcher exists.
///
/// Deterministic per detected item (same label + colours -> same products), so
/// the page doesn't reshuffle when you come back to it. All numbers are made up.
class SampleProductMatchService implements ProductMatchService {
  const SampleProductMatchService({this.count = 3});

  final int count;

  @override
  Future<List<ProductMatch>> matchesFor(FurnitureItem item) async {
    final label = titleCase(item.label);
    final profile = _profileFor(item.label);
    final seed = _fnv1a('${item.label}|${item.features.dominantColors.join(',')}');

    // The most similar match first, like a real ranking would be.
    final scores = [0.94, 0.88, 0.81, 0.75, 0.70];
    // Cheapest .. dearest spread so the three cards don't all look alike.
    final priceSpread = [0.45, 0.20, 0.75, 0.60, 0.35];

    return [
      for (var i = 0; i < count; i++)
        ProductMatch(
          id: 'sample-${item.id}-${i + 1}',
          name: '$label #${i + 1}',
          price: _roundPrice(
            profile.priceMin +
                (profile.priceMax - profile.priceMin) *
                    (priceSpread[i % priceSpread.length] * 0.8 + 0.2 * _unit(seed, i, 1)),
          ),
          category: profile.category,
          widthCm: _pick(profile.width, seed, i, 2),
          depthCm: _pick(profile.depth, seed, i, 3),
          heightCm: _pick(profile.height, seed, i, 4),
          colorName: _colorFor(item, i),
          material: profile.materials[(seed + i) % profile.materials.length],
          score: scores[i % scores.length],
          isSample: true,
        ),
    ];
  }

  String? _colorFor(FurnitureItem item, int i) {
    final colors = item.features.dominantColors;
    if (colors.isEmpty) return null;
    // The first match follows the item's main colour; the others its next ones.
    return nearestColorName(colors[i % colors.length]);
  }
}

/// "sofa" -> "Sofa", "coffee_table" -> "Coffee Table".
String titleCase(String raw) {
  final words = raw
      .trim()
      .split(RegExp(r'[\s_]+'))
      .where((w) => w.isNotEmpty)
      .map((w) => w[0].toUpperCase() + w.substring(1));
  final joined = words.join(' ');
  return joined.isEmpty ? 'Item' : joined;
}

// ── Colour naming ───────────────────────────────────────────────────────────

const _namedColors = <String, int>{
  'White': 0xF5F5F0,
  'Cream': 0xEDE3CF,
  'Beige': 0xD2BB98,
  'Light Grey': 0xC8C8C6,
  'Grey': 0x8E8E8C,
  'Charcoal': 0x3A3A3A,
  'Black': 0x1A1A1A,
  'Oak': 0xC8A46E,
  'Walnut': 0x6B4A2F,
  'Brown': 0x8B5A3C,
  'Terracotta': 0xC0714F,
  'Sage Green': 0x9CAF88,
  'Green': 0x4F7A45,
  'Blue': 0x5B7FA6,
  'Navy': 0x23324A,
  'Mustard': 0xD1A23A,
  'Pink': 0xE0A9A0,
};

/// The closest everyday colour name for a "#rrggbb" hex, or null if the string
/// isn't one.
String? nearestColorName(String hex) {
  final match = RegExp(r'^#?([0-9a-fA-F]{6})$').firstMatch(hex.trim());
  if (match == null) return null;
  final value = int.parse(match.group(1)!, radix: 16);
  final r = (value >> 16) & 0xFF, g = (value >> 8) & 0xFF, b = value & 0xFF;

  String? best;
  var bestDistance = 1 << 30;
  _namedColors.forEach((name, rgb) {
    final dr = r - ((rgb >> 16) & 0xFF);
    final dg = g - ((rgb >> 8) & 0xFF);
    final db = b - (rgb & 0xFF);
    final distance = dr * dr + dg * dg + db * db;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = name;
    }
  });
  return best;
}

// ── Typical products per kind of furniture (made-up ranges, THB / cm) ───────

class _Profile {
  const _Profile({
    required this.category,
    required this.priceMin,
    required this.priceMax,
    required this.width,
    required this.depth,
    required this.height,
    required this.materials,
  });

  final String category;
  final double priceMin, priceMax;
  final (double, double) width, depth, height;
  final List<String> materials;
}

_Profile _profileFor(String rawLabel) {
  final label = rawLabel.toLowerCase();
  bool has(List<String> words) => words.any(label.contains);

  if (has(['lamp', 'chandelier', 'light'])) return _lamp;
  if (has(['nightstand', 'bedside'])) return _nightstand;
  if (RegExp(r'\bbed\b').hasMatch(label) || label.contains('headboard')) return _bed;
  if (has(['sofa', 'couch'])) return _sofa;
  if (has(['chair', 'stool', 'bench', 'ottoman'])) return _seat;
  if (has(['cabinet', 'shelf', 'shelves', 'bookcase', 'wardrobe', 'dresser', 'tv stand', 'media console', 'sideboard'])) {
    return _storage;
  }
  if (has(['table', 'desk'])) return _table;
  if (has(['rug', 'carpet'])) return _rug;
  if (has(['curtain', 'blind'])) return _curtain;
  if (label.contains('mirror')) return _mirror;
  if (has(['plant', 'planter'])) return _plant;
  if (has(['painting', 'artwork', 'picture', 'frame', 'wall hanging'])) return _art;
  if (has(['vase', 'pillow', 'cushion', 'blanket', 'decor'])) return _decor;
  return _generic;
}

const _sofa = _Profile(
  category: 'Sofa', priceMin: 9900, priceMax: 34900,
  width: (170, 240), depth: (80, 100), height: (75, 90),
  materials: ['Fabric', 'Leather', 'Linen blend'],
);
const _bed = _Profile(
  category: 'Bed', priceMin: 12900, priceMax: 39900,
  width: (150, 190), depth: (200, 215), height: (90, 115),
  materials: ['Solid wood', 'Upholstered fabric'],
);
const _seat = _Profile(
  category: 'Chair', priceMin: 2990, priceMax: 14900,
  width: (55, 85), depth: (55, 85), height: (75, 95),
  materials: ['Oak', 'Fabric', 'Rattan', 'Leather'],
);
const _table = _Profile(
  category: 'Table', priceMin: 3490, priceMax: 19900,
  width: (60, 160), depth: (40, 90), height: (40, 75),
  materials: ['Oak', 'Walnut', 'Glass and metal'],
);
const _storage = _Profile(
  category: 'Storage', priceMin: 5990, priceMax: 24900,
  width: (60, 180), depth: (30, 50), height: (80, 200),
  materials: ['Oak veneer', 'Solid wood', 'Metal'],
);
const _nightstand = _Profile(
  category: 'Bedside table', priceMin: 1990, priceMax: 7900,
  width: (35, 55), depth: (30, 45), height: (40, 60),
  materials: ['Oak veneer', 'Solid wood'],
);
const _lamp = _Profile(
  category: 'Lighting', priceMin: 990, priceMax: 7990,
  width: (20, 50), depth: (20, 50), height: (35, 160),
  materials: ['Metal', 'Paper shade', 'Glass'],
);
const _rug = _Profile(
  category: 'Rug', priceMin: 1990, priceMax: 15900,
  width: (120, 240), depth: (170, 340), height: (1, 2),
  materials: ['Wool', 'Cotton', 'Jute'],
);
const _curtain = _Profile(
  category: 'Curtain', priceMin: 790, priceMax: 5990,
  width: (140, 300), depth: (1, 2), height: (220, 260),
  materials: ['Linen', 'Cotton', 'Polyester'],
);
const _mirror = _Profile(
  category: 'Mirror', priceMin: 1290, priceMax: 8990,
  width: (40, 100), depth: (3, 5), height: (60, 180),
  materials: ['Glass with oak frame', 'Glass with metal frame'],
);
const _plant = _Profile(
  category: 'Plant', priceMin: 390, priceMax: 3990,
  width: (20, 60), depth: (20, 60), height: (40, 180),
  materials: ['Ceramic pot', 'Woven basket'],
);
const _art = _Profile(
  category: 'Wall decor', priceMin: 490, priceMax: 6990,
  width: (30, 100), depth: (2, 4), height: (40, 120),
  materials: ['Framed print', 'Canvas', 'Woven'],
);
const _decor = _Profile(
  category: 'Decor', priceMin: 290, priceMax: 2490,
  width: (10, 50), depth: (10, 50), height: (10, 40),
  materials: ['Ceramic', 'Cotton', 'Linen'],
);
const _generic = _Profile(
  category: 'Furniture', priceMin: 990, priceMax: 9990,
  width: (40, 120), depth: (40, 90), height: (40, 100),
  materials: ['Wood', 'Metal', 'Fabric'],
);

// ── Deterministic "randomness" ──────────────────────────────────────────────

/// FNV-1a, so the sample data is the same on every run and platform (Dart's
/// String.hashCode makes no such promise).
int _fnv1a(String text) {
  var hash = 0x811C9DC5;
  for (final unit in text.codeUnits) {
    hash ^= unit;
    hash = (hash * 0x01000193) & 0xFFFFFFFF;
  }
  return hash;
}

/// A repeatable value in [0, 1) for (seed, index, salt).
double _unit(int seed, int index, int salt) =>
    (_fnv1a('$seed:$index:$salt') % 1000) / 1000;

double _pick((double, double) range, int seed, int index, int salt) =>
    (range.$1 + (range.$2 - range.$1) * _unit(seed, index, salt)).roundToDouble();

/// Prices end in 90, like a shelf label.
double _roundPrice(double raw) => (raw / 100).round() * 100 - 10;
