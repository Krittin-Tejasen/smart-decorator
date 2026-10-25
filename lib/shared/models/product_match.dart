/// A catalogue product proposed as a match for one piece of furniture detected
/// in the generated room.
///
/// The fields mirror what the `ikea_furniture` table has (main_title,
/// main_image, thb_price, category, width_cm / depth_cm / height_cm, colour,
/// material, in_stock, url) plus the image similarity [score] the matcher
/// gives it. Built from the backend's POST /match-products with [fromJson].
class ProductMatch {
  final String id;
  final String name;

  /// Product photo URL; null shows a category icon instead.
  final String? imageUrl;

  /// Price in THB.
  final double price;
  final String category;

  final double? widthCm;
  final double? depthCm;
  final double? heightCm;

  final String? colorName;
  final String? material;
  final bool inStock;
  final String? url;

  /// Visual similarity to the detected item, 0..1.
  final double score;

  /// How sure the picture match is: "high", "medium" or "low". The backend
  /// sends it; without it (sample data) it is worked out from [score] with the
  /// same cut-offs.
  final String confidence;

  /// True for placeholder data shown before real product matching exists. The
  /// UI must label it as such so nobody mistakes it for a real result.
  final bool isSample;

  const ProductMatch({
    required this.id,
    required this.name,
    this.imageUrl,
    required this.price,
    required this.category,
    this.widthCm,
    this.depthCm,
    this.heightCm,
    this.colorName,
    this.material,
    this.inStock = true,
    this.url,
    required this.score,
    String? confidence,
    this.isSample = false,
  }) : confidence = confidence ?? (score >= 0.75 ? 'high' : (score >= 0.65 ? 'medium' : 'low'));

  /// One entry of `matches` in the /match-products response.
  factory ProductMatch.fromJson(Map<String, dynamic> json) {
    double? number(String key) => (json[key] as num?)?.toDouble();
    String? text(String key) {
      final value = json[key];
      return value is String && value.trim().isNotEmpty ? value : null;
    }

    return ProductMatch(
      id: json['id']?.toString() ?? '',
      name: text('name') ?? 'Product',
      imageUrl: text('image_url'),
      price: number('price') ?? 0,
      category: text('category') ?? '',
      widthCm: number('width_cm'),
      depthCm: number('depth_cm'),
      heightCm: number('height_cm'),
      colorName: text('color_name'),
      material: text('material'),
      inStock: json['in_stock'] as bool? ?? false,
      url: text('url'),
      score: number('score') ?? 0,
      confidence: text('confidence'),
    );
  }

  /// "W 210 × D 90 × H 85 cm", or null when no dimension is known.
  String? get dimensionsLabel {
    final parts = <String>[
      if (widthCm != null) 'W ${widthCm!.round()}',
      if (depthCm != null) 'D ${depthCm!.round()}',
      if (heightCm != null) 'H ${heightCm!.round()}',
    ];
    return parts.isEmpty ? null : '${parts.join(' × ')} cm';
  }

  /// "Strong match", "Possible match" or "Weak match".
  String get confidenceLabel => switch (confidence) {
        'high' => 'Strong match',
        'medium' => 'Possible match',
        _ => 'Weak match',
      };

  /// "94% match".
  String get scoreLabel => '${(score * 100).round()}% match';
}

/// A budget for matches, in THB. Either end may be open.
class PriceRange {
  final double? min;
  final double? max;

  const PriceRange({this.min, this.max});

  /// Whether [price] falls inside the range.
  bool contains(double price) => (min == null || price >= min!) && (max == null || price <= max!);

  @override
  bool operator ==(Object other) => other is PriceRange && other.min == min && other.max == max;

  @override
  int get hashCode => Object.hash(min, max);

  /// "Under ฿2,000", "฿2,000–฿10,000", "Over ฿30,000".
  String get label {
    if (min == null && max == null) return 'Any price';
    if (min == null) return 'Under ${formatBaht(max!)}';
    if (max == null) return 'Over ${formatBaht(min!)}';
    return '${formatBaht(min!)}–${formatBaht(max!)}';
  }
}

/// "฿12,990" — whole baht with thousands separators.
String formatBaht(num amount) {
  final digits = amount.round().abs().toString();
  final buffer = StringBuffer();
  for (var i = 0; i < digits.length; i++) {
    if (i > 0 && (digits.length - i) % 3 == 0) buffer.write(',');
    buffer.write(digits[i]);
  }
  return '${amount < 0 ? '-' : ''}฿$buffer';
}
