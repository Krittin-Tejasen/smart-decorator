/// A catalogue product proposed as a match for one piece of furniture detected
/// in the generated room.
///
/// The fields mirror what the `ikea_furniture` table already has (main_title,
/// main_image, thb_price, category, width_cm / depth_cm / height_cm, colour,
/// material tags, in_stock, url) plus the similarity [score] product matching
/// produces, so the real matcher can fill this in without the screens changing.
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
    this.isSample = false,
  });

  /// "W 210 × D 90 × H 85 cm", or null when no dimension is known.
  String? get dimensionsLabel {
    final parts = <String>[
      if (widthCm != null) 'W ${widthCm!.round()}',
      if (depthCm != null) 'D ${depthCm!.round()}',
      if (heightCm != null) 'H ${heightCm!.round()}',
    ];
    return parts.isEmpty ? null : '${parts.join(' × ')} cm';
  }

  /// "94% match".
  String get scoreLabel => '${(score * 100).round()}% match';
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
