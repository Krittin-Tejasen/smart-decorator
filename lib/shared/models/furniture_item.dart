class BoundingBox {
  final double xMin;
  final double yMin;
  final double xMax;
  final double yMax;

  BoundingBox({
    required this.xMin,
    required this.yMin,
    required this.xMax,
    required this.yMax,
  });

  factory BoundingBox.fromJson(Map<String, dynamic> json) {
    return BoundingBox(
      xMin: (json['x_min'] as num?)?.toDouble() ?? 0,
      yMin: (json['y_min'] as num?)?.toDouble() ?? 0,
      xMax: (json['x_max'] as num?)?.toDouble() ?? 0,
      yMax: (json['y_max'] as num?)?.toDouble() ?? 0,
    );
  }
}

/// One measured colour of an item. [catalogColor] is one of the product
/// catalog's 12 colour words (white, beige, gray, brown …) — the key for matching.
class ColorSwatch {
  final String hex;
  final String catalogColor;

  /// Share of the item's pixels, 0-1.
  final double share;

  ColorSwatch({required this.hex, required this.catalogColor, required this.share});

  factory ColorSwatch.fromJson(Map<String, dynamic> json) => ColorSwatch(
        hex: _text(json['hex']) ?? '',
        catalogColor: _text(json['catalog_color']) ?? '',
        share: _number(json['share']),
      );
}

/// A material the model thinks the item is made of (wood, fabric, leather …).
class MaterialGuess {
  final String name;
  final double confidence;

  MaterialGuess({required this.name, required this.confidence});

  factory MaterialGuess.fromJson(Map<String, dynamic> json) => MaterialGuess(
        name: _text(json['name']) ?? '',
        confidence: _number(json['confidence']),
      );
}

class FurnitureFeatures {
  final List<String> dominantColors;
  final double areaPct;
  final double aspectRatio;

  /// Measured colours, most common first (empty for results from an older backend).
  final List<ColorSwatch> colors;

  /// Up to two materials, most likely first (empty when the backend couldn't
  /// or didn't ask the model — see [source]).
  final List<MaterialGuess> materials;

  /// e.g. "light oak, matte".
  final String? finish;

  /// The model's everyday name for the item's main colour, e.g. "terracotta".
  final String? colorName;

  /// "gemini" when the model filled in materials/finish/colour name, else
  /// "pixels_only" (colours measured, nothing more).
  final String source;

  /// False when the model says this cut-out is not a recognisable item.
  final bool usable;

  FurnitureFeatures({
    required this.dominantColors,
    required this.areaPct,
    required this.aspectRatio,
    this.colors = const [],
    this.materials = const [],
    this.finish,
    this.colorName,
    this.source = 'pixels_only',
    this.usable = true,
  });

  bool get hasMaterial => materials.isNotEmpty;

  factory FurnitureFeatures.fromJson(Map<String, dynamic> json) {
    final colors = json['dominant_colors'];
    final swatches = json['colors'];
    final materials = json['materials'];
    final meta = json['meta'];
    return FurnitureFeatures(
      dominantColors: colors is List
          ? colors.map((c) => c.toString()).toList()
          : [],
      areaPct: (json['area_pct'] as num?)?.toDouble() ?? 0,
      aspectRatio: (json['aspect_ratio'] as num?)?.toDouble() ?? 0,
      colors: swatches is List
          ? swatches
              .whereType<Map<String, dynamic>>()
              .map(ColorSwatch.fromJson)
              .toList()
          : const [],
      materials: materials is List
          ? materials
              .whereType<Map<String, dynamic>>()
              .map(MaterialGuess.fromJson)
              .where((m) => m.name.isNotEmpty)
              .toList()
          : const [],
      finish: _nonEmpty(json['finish']),
      colorName: _nonEmpty(json['color_name']),
      source: (meta is Map ? _text(meta['source']) : null) ?? 'pixels_only',
      usable: meta is Map && meta['usable'] is bool ? meta['usable'] as bool : true,
    );
  }
}

/// One piece of furniture detected in a generated room image — mirrors
/// backend/segmentation.py's `FurnitureItem`.
class FurnitureItem {
  final String id;
  final String label;
  final double confidence;
  final BoundingBox bbox;

  /// data:image/jpeg;base64,... — a thumbnail of the item's bounding box
  /// (it still contains whatever surrounds the item in the room).
  final String cropImage;

  /// data:image/png;base64,... — the item cut out of the room on a transparent
  /// background, trimmed to the item. Null when the backend had no real mask
  /// for it (then show [cropImage] instead).
  final String? cutoutImage;

  /// data:image/png;base64,... — full-size binary mask.
  final String maskImage;

  /// True when [maskImage] is a real SAM 2 mask rather than a bbox rectangle.
  final bool maskPrecise;
  final FurnitureFeatures features;

  FurnitureItem({
    required this.id,
    required this.label,
    required this.confidence,
    required this.bbox,
    required this.cropImage,
    this.cutoutImage,
    required this.maskImage,
    required this.maskPrecise,
    required this.features,
  });

  bool get hasCutout => cutoutImage != null && cutoutImage!.isNotEmpty;

  factory FurnitureItem.fromJson(Map<String, dynamic> json) {
    return FurnitureItem(
      id: json['id'] as String? ?? '',
      label: json['label'] as String? ?? 'Item',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0,
      bbox: BoundingBox.fromJson(
        json['bbox'] as Map<String, dynamic>? ?? const {},
      ),
      cropImage: json['crop_image'] as String? ?? '',
      cutoutImage: _nonEmpty(json['cutout_image']),
      maskImage: json['mask_image'] as String? ?? '',
      maskPrecise: json['mask_precise'] as bool? ?? false,
      features: FurnitureFeatures.fromJson(
        json['features'] as Map<String, dynamic>? ?? const {},
      ),
    );
  }
}

String? _nonEmpty(Object? value) =>
    value is String && value.isNotEmpty ? value : null;

// The extracted-feature fields are optional extras: a wrong type in one of
// them must not take down the whole segmentation result.
String? _text(Object? value) => _nonEmpty(value);

double _number(Object? value) => value is num ? value.toDouble() : 0;

/// Mirrors backend/segmentation.py's `SegmentationResult`.
class SegmentationResult {
  final List<FurnitureItem> items;
  final Map<String, int> counts;
  final int total;
  final String method;

  /// Why the local Grounding DINO + SAM 2 pipeline wasn't used (null when it
  /// was). Non-null means the items only have rectangular boxes.
  final String? fallbackReason;

  SegmentationResult({
    required this.items,
    required this.counts,
    required this.total,
    required this.method,
    this.fallbackReason,
  });

  factory SegmentationResult.fromJson(Map<String, dynamic> json) {
    final rawItems = json['items'];
    final rawCounts = json['counts'];

    return SegmentationResult(
      items: rawItems is List
          ? rawItems
              .map((e) => FurnitureItem.fromJson(e as Map<String, dynamic>))
              .toList()
          : [],
      counts: rawCounts is Map
          ? rawCounts.map(
              (key, value) => MapEntry(key.toString(), (value as num).toInt()),
            )
          : {},
      total: (json['total'] as num?)?.toInt() ?? 0,
      method: json['method'] as String? ?? 'none',
      fallbackReason: _nonEmpty(json['fallback_reason']),
    );
  }
}
