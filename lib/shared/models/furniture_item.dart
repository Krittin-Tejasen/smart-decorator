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

/// Real-world size estimate for a furniture item, computed from LiDAR depth
/// + camera pose when the scan provided them. Mirrors backend/segmentation.py's
/// `FurnitureDimensions`. Null on the item when the backend couldn't compute
/// it (no spatial data sent, or not enough valid depth under the item).
class FurnitureDimensions {
  final double widthM;
  final double depthM;
  final double heightM;

  FurnitureDimensions({
    required this.widthM,
    required this.depthM,
    required this.heightM,
  });

  factory FurnitureDimensions.fromJson(Map<String, dynamic> json) {
    return FurnitureDimensions(
      widthM: (json['width_m'] as num?)?.toDouble() ?? 0,
      depthM: (json['depth_m'] as num?)?.toDouble() ?? 0,
      heightM: (json['height_m'] as num?)?.toDouble() ?? 0,
    );
  }
}

class FurnitureFeatures {
  final List<String> dominantColors;
  final double areaPct;
  final double aspectRatio;

  FurnitureFeatures({
    required this.dominantColors,
    required this.areaPct,
    required this.aspectRatio,
  });

  factory FurnitureFeatures.fromJson(Map<String, dynamic> json) {
    final colors = json['dominant_colors'];
    return FurnitureFeatures(
      dominantColors: colors is List
          ? colors.map((c) => c.toString()).toList()
          : [],
      areaPct: (json['area_pct'] as num?)?.toDouble() ?? 0,
      aspectRatio: (json['aspect_ratio'] as num?)?.toDouble() ?? 0,
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

  /// Real-world width/depth/height, when the request included spatial data
  /// and the backend could compute it. Null otherwise.
  final FurnitureDimensions? dimensionsM;

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
    this.dimensionsM,
  });

  bool get hasCutout => cutoutImage != null && cutoutImage!.isNotEmpty;

  factory FurnitureItem.fromJson(Map<String, dynamic> json) {
    final dims = json['dimensions_m'];
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
      dimensionsM: dims is Map<String, dynamic>
          ? FurnitureDimensions.fromJson(dims)
          : null,
    );
  }
}

String? _nonEmpty(Object? value) =>
    value is String && value.isNotEmpty ? value : null;

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
