import 'package:flutter_test/flutter_test.dart';
import 'package:smart_decorator/shared/models/furniture_item.dart';

Map<String, dynamic> itemJson({Object? cutout = 'omit'}) => {
      'id': '0',
      'label': 'sofa',
      'confidence': 0.75,
      'bbox': {'x_min': 0.3, 'y_min': 0.5, 'x_max': 0.7, 'y_max': 0.7},
      'crop_image': 'data:image/jpeg;base64,AAAA',
      if (cutout != 'omit') 'cutout_image': cutout,
      'mask_image': 'data:image/png;base64,BBBB',
      'mask_precise': true,
      'features': {
        'dominant_colors': ['#725f4e', '#98826e'],
        'area_pct': 5.7,
        'aspect_ratio': 2.9,
      },
    };

void main() {
  group('FurnitureItem.fromJson', () {
    test('reads the cut-out the backend sends for real masks', () {
      final item = FurnitureItem.fromJson(itemJson(cutout: 'data:image/png;base64,CCCC'));

      expect(item.hasCutout, isTrue);
      expect(item.cutoutImage, 'data:image/png;base64,CCCC');
      expect(item.cropImage, 'data:image/jpeg;base64,AAAA');
    });

    test('a null cut-out (no real mask) means "use the crop"', () {
      final item = FurnitureItem.fromJson(itemJson(cutout: null));

      expect(item.hasCutout, isFalse);
      expect(item.cutoutImage, isNull);
    });

    test('an empty-string cut-out is treated as missing, not as an image', () {
      expect(FurnitureItem.fromJson(itemJson(cutout: '')).hasCutout, isFalse);
    });

    test('older backends without the field still parse', () {
      final item = FurnitureItem.fromJson(itemJson());

      expect(item.hasCutout, isFalse);
      expect(item.label, 'sofa');
      expect(item.maskPrecise, isTrue);
      expect(item.features.dominantColors, ['#725f4e', '#98826e']);
    });
  });

  group('SegmentationResult.fromJson', () {
    Map<String, dynamic> resultJson({Object? reason = 'omit'}) => {
          'items': [itemJson()],
          'counts': {'sofa': 1},
          'total': 1,
          'method': 'grounding_dino_sam2',
          if (reason != 'omit') 'fallback_reason': reason,
        };

    test('carries why the local pipeline was skipped', () {
      final result = SegmentationResult.fromJson(
        resultJson(reason: 'TypeError: unexpected keyword argument'),
      );

      expect(result.fallbackReason, contains('TypeError'));
    });

    test('no fallback reason when the local pipeline ran', () {
      expect(SegmentationResult.fromJson(resultJson(reason: null)).fallbackReason, isNull);
      expect(SegmentationResult.fromJson(resultJson()).fallbackReason, isNull);
      expect(SegmentationResult.fromJson(resultJson()).items.single.label, 'sofa');
    });
  });
}
