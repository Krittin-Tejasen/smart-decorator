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

  group('FurnitureFeatures.fromJson', () {
    Map<String, dynamic> withFeatures(Map<String, dynamic> extra) {
      final json = itemJson();
      json['features'] = <String, dynamic>{
        ...(json['features'] as Map<String, dynamic>),
        ...extra,
      };
      return json;
    }

    test('reads measured colours, materials, finish, colour name and meta', () {
      final f = FurnitureItem.fromJson(withFeatures({
        'colors': [
          {'hex': '#6d422c', 'catalog_color': 'brown', 'share': 0.7, 'lab': [35.0, 15.0, 20.0]},
          {'hex': '#c3a687', 'catalog_color': 'beige', 'share': 0.3, 'lab': [70.0, 8.0, 20.0]},
        ],
        'materials': [
          {'name': 'ceramic', 'confidence': 0.95},
          {'name': 'wood', 'confidence': 0.2},
        ],
        'finish': 'matte, unglazed',
        'color_name': 'terracotta',
        'meta': {'source': 'gemini', 'usable': false},
      })).features;

      expect(f.colors.map((c) => c.catalogColor), ['brown', 'beige']);
      expect(f.colors.first.share, 0.7);
      expect(f.colors.first.hex, '#6d422c');
      expect(f.materials.map((m) => m.name), ['ceramic', 'wood']);
      expect(f.materials.first.confidence, 0.95);
      expect(f.hasMaterial, isTrue);
      expect(f.finish, 'matte, unglazed');
      expect(f.colorName, 'terracotta');
      expect(f.source, 'gemini');
      expect(f.usable, isFalse);
    });

    test('an older backend without the new fields still parses, with safe defaults', () {
      final f = FurnitureItem.fromJson(itemJson()).features;

      expect(f.colors, isEmpty);
      expect(f.materials, isEmpty);
      expect(f.hasMaterial, isFalse);
      expect(f.finish, isNull);
      expect(f.colorName, isNull);
      expect(f.source, 'pixels_only');
      expect(f.usable, isTrue);
    });

    test('malformed new fields are ignored rather than crashing the whole result', () {
      final f = FurnitureItem.fromJson(withFeatures({
        'colors': 'brown',
        'materials': [5, {'name': '', 'confidence': 1}, {'name': 'wood', 'confidence': 'x'}],
        'finish': '',
        'color_name': 7,
        'meta': 'gemini',
      })).features;

      expect(f.colors, isEmpty);
      expect(f.materials.map((m) => m.name), ['wood']);
      expect(f.materials.single.confidence, 0);
      expect(f.finish, isNull);
      expect(f.colorName, isNull);
      expect(f.source, 'pixels_only');
      expect(f.usable, isTrue);
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
