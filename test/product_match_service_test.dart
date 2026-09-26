import 'package:flutter_test/flutter_test.dart';
import 'package:smart_decorator/core/services/product_match_service.dart';
import 'package:smart_decorator/shared/models/furniture_item.dart';
import 'package:smart_decorator/shared/models/product_match.dart';

FurnitureItem furniture(String label, {List<String> colors = const ['#725f4e', '#98826e', '#b9b1ac'], String id = '0'}) {
  return FurnitureItem(
    id: id,
    label: label,
    confidence: 0.8,
    bbox: BoundingBox(xMin: 0, yMin: 0, xMax: 1, yMax: 1),
    cropImage: '',
    maskImage: '',
    maskPrecise: true,
    features: FurnitureFeatures(dominantColors: colors, areaPct: 5, aspectRatio: 1),
  );
}

const service = SampleProductMatchService();

void main() {
  group('sample matches', () {
    test('three numbered products per item, most similar first', () async {
      final matches = await service.matchesFor(furniture('sofa'));

      expect(matches.map((m) => m.name), ['Sofa #1', 'Sofa #2', 'Sofa #3']);
      final scores = matches.map((m) => m.score).toList();
      expect(scores, [...scores]..sort((a, b) => b.compareTo(a)));
      expect(scores.first, lessThan(1.0));
      expect(matches.map((m) => m.scoreLabel), ['94% match', '88% match', '81% match']);
    });

    test('every one is flagged as sample data', () async {
      final matches = await service.matchesFor(furniture('armchair'));

      expect(matches.every((m) => m.isSample), isTrue);
      expect(matches.every((m) => m.imageUrl == null), isTrue);
    });

    test('the same item always gets the same products', () async {
      final a = await service.matchesFor(furniture('sofa'));
      final b = await service.matchesFor(furniture('sofa'));

      expect(a.map((m) => [m.name, m.price, m.widthCm, m.material, m.colorName]),
          b.map((m) => [m.name, m.price, m.widthCm, m.material, m.colorName]));
    });

    test('a different item gets different products', () async {
      final sofa = await service.matchesFor(furniture('sofa'));
      final bed = await service.matchesFor(furniture('bed'));

      expect(sofa.map((m) => m.price), isNot(bed.map((m) => m.price)));
      expect(bed.first.category, 'Bed');
    });

    test('count is configurable and ids are unique', () async {
      final matches = await const SampleProductMatchService(count: 5).matchesFor(furniture('rug', id: '7'));

      expect(matches, hasLength(5));
      expect(matches.map((m) => m.id).toSet(), hasLength(5));
      expect(matches.first.id, 'sample-7-1');
    });

    test('names follow the label formatting used on the cards', () async {
      final matches = await service.matchesFor(furniture('coffee_table'));

      expect(matches.first.name, 'Coffee Table #1');
    });
  });

  group('plausible numbers per kind of furniture', () {
    Future<void> expectRanges(String label, String category,
        {required double price, required double priceMax, required double widthMin, required double widthMax}) async {
      final matches = await service.matchesFor(furniture(label));
      for (final m in matches) {
        expect(m.category, category, reason: label);
        expect(m.price, inInclusiveRange(price - 10, priceMax), reason: '$label price');
        expect(m.widthCm, inInclusiveRange(widthMin, widthMax), reason: '$label width');
        expect(m.dimensionsLabel, isNotNull);
        expect(m.material, isNotEmpty);
      }
    }

    test('sofa', () => expectRanges('sofa', 'Sofa', price: 9900, priceMax: 34900, widthMin: 170, widthMax: 240));
    test('bed', () => expectRanges('bed', 'Bed', price: 12900, priceMax: 39900, widthMin: 150, widthMax: 190));
    test('armchair', () => expectRanges('armchair', 'Chair', price: 2990, priceMax: 14900, widthMin: 55, widthMax: 85));
    test('coffee table', () => expectRanges('coffee table', 'Table', price: 3490, priceMax: 19900, widthMin: 60, widthMax: 160));
    test('floor lamp', () => expectRanges('floor lamp', 'Lighting', price: 990, priceMax: 7990, widthMin: 20, widthMax: 50));
    test('rug', () => expectRanges('rug', 'Rug', price: 1990, priceMax: 15900, widthMin: 120, widthMax: 240));

    test('label matching picks the right kind, not the first word that fits', () async {
      Future<String> of(String label) async => (await service.matchesFor(furniture(label))).first.category;

      expect(await of('table lamp'), 'Lighting', reason: 'a table lamp is a lamp, not a table');
      expect(await of('bedside table'), 'Bedside table', reason: 'not a bed');
      expect(await of('nightstand'), 'Bedside table');
      expect(await of('console table'), 'Table');
      expect(await of('tv stand'), 'Storage');
      expect(await of('potted plant'), 'Plant');
      expect(await of('floating shelf'), 'Storage');
    });

    test('an unknown label still gets sensible generic products', () async {
      final matches = await service.matchesFor(furniture('zorblax'));

      expect(matches.first.category, 'Furniture');
      expect(matches.first.name, 'Zorblax #1');
      expect(matches.first.price, greaterThan(0));
    });

    test('an item without colours simply has no colour name', () async {
      final matches = await service.matchesFor(furniture('sofa', colors: const []));

      expect(matches.every((m) => m.colorName == null), isTrue);
    });

    test('prices end in 90 like a shelf label', () async {
      for (final label in ['sofa', 'vase', 'rug', 'zorblax']) {
        for (final m in await service.matchesFor(furniture(label))) {
          expect(m.price % 100, 90, reason: '$label ${m.price}');
        }
      }
    });
  });

  group('colour names', () {
    test('picks the nearest everyday name', () {
      expect(nearestColorName('#ffffff'), 'White');
      expect(nearestColorName('#000000'), 'Black');
      expect(nearestColorName('#c0714f'), 'Terracotta');
      expect(nearestColorName('#9cafe8'), isNotNull);
    });

    test('accepts hex without # and rejects junk', () {
      expect(nearestColorName('ffffff'), 'White');
      expect(nearestColorName('not a colour'), isNull);
      expect(nearestColorName('#fff'), isNull);
    });
  });

  group('formatting', () {
    test('baht with thousands separators', () {
      expect(formatBaht(12990), '฿12,990');
      expect(formatBaht(990), '฿990');
      expect(formatBaht(1000000), '฿1,000,000');
      expect(formatBaht(0), '฿0');
    });

    test('dimensions read like a shop listing', () {
      const full = ProductMatch(
        id: 'a', name: 'n', price: 1, category: 'c', score: 0.9,
        widthCm: 210.4, depthCm: 90, heightCm: 85,
      );
      const partial = ProductMatch(id: 'b', name: 'n', price: 1, category: 'c', score: 0.9, widthCm: 40);
      const none = ProductMatch(id: 'c', name: 'n', price: 1, category: 'c', score: 0.9);

      expect(full.dimensionsLabel, 'W 210 × D 90 × H 85 cm');
      expect(partial.dimensionsLabel, 'W 40 cm');
      expect(none.dimensionsLabel, isNull);
    });

    test('titleCase handles spaces, underscores and empties', () {
      expect(titleCase('coffee table'), 'Coffee Table');
      expect(titleCase('tv_stand'), 'Tv Stand');
      expect(titleCase('  '), 'Item');
    });
  });
}
