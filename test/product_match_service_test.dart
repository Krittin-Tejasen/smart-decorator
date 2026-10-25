import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:smart_decorator/core/services/product_match_service.dart';
import 'package:smart_decorator/shared/models/furniture_item.dart';
import 'package:smart_decorator/shared/models/product_match.dart';
import 'package:smart_decorator/shared/utils/data_url.dart';

import 'support/sample_images.dart';

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

/// Answers every request with [body] and remembers what was sent.
class FakeAdapter implements HttpClientAdapter {
  FakeAdapter(this.body, {this.status = 200});
  final Map<String, dynamic> body;
  final int status;
  RequestOptions? lastRequest;

  @override
  Future<ResponseBody> fetch(RequestOptions options, Stream<Uint8List>? requestStream, Future<void>? cancelFuture) async {
    lastRequest = options;
    return ResponseBody.fromString(jsonEncode(body), status, headers: {
      Headers.contentTypeHeader: [Headers.jsonContentType],
    });
  }

  @override
  void close({bool force = false}) {}
}

Map<String, String> formFields(RequestOptions options) =>
    {for (final field in (options.data as FormData).fields) field.key: field.value};

FurnitureItem withImages(String label, {bool cutout = true, bool precise = true, bool crop = true}) => FurnitureItem(
      id: '3',
      label: label,
      confidence: 0.8,
      bbox: BoundingBox(xMin: 0, yMin: 0, xMax: 1, yMax: 1),
      cropImage: crop ? kSampleCropDataUrl : '',
      cutoutImage: cutout ? kSampleCutoutDataUrl : null,
      maskImage: '',
      maskPrecise: precise,
      features: FurnitureFeatures(dominantColors: const [], areaPct: 5, aspectRatio: 1),
    );

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

  group('backend matches', () {
    const response = {
      'furniture_type': 'sofa',
      'filter_level': 'type',
      'matches': [
        {
          'id': '1042', 'name': 'KIVIK 3-seat sofa, Tibbleby beige/grey', 'image_url': 'https://www.ikea.com/x.jpg',
          'price': 19990.0, 'category': 'Sofas', 'width_cm': 228.0, 'depth_cm': 95.0, 'height_cm': 83.0,
          'color_name': 'beige', 'material': 'fabric', 'in_stock': true,
          'url': 'https://www.ikea.com/gb/en/p/kivik-s19440533/', 'score': 0.9123, 'link_ok': true,
        },
        {'id': '7', 'name': 'Bare', 'price': null, 'score': 0.8},
      ],
    };

    BackendProductMatchService serviceWith(FakeAdapter adapter) =>
        BackendProductMatchService(dio: Dio(BaseOptions(baseUrl: 'http://test'))..httpClientAdapter = adapter);

    test('posts the item and reads the matches into ProductMatch', () async {
      final adapter = FakeAdapter(response);
      final matches = await serviceWith(adapter).matchesFor(withImages('sofa'));

      expect(adapter.lastRequest!.path, '/match-products');
      expect(formFields(adapter.lastRequest!), {'label': 'sofa', 'top_k': '5'});

      final kivik = matches.first;
      expect(kivik.name, 'KIVIK 3-seat sofa, Tibbleby beige/grey');
      expect(kivik.price, 19990);
      expect(kivik.dimensionsLabel, 'W 228 × D 95 × H 83 cm');
      expect(kivik.scoreLabel, '91% match');
      expect(kivik.url, 'https://www.ikea.com/gb/en/p/kivik-s19440533/');
      expect(kivik.isSample, isFalse);

      final bare = matches.last;
      expect(bare.price, 0);
      expect(bare.imageUrl, isNull);
      expect(bare.dimensionsLabel, isNull);
      expect(bare.inStock, isFalse);
    });

    test('sends the price range', () async {
      final adapter = FakeAdapter(response);
      await serviceWith(adapter).matchesFor(withImages('sofa'), price: const PriceRange(min: 2000, max: 10000));

      final fields = formFields(adapter.lastRequest!);
      expect(fields['min_price'], '2000.0');
      expect(fields['max_price'], '10000.0');
    });

    FurnitureItem withFeatures(FurnitureFeatures features) => FurnitureItem(
          id: '3',
          label: 'sofa',
          confidence: 0.8,
          bbox: BoundingBox(xMin: 0, yMin: 0, xMax: 1, yMax: 1),
          cropImage: kSampleCropDataUrl,
          cutoutImage: kSampleCutoutDataUrl,
          maskImage: '',
          maskPrecise: true,
          features: features,
        );

    final extracted = FurnitureFeatures(
      dominantColors: const ['#6d422c'],
      areaPct: 5,
      aspectRatio: 1,
      colors: [
        ColorSwatch(hex: '#6d422c', catalogColor: 'brown', share: 0.7),
        ColorSwatch(hex: '#c3a687', catalogColor: 'beige', share: 0.3),
      ],
      materials: [MaterialGuess(name: 'wood', confidence: 0.9), MaterialGuess(name: 'fabric', confidence: 0.4)],
      source: 'gemini',
    );

    test('sends the item colours and materials so the backend can nudge the ranking', () async {
      final adapter = FakeAdapter(response);
      await serviceWith(adapter).matchesFor(withFeatures(extracted));

      final fields = formFields(adapter.lastRequest!);
      expect(jsonDecode(fields['colors']!), [
        {'catalog_color': 'brown', 'share': 0.7},
        {'catalog_color': 'beige', 'share': 0.3},
      ]);
      expect(jsonDecode(fields['materials']!), [
        {'name': 'wood', 'confidence': 0.9},
        {'name': 'fabric', 'confidence': 0.4},
      ]);
    });

    test('sends only what was extracted: colours without materials when the model was not asked', () {
      final onlyColours = FurnitureFeatures(
        dominantColors: const [],
        areaPct: 5,
        aspectRatio: 1,
        colors: [ColorSwatch(hex: '#6d422c', catalogColor: 'brown', share: 1)],
      );
      final fields = BackendProductMatchService.featureFields(withFeatures(onlyColours));
      expect(fields.keys, ['colors']);
    });

    test('sends nothing for items with no extracted features or an unusable cut-out', () {
      expect(BackendProductMatchService.featureFields(withImages('sofa')), isEmpty);

      final unusable = FurnitureFeatures(
        dominantColors: const [],
        areaPct: 5,
        aspectRatio: 1,
        colors: extracted.colors,
        materials: extracted.materials,
        source: 'gemini',
        usable: false,
      );
      expect(BackendProductMatchService.featureFields(withFeatures(unusable)), isEmpty);
    });

    test('ignores colours without a catalog word or a share', () {
      final messy = FurnitureFeatures(
        dominantColors: const [],
        areaPct: 5,
        aspectRatio: 1,
        colors: [
          ColorSwatch(hex: '#000000', catalogColor: '', share: 0.5),
          ColorSwatch(hex: '#111111', catalogColor: 'black', share: 0),
        ],
      );
      expect(BackendProductMatchService.featureFields(withFeatures(messy)), isEmpty);
    });

    test('a response without matches is an empty list', () async {
      final matches = await serviceWith(FakeAdapter({'matches': []})).matchesFor(withImages('sofa'));
      expect(matches, isEmpty);
    });

    test('server errors surface, so the screen can offer a retry', () async {
      final service = serviceWith(FakeAdapter({'detail': 'PINECONE_API_KEY is not configured.'}, status: 503));
      expect(service.matchesFor(withImages('sofa')), throwsA(isA<DioException>()));
    });

    test('sends the cut-out when the mask is real, otherwise the crop', () {
      expect(BackendProductMatchService.imageFor(withImages('sofa')), decodeDataUrl(kSampleCutoutDataUrl));
      expect(BackendProductMatchService.imageFor(withImages('sofa', precise: false)), decodeDataUrl(kSampleCropDataUrl));
      expect(BackendProductMatchService.imageFor(withImages('sofa', cutout: false)), decodeDataUrl(kSampleCropDataUrl));
      expect(BackendProductMatchService.imageFor(withImages('sofa', cutout: false, crop: false)), isNull);
    });

    test('an item with no image cannot be matched', () {
      expect(
        BackendProductMatchService(dio: Dio()).matchesFor(withImages('sofa', cutout: false, crop: false)),
        throwsStateError,
      );
    });
  });

  group('price ranges', () {
    test('labels and bounds', () {
      expect(const PriceRange().label, 'Any price');
      expect(const PriceRange(max: 2000).label, 'Under ฿2,000');
      expect(const PriceRange(min: 2000, max: 10000).label, '฿2,000–฿10,000');
      expect(const PriceRange(min: 30000).label, 'Over ฿30,000');
      expect(const PriceRange(min: 2000, max: 10000).contains(10000), isTrue);
      expect(const PriceRange(min: 2000, max: 10000).contains(10001), isFalse);
      expect(const PriceRange(min: 1, max: 2), const PriceRange(min: 1, max: 2));
    });

    test('the sample matcher respects a range too', () async {
      final all = await service.matchesFor(furniture('sofa'));
      final cheapest = all.map((m) => m.price).reduce((a, b) => a < b ? a : b);
      final cheap = await service.matchesFor(furniture('sofa'), price: PriceRange(max: cheapest));

      expect(cheap.map((m) => m.price), [cheapest]);
      expect(all.length, greaterThan(1));
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

  group('confidence label', () {
    ProductMatch at(double score, {String? confidence}) =>
        ProductMatch(id: 'x', name: 'n', price: 1, category: 'c', score: score, confidence: confidence);

    test('comes from the backend when it sends one', () {
      expect(at(0.9, confidence: 'low').confidenceLabel, 'Weak match');
      expect(ProductMatch.fromJson({'id': '1', 'score': 0.62, 'confidence': 'medium'}).confidenceLabel, 'Possible match');
    });

    test('is worked out from the score when it is missing', () {
      expect(at(0.80).confidenceLabel, 'Strong match');
      expect(at(0.68).confidenceLabel, 'Possible match');
      expect(at(0.55).confidenceLabel, 'Weak match');
    });
  });
}
