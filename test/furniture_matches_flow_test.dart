import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:smart_decorator/core/services/product_match_service.dart';
import 'package:smart_decorator/features/results/presentation/furniture_matches_screen.dart';
import 'package:smart_decorator/features/results/presentation/results_screen.dart';
import 'package:smart_decorator/features/results/providers/furniture_matches_provider.dart';
import 'package:smart_decorator/features/results/widgets/furniture_image.dart';
import 'package:smart_decorator/features/results/widgets/furniture_segment_card.dart';
import 'package:smart_decorator/features/results/widgets/product_match_card.dart';
import 'package:smart_decorator/routes/app_router.dart';
import 'package:smart_decorator/shared/models/furniture_item.dart';
import 'package:smart_decorator/shared/models/product_match.dart';
import 'package:smart_decorator/shared/providers/app_state_provider.dart';

import 'support/sample_images.dart';

FurnitureItem furniture(
  String label, {
  String id = '0',
  bool cutout = true,
  bool crop = true,
  List<String> colors = const ['#725f4e', '#98826e'],
  FurnitureFeatures? features,
}) {
  return FurnitureItem(
    id: id,
    label: label,
    confidence: 0.8,
    bbox: BoundingBox(xMin: 0, yMin: 0, xMax: 1, yMax: 1),
    cropImage: crop ? kSampleCropDataUrl : '',
    cutoutImage: cutout ? kSampleCutoutDataUrl : null,
    maskImage: '',
    maskPrecise: cutout,
    features: features ?? FurnitureFeatures(dominantColors: colors, areaPct: 5, aspectRatio: 1),
  );
}

/// An app state that already holds a generated room's furniture.
class FakeAppState extends AppStateNotifier {
  FakeAppState(List<FurnitureItem> items) {
    state = state.copyWith(segmentedFurniture: items);
  }
}

class ControlledService implements ProductMatchService {
  ControlledService(this.onCall);
  final Future<List<ProductMatch>> Function(FurnitureItem item, int callNumber) onCall;
  int calls = 0;

  /// The `use` of every call, in order.
  final uses = <FeatureUse>[];

  @override
  Future<List<ProductMatch>> matchesFor(
    FurnitureItem item, {
    PriceRange? price,
    FeatureUse use = FeatureUse.none,
  }) {
    uses.add(use);
    return onCall(item, ++calls);
  }
}

late GoRouter router;

/// The sample matcher unless a test passes its own, so no test calls the
/// real backend.
Widget appWith(List<FurnitureItem> items, {ProductMatchService service = const SampleProductMatchService()}) {
  router = GoRouter(
    initialLocation: '/results',
    routes: [
      GoRoute(path: '/results', builder: (context, state) => const ResultsScreen()),
      furnitureMatchesRoute,
      GoRoute(path: '/home', builder: (context, state) => const Scaffold(body: Text('HOME PAGE'))),
    ],
  );
  return ProviderScope(
    overrides: [
      appStateProvider.overrideWith((ref) => FakeAppState(items)),
      productMatchServiceProvider.overrideWithValue(service),
    ],
    child: MaterialApp.router(routerConfig: router),
  );
}

/// A phone-sized screen, so the whole results list is on screen.
void usePhone(WidgetTester tester) {
  tester.view.physicalSize = const Size(390 * 3, 900 * 3);
  tester.view.devicePixelRatio = 3;
  addTearDown(tester.view.reset);
}

Future<void> settle(WidgetTester tester) async {
  await tester.pump();
  await tester.pump(const Duration(milliseconds: 400));
}

Future<void> tapCard(WidgetTester tester, String label) async {
  final card = find.text(label).first;
  await tester.ensureVisible(card);
  await tester.tap(card);
  await settle(tester);
}

void main() {
  group('results: the furniture cards', () {
    testWidgets('one card per detected item, with readable names', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa'), furniture('coffee_table', id: '1')]));
      await settle(tester);

      expect(find.text('2 Furniture Detected'), findsOneWidget);
      expect(find.text('Sofa'), findsOneWidget);
      expect(find.text('Coffee Table'), findsOneWidget);
    });

    testWidgets("shows each item's rectangle crop, even when a cut-out exists", (tester) async {
      // The cut-out (SAM 2 mask) comes out in pieces for big or partly hidden
      // items like a sofa or a rug, so the app shows the plain photo crop.
      usePhone(tester);
      await tester.pumpWidget(appWith([
        furniture('sofa'),
        furniture('rug', id: '1', cutout: false),
      ]));
      await settle(tester);

      expect(find.byKey(const ValueKey('furniture-crop')), findsNWidgets(2));
      expect(find.byKey(const ValueKey('furniture-cutout')), findsNothing);
    });

    testWidgets('the results image has no heart button (saving is "Save Room Data")', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa')]));
      await settle(tester);

      expect(find.byIcon(Icons.favorite_border_rounded), findsNothing);
      expect(find.byIcon(Icons.favorite_rounded), findsNothing);
      expect(find.byIcon(Icons.ios_share_rounded), findsOneWidget, reason: 'only the heart is hidden');
    });

    testWidgets('an item with no image at all still gets a card with an icon', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('vase', cutout: false, crop: false)]));
      await settle(tester);

      expect(find.text('Vase'), findsOneWidget);
      expect(find.byKey(const ValueKey('furniture-cutout')), findsNothing);
      expect(find.byKey(const ValueKey('furniture-crop')), findsNothing);
      expect(
        find.descendant(of: find.byType(FurnitureSegmentCard), matching: find.byIcon(Icons.chair_rounded)),
        findsOneWidget,
      );
    });
  });

  group('FurnitureImage', () {
    Widget host(FurnitureItem item, {bool preferCutout = false}) => MaterialApp(
          home: Scaffold(
            body: SizedBox(
              width: 120,
              height: 120,
              child: FurnitureImage(item: item, preferCutout: preferCutout),
            ),
          ),
        );

    testWidgets('shows the crop by default', (tester) async {
      await tester.pumpWidget(host(furniture('sofa')));
      await tester.pump();

      expect(find.byKey(const ValueKey('furniture-crop')), findsOneWidget);
      expect(find.byKey(const ValueKey('furniture-cutout')), findsNothing);
    });

    testWidgets('shows the whole crop (fit inside the tile), never cropping its sides off', (tester) async {
      await tester.pumpWidget(host(furniture('sofa')));
      await tester.pump();

      final image = tester.widget<Image>(find.byKey(const ValueKey('furniture-crop')));
      expect(image.fit, BoxFit.contain);
    });

    testWidgets('shows the cut-out when asked to and one exists', (tester) async {
      await tester.pumpWidget(host(furniture('sofa'), preferCutout: true));
      await tester.pump();

      expect(find.byKey(const ValueKey('furniture-cutout')), findsOneWidget);
      expect(find.byKey(const ValueKey('furniture-crop')), findsNothing);
    });

    testWidgets('asked for a cut-out it does not have, it falls back to the crop', (tester) async {
      await tester.pumpWidget(host(furniture('rug', cutout: false), preferCutout: true));
      await tester.pump();

      expect(find.byKey(const ValueKey('furniture-crop')), findsOneWidget);
    });
  });

  group('extracted material and colour', () {
    testWidgets('the header shows the item\'s materials and colour name', (tester) async {
      usePhone(tester);
      final features = FurnitureFeatures(
        dominantColors: const ['#6d422c'],
        areaPct: 5,
        aspectRatio: 1,
        materials: [
          MaterialGuess(name: 'ceramic', confidence: 0.95),
          MaterialGuess(name: 'wood', confidence: 0.3),
        ],
        colorName: 'terracotta',
        source: 'gemini',
      );
      await tester.pumpWidget(appWith([furniture('vase', features: features)]));
      await settle(tester);

      await tapCard(tester, 'Vase');

      expect(find.byKey(const ValueKey('detected-traits')), findsOneWidget);
      expect(find.text('Ceramic, Wood · terracotta'), findsOneWidget);
    });

    test('describeTraits copes with a colour name only, a material only, or nothing', () {
      FurnitureFeatures with_({List<MaterialGuess> m = const [], String? name}) => FurnitureFeatures(
            dominantColors: const [],
            areaPct: 1,
            aspectRatio: 1,
            materials: m,
            colorName: name,
          );

      expect(describeTraits(with_(name: 'beige')), 'beige');
      expect(describeTraits(with_(m: [MaterialGuess(name: 'leather', confidence: 1)])), 'Leather');
      expect(describeTraits(with_()), isNull);
    });
  });

  group('tapping a card opens that item\'s matching products', () {
    testWidgets('shows the item, a clear sample-data notice and three numbered products', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa')]));
      await settle(tester);

      await tapCard(tester, 'Sofa');

      // header: which item this page is about (the results page is still
      // underneath in the navigation stack, so look only inside this page)
      final page = find.byType(FurnitureMatchesScreen);
      expect(find.descendant(of: page, matching: find.text('Sofa')), findsNWidgets(2),
          reason: 'app bar title + header');
      expect(find.text('Detected in your design'), findsOneWidget);
      expect(find.descendant(of: page, matching: find.byKey(const ValueKey('furniture-crop'))), findsOneWidget);

      // nothing extracted -> no material/colour line is invented
      expect(find.byKey(const ValueKey('detected-traits')), findsNothing);

      // it is obvious this is not real matching
      expect(find.text('Matching products'), findsOneWidget);
      expect(find.text('Sample data'), findsOneWidget);
      expect(find.textContaining('not real catalog products'), findsOneWidget);

      // the three products, most similar first
      expect(find.text('Sofa #1'), findsOneWidget);
      expect(find.text('Sofa #2'), findsOneWidget);
      expect(find.text('Sofa #3'), findsOneWidget);
      expect(find.text('94% match'), findsOneWidget);
      expect(find.text('Sample'), findsNWidgets(3), reason: 'every product card is tagged');
      final cards = find.byType(ProductMatchCard);
      expect(find.descendant(of: cards, matching: find.textContaining('฿')), findsNWidgets(3));
      expect(find.descendant(of: cards, matching: find.textContaining('cm')), findsNWidgets(3));
    });

    testWidgets('the tapped item is the one shown', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa'), furniture('floor lamp', id: '1')]));
      await settle(tester);

      await tapCard(tester, 'Floor Lamp');

      expect(find.text('Floor Lamp #1'), findsOneWidget);
      expect(find.text('Sofa #1'), findsNothing);
    });

    testWidgets('back returns to the results, which are still there', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa'), furniture('rug', id: '1')]));
      await settle(tester);
      await tapCard(tester, 'Sofa');
      expect(find.text('Sofa #1'), findsOneWidget);

      await tester.pageBack();
      await tester.pumpAndSettle(); // the results page has no endless animation

      expect(find.text('Sofa #1'), findsNothing);
      expect(find.text('2 Furniture Detected'), findsOneWidget);
    });

    testWidgets('a navigation that skips the redirect and has no item returns to results, not a crash', (tester) async {
      // Seen on web: the browser's back button restored this route without
      // its `extra` and the builder crashed on a bad cast (red error screen).
      usePhone(tester);
      late GoRouter local;
      local = GoRouter(
        initialLocation: '/bare',
        routes: [
          GoRoute(path: '/bare', builder: (context, state) => buildFurnitureMatchesPage(null)),
          GoRoute(path: '/results', builder: (context, state) => const Scaffold(body: Text('RESULTS PAGE'))),
        ],
      );
      await tester.pumpWidget(MaterialApp.router(routerConfig: local));
      await settle(tester);

      expect(tester.takeException(), isNull);
      expect(find.text('RESULTS PAGE'), findsOneWidget);
    });

    testWidgets('opening the route with no item (hot restart, bad link) goes back to results', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa')]));
      await settle(tester);

      router.go('/furniture-matches');
      await settle(tester);

      expect(find.text('1 Furniture Detected'), findsOneWidget);
      expect(tester.takeException(), isNull);
    });
  });

  group('the matches page while the matcher works', () {
    testWidgets('shows placeholders while loading, then the products', (tester) async {
      usePhone(tester);
      final finish = Completer<List<ProductMatch>>();
      await tester.pumpWidget(appWith([furniture('sofa')], service: ControlledService((_, _) => finish.future)));
      await settle(tester);

      await tapCard(tester, 'Sofa');
      expect(find.byKey(const ValueKey('matches-loading')), findsOneWidget);
      expect(find.text('Sofa #1'), findsNothing);

      finish.complete(await const SampleProductMatchService().matchesFor(furniture('sofa')));
      await settle(tester);

      expect(find.byKey(const ValueKey('matches-loading')), findsNothing);
      expect(find.text('Sofa #1'), findsOneWidget);
    });

    testWidgets('a failure offers a retry that works', (tester) async {
      usePhone(tester);
      final service = ControlledService((item, call) async {
        if (call == 1) throw Exception('matcher down');
        return const SampleProductMatchService().matchesFor(item);
      });
      await tester.pumpWidget(appWith([furniture('sofa')], service: service));
      await settle(tester);

      await tapCard(tester, 'Sofa');
      expect(find.text('Could not load matching products'), findsOneWidget);
      expect(find.text('Sofa #1'), findsNothing);

      await tester.tap(find.text('Try again'));
      await settle(tester);

      expect(service.calls, 2);
      expect(find.text('Could not load matching products'), findsNothing);
      expect(find.text('Sofa #1'), findsOneWidget);
    });

    testWidgets('no matches says so instead of showing an empty page', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa')], service: ControlledService((_, _) async => const [])));
      await settle(tester);

      await tapCard(tester, 'Sofa');

      expect(find.text('No matching products found'), findsOneWidget);
    });

    testWidgets('a real (non-sample) match is not tagged as a sample', (tester) async {
      usePhone(tester);
      const real = ProductMatch(
        id: 'real-1', name: 'KIVIK 3-seat sofa', price: 19990, category: 'Sofa',
        widthCm: 228, depthCm: 95, heightCm: 83, colorName: 'Grey', material: 'Fabric', score: 0.91,
      );
      await tester.pumpWidget(appWith([furniture('sofa')], service: ControlledService((_, _) async => const [real])));
      await settle(tester);

      await tapCard(tester, 'Sofa');

      expect(find.text('KIVIK 3-seat sofa'), findsOneWidget);
      expect(find.text('฿19,990'), findsOneWidget);
      expect(find.text('W 228 × D 95 × H 83 cm'), findsOneWidget);
      expect(find.text('91% match'), findsOneWidget);
      expect(find.text('Sample'), findsNothing);
      expect(find.text('Sample data'), findsNothing, reason: 'no sample badge for real results');
      expect(find.textContaining('not real catalog products'), findsNothing);
    });

    testWidgets('no confident match explains why', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa')], service: ControlledService((_, _) async => const [])));
      await settle(tester);
      await tapCard(tester, 'Sofa');

      expect(find.textContaining('Nothing in the catalog looks close enough'), findsOneWidget);
    });

    testWidgets('there is no price filter on the page (removed for now)', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa')]));
      await settle(tester);
      await tapCard(tester, 'Sofa');

      expect(find.byType(ChoiceChip), findsNothing);
      expect(find.text('Any price'), findsNothing);
    });
  });

  group('re-rank by colour / material', () {
    const real = ProductMatch(
      id: 'real-1', name: 'KIVIK 3-seat sofa', price: 19990, category: 'Sofa', score: 0.91,
    );

    final withFeatures = FurnitureFeatures(
      dominantColors: const ['#6d422c'],
      areaPct: 5,
      aspectRatio: 1,
      colors: [MeasuredColor(hex: '#6d422c', catalogColor: 'brown', share: 1)],
      materials: [MaterialGuess(name: 'wood', confidence: 0.9)],
      source: 'gemini',
    );

    FilterChip chipOf(WidgetTester tester, String key) => tester.widget<FilterChip>(find.byKey(ValueKey(key)));

    Future<ControlledService> openMatches(WidgetTester tester, FurnitureItem item) async {
      usePhone(tester);
      final service = ControlledService((_, _) async => const [real]);
      await tester.pumpWidget(appWith([item], service: service));
      await settle(tester);
      await tapCard(tester, 'Sofa');
      return service;
    }

    testWidgets('both start off, so the first search uses neither', (tester) async {
      final service = await openMatches(tester, furniture('sofa', features: withFeatures));

      expect(service.uses, [FeatureUse.none]);
      expect(chipOf(tester, 'use-colour').selected, isFalse);
      expect(chipOf(tester, 'use-material').selected, isFalse);
      expect(chipOf(tester, 'use-colour').onSelected, isNotNull);
      expect(find.text('Re-rank by'), findsOneWidget);
    });

    testWidgets('colour and material switch on separately and search again', (tester) async {
      final service = await openMatches(tester, furniture('sofa', features: withFeatures));

      await tester.tap(find.byKey(const ValueKey('use-colour')));
      await settle(tester);
      expect(service.uses.last, const FeatureUse(colour: true));
      expect(chipOf(tester, 'use-colour').selected, isTrue);
      expect(chipOf(tester, 'use-material').selected, isFalse);

      await tester.tap(find.byKey(const ValueKey('use-material')));
      await settle(tester);
      expect(service.uses.last, FeatureUse.all);

      await tester.tap(find.byKey(const ValueKey('use-colour')));
      await settle(tester);
      expect(service.uses.last, const FeatureUse(material: true));
      expect(service.calls, 4);
      expect(find.text('KIVIK 3-seat sofa'), findsOneWidget);
    });

    testWidgets('a chip is unavailable when that feature was not extracted', (tester) async {
      await openMatches(tester, furniture('sofa'));  // no colours or materials

      expect(chipOf(tester, 'use-colour').onSelected, isNull);
      expect(chipOf(tester, 'use-material').onSelected, isNull);
      expect(chipOf(tester, 'use-colour').selected, isFalse);
    });

    testWidgets('colour is available without a material guess, and not the other way round', (tester) async {
      final onlyColour = FurnitureFeatures(
        dominantColors: const ['#6d422c'],
        areaPct: 5,
        aspectRatio: 1,
        colors: [MeasuredColor(hex: '#6d422c', catalogColor: 'brown', share: 1)],
      );
      await openMatches(tester, furniture('sofa', features: onlyColour));

      expect(chipOf(tester, 'use-colour').onSelected, isNotNull);
      expect(chipOf(tester, 'use-material').onSelected, isNull);
    });

    testWidgets('both are unavailable for a cut-out the model called not a real item', (tester) async {
      final unusable = FurnitureFeatures(
        dominantColors: const [],
        areaPct: 5,
        aspectRatio: 1,
        colors: withFeatures.colors,
        materials: withFeatures.materials,
        source: 'gemini',
        usable: false,
      );
      await openMatches(tester, furniture('sofa', features: unusable));

      expect(chipOf(tester, 'use-colour').onSelected, isNull);
      expect(chipOf(tester, 'use-material').onSelected, isNull);
    });

    testWidgets('sample data has nothing to re-rank, so no chips', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([furniture('sofa', features: withFeatures)]));
      await settle(tester);
      await tapCard(tester, 'Sofa');

      expect(find.text('Sofa #1'), findsOneWidget);
      expect(find.byKey(const ValueKey('use-colour')), findsNothing);
      expect(find.byKey(const ValueKey('use-material')), findsNothing);
    });

    testWidgets('the setting carries over to the next item you open', (tester) async {
      usePhone(tester);
      final service = ControlledService((_, _) async => const [real]);
      await tester.pumpWidget(appWith([
        furniture('sofa', features: withFeatures),
        furniture('armchair', id: '1', features: withFeatures),
      ], service: service));
      await settle(tester);

      await tapCard(tester, 'Sofa');
      await tester.tap(find.byKey(const ValueKey('use-colour')));
      await settle(tester);
      router.pop();
      await settle(tester);

      await tapCard(tester, 'Armchair');
      expect(service.uses.last, const FeatureUse(colour: true));
      expect(chipOf(tester, 'use-colour').selected, isTrue);
    });
  });

  group('store link', () {
    const withLink = ProductMatch(
      id: '1', name: 'KIVIK 3-seat sofa', price: 19990, category: 'Sofa', score: 0.9,
      url: 'https://www.ikea.com/gb/en/p/kivik-sofa-s19440533/',
    );

    Widget card(ProductMatch product, OpenProductLink open) =>
        MaterialApp(home: Scaffold(body: ProductMatchCard(product: product, openLink: open)));

    testWidgets('opens the product page', (tester) async {
      final opened = <Uri>[];
      await tester.pumpWidget(card(withLink, (url) async {
        opened.add(url);
        return true;
      }));

      expect(find.text('View at IKEA'), findsOneWidget);
      await tester.tap(find.byKey(const ValueKey('product-store-link')));
      await tester.pump();

      expect(opened, [Uri.parse(withLink.url!)]);
    });

    testWidgets('says so when the page cannot be opened', (tester) async {
      await tester.pumpWidget(card(withLink, (url) async => false));

      await tester.tap(find.byKey(const ValueKey('product-store-link')));
      await tester.pump();

      expect(find.text('Could not open the store page'), findsOneWidget);
    });

    testWidgets('no link, no button', (tester) async {
      const noLink = ProductMatch(id: '2', name: 'n', price: 1, category: 'c', score: 0.9);
      await tester.pumpWidget(card(noLink, (url) async => true));

      expect(find.byKey(const ValueKey('product-store-link')), findsNothing);
    });
  });
}
