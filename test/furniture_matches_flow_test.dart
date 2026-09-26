import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:smart_decorator/core/services/product_match_service.dart';
import 'package:smart_decorator/features/results/presentation/furniture_matches_screen.dart';
import 'package:smart_decorator/features/results/presentation/results_screen.dart';
import 'package:smart_decorator/features/results/providers/furniture_matches_provider.dart';
import 'package:smart_decorator/features/results/widgets/furniture_segment_card.dart';
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
    features: FurnitureFeatures(dominantColors: colors, areaPct: 5, aspectRatio: 1),
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

  @override
  Future<List<ProductMatch>> matchesFor(FurnitureItem item) => onCall(item, ++calls);
}

late GoRouter router;

Widget appWith(List<FurnitureItem> items, {ProductMatchService? service}) {
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
      if (service != null) productMatchServiceProvider.overrideWithValue(service),
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

    testWidgets('shows the cut-out when there is one, otherwise the crop', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith([
        furniture('sofa'),
        furniture('rug', id: '1', cutout: false),
      ]));
      await settle(tester);

      expect(find.byKey(const ValueKey('furniture-cutout')), findsOneWidget);
      expect(find.byKey(const ValueKey('furniture-crop')), findsOneWidget);
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
      expect(find.descendant(of: page, matching: find.byKey(const ValueKey('furniture-cutout'))), findsOneWidget);

      // it is obvious this is not real matching
      expect(find.text('Matching products'), findsOneWidget);
      expect(find.text('Sample data'), findsOneWidget);
      expect(find.textContaining('Real product matching is still in development'), findsOneWidget);

      // the three products, most similar first
      expect(find.text('Sofa #1'), findsOneWidget);
      expect(find.text('Sofa #2'), findsOneWidget);
      expect(find.text('Sofa #3'), findsOneWidget);
      expect(find.text('94% match'), findsOneWidget);
      expect(find.text('Sample'), findsNWidgets(3), reason: 'every product card is tagged');
      expect(find.textContaining('฿'), findsNWidgets(3));
      expect(find.textContaining('cm'), findsNWidgets(3));
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
    });
  });
}
