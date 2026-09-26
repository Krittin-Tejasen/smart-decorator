import 'dart:async';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:smart_decorator/core/services/design_repository.dart';
import 'package:smart_decorator/core/services/segmentation_service.dart';
import 'package:smart_decorator/features/history/presentation/history_screen.dart';
import 'package:smart_decorator/features/history/providers/history_providers.dart';
import 'package:smart_decorator/features/results/presentation/furniture_matches_screen.dart';
import 'package:smart_decorator/routes/app_router.dart';
import 'package:smart_decorator/shared/models/design_labels.dart';
import 'package:smart_decorator/shared/models/furniture_item.dart';

import 'support/fake_design_repository.dart';
import 'support/sample_images.dart';

FurnitureItem item(String label, {String id = '0'}) => FurnitureItem(
      id: id,
      label: label,
      confidence: 0.8,
      bbox: BoundingBox(xMin: 0, yMin: 0, xMax: 1, yMax: 1),
      cropImage: kSampleCropDataUrl,
      maskImage: '',
      maskPrecise: false,
      features: FurnitureFeatures(dominantColors: const ['#725f4e'], areaPct: 5, aspectRatio: 1),
    );

/// Stands in for the backend's /segment-furniture.
class FakeSegmentation implements SegmentationService {
  FakeSegmentation(this.onSegment);
  final Future<SegmentationResult> Function(Uint8List bytes, int call) onSegment;
  int calls = 0;
  final List<Uint8List> received = [];

  @override
  Future<SegmentationResult> segment(Uint8List imageBytes, {String filename = 'room.png'}) {
    received.add(imageBytes);
    return onSegment(imageBytes, ++calls);
  }
}

SegmentationResult resultOf(List<FurnitureItem> items) => SegmentationResult(
      items: items,
      counts: {for (final i in items) i.label: 1},
      total: items.length,
      method: 'grounding_dino_sam2',
    );

late GoRouter router;

Widget appWith(
  FakeDesignRepository repo, {
  FakeSegmentation? segmentation,
  Future<Uint8List> Function(String url)? loader,
}) {
  router = GoRouter(
    initialLocation: '/history',
    routes: [
      GoRoute(path: '/history', builder: (context, state) => const HistoryScreen()),
      savedDesignRoute,
      furnitureMatchesRoute,
      GoRoute(path: '/results', builder: (context, state) => const Scaffold(body: Text('RESULTS PAGE'))),
      GoRoute(path: '/home', builder: (context, state) => const Scaffold(body: Text('HOME PAGE'))),
    ],
  );
  return ProviderScope(
    overrides: [
      designRepositoryProvider.overrideWithValue(repo),
      if (segmentation != null) segmentationServiceProvider.overrideWithValue(segmentation),
      imageBytesLoaderProvider.overrideWithValue(loader ?? (url) async => Uint8List.fromList([1, 2, 3])),
    ],
    child: MaterialApp.router(routerConfig: router),
  );
}

void usePhone(WidgetTester tester) {
  tester.view.physicalSize = const Size(390 * 3, 900 * 3);
  tester.view.devicePixelRatio = 3;
  addTearDown(tester.view.reset);
}

Future<void> settle(WidgetTester tester) async {
  await tester.pump();
  await tester.pump(const Duration(milliseconds: 400));
}

void main() {
  group('History list', () {
    testWidgets('empty: says what to do, matching the new save button', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith(FakeDesignRepository()));
      await settle(tester);

      expect(find.text('No Design History Yet'), findsOneWidget);
      expect(find.text('Tap Save Room Data on a generated design to save it here'), findsOneWidget);
      expect(find.textContaining('heart'), findsNothing, reason: 'the heart is gone');
    });

    testWidgets('lists the saved rooms with readable names, date, and how many of the limit', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [
        savedDesign('a', roomType: 'living_room', style: 'japandi'),
        savedDesign('b', roomType: 'home_office', style: 'industrial_loft'),
      ]);
      await tester.pumpWidget(appWith(repo));
      await settle(tester);

      expect(find.text('2 of 5 saved rooms'), findsOneWidget);
      expect(find.text('Living Room'), findsOneWidget);
      expect(find.text('Home Office'), findsOneWidget);
      expect(find.text('Style: Japandi'), findsOneWidget);
      expect(find.text('Style: Industrial / Loft'), findsOneWidget);
      expect(find.text(formatSavedAt(DateTime.utc(2026, 9, 27, 14, 5))), findsNWidgets(2));
    });

    testWidgets('a failure to load shows the reason', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository()..listError = Exception('offline');
      await tester.pumpWidget(appWith(repo));
      await settle(tester);

      expect(find.textContaining('Could not load history'), findsOneWidget);
      expect(find.textContaining('offline'), findsOneWidget);
    });
  });

  group('deleting a saved room', () {
    testWidgets('asks first; Cancel keeps it', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')]);
      await tester.pumpWidget(appWith(repo));
      await settle(tester);

      await tester.tap(find.byKey(const ValueKey('delete-a')));
      await settle(tester);
      expect(find.text('Delete this room?'), findsOneWidget);

      await tester.tap(find.text('Cancel'));
      await settle(tester);

      expect(repo.deleted, isEmpty);
      expect(find.text('1 of 5 saved rooms'), findsOneWidget);
    });

    testWidgets('confirming deletes it and the count and list update', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [
        savedDesign('a', roomType: 'bedroom'),
        savedDesign('b', roomType: 'kitchen'),
      ]);
      await tester.pumpWidget(appWith(repo));
      await settle(tester);

      await tester.tap(find.byKey(const ValueKey('delete-a')));
      await settle(tester);
      await tester.tap(find.text('Delete'));
      await settle(tester);

      expect(repo.deleted.single.id, 'a');
      expect(find.text('Room deleted'), findsOneWidget);
      expect(find.text('1 of 5 saved rooms'), findsOneWidget);
      expect(find.text('Bedroom'), findsNothing);
      expect(find.text('Kitchen'), findsOneWidget);
    });

    testWidgets('deleting the last room returns to the empty state', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')]);
      await tester.pumpWidget(appWith(repo));
      await settle(tester);

      await tester.tap(find.byKey(const ValueKey('delete-a')));
      await settle(tester);
      await tester.tap(find.text('Delete'));
      await settle(tester);

      expect(find.text('No Design History Yet'), findsOneWidget);
    });

    testWidgets('a failed delete says so and keeps the room', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')])..deleteError = Exception('nope');
      await tester.pumpWidget(appWith(repo));
      await settle(tester);

      await tester.tap(find.byKey(const ValueKey('delete-a')));
      await settle(tester);
      await tester.tap(find.text('Delete'));
      await settle(tester);

      expect(find.textContaining('Could not delete room'), findsOneWidget);
      expect(find.text('1 of 5 saved rooms'), findsOneWidget);
    });
  });

  group('opening a saved room', () {
    testWidgets('looks like Results and works out the furniture again, with a loading state first', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')]);
      final finish = Completer<SegmentationResult>();
      final segmentation = FakeSegmentation((_, _) => finish.future);
      await tester.pumpWidget(appWith(repo, segmentation: segmentation));
      await settle(tester);

      await tester.tap(find.text('Living Room'));
      await settle(tester);

      expect(find.text('Decorated Room'), findsOneWidget);
      expect(find.text('After'), findsOneWidget, reason: 'the saved photo makes Before/After available');
      expect(find.text('Before'), findsOneWidget);
      expect(find.text('Detecting furniture…'), findsOneWidget);
      expect(find.byKey(const ValueKey('furniture-loading')), findsOneWidget);
      expect(segmentation.calls, 1);
      expect(segmentation.received.single, [1, 2, 3], reason: 'the saved generated image goes to segmentation');

      finish.complete(resultOf([item('sofa', id: '0'), item('rug', id: '1')]));
      await settle(tester);

      expect(find.byKey(const ValueKey('furniture-loading')), findsNothing);
      expect(find.text('2 Furniture Detected'), findsOneWidget);
      expect(find.text('Sofa'), findsOneWidget);
      expect(find.text('Rug'), findsOneWidget);
    });

    testWidgets('no saved photo: no Before/After toggle', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a', sourceImagePath: null)]);
      await tester.pumpWidget(appWith(repo, segmentation: FakeSegmentation((_, _) async => resultOf([]))));
      await settle(tester);

      await tester.tap(find.text('Living Room'));
      await settle(tester);

      expect(find.text('Before'), findsNothing);
      expect(find.text('0 Furniture Detected'), findsOneWidget);
    });

    testWidgets('a detection failure offers a retry that works', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')]);
      final segmentation = FakeSegmentation((_, call) async {
        if (call == 1) throw Exception('backend down');
        return resultOf([item('sofa')]);
      });
      await tester.pumpWidget(appWith(repo, segmentation: segmentation));
      await settle(tester);
      await tester.tap(find.text('Living Room'));
      await settle(tester);

      expect(find.text('Could not detect furniture'), findsOneWidget);

      await tester.tap(find.text('Try again'));
      await settle(tester);

      expect(segmentation.calls, 2);
      expect(find.text('Could not detect furniture'), findsNothing);
      expect(find.text('1 Furniture Detected'), findsOneWidget);
    });

    testWidgets('an image that cannot be downloaded is an error, not an empty room', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')]);
      final segmentation = FakeSegmentation((_, _) async => resultOf([item('sofa')]));
      await tester.pumpWidget(appWith(repo, segmentation: segmentation, loader: (_) async => Uint8List(0)));
      await settle(tester);

      await tester.tap(find.text('Living Room'));
      await settle(tester);

      expect(find.text('Could not detect furniture'), findsOneWidget);
      expect(segmentation.calls, 0, reason: 'nothing to detect in an empty download');
    });

    testWidgets('reopening the same room does not run detection again', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')]);
      final segmentation = FakeSegmentation((_, _) async => resultOf([item('sofa')]));
      await tester.pumpWidget(appWith(repo, segmentation: segmentation));
      await settle(tester);

      await tester.tap(find.text('Living Room'));
      await settle(tester);
      await tester.pageBack();
      await tester.pumpAndSettle();
      await tester.tap(find.text('Living Room'));
      await settle(tester);

      expect(find.text('1 Furniture Detected'), findsOneWidget);
      expect(segmentation.calls, 1);
    });

    testWidgets('a detected item opens its matching products, like on Results', (tester) async {
      usePhone(tester);
      final repo = FakeDesignRepository(designs: [savedDesign('a')]);
      await tester.pumpWidget(appWith(repo, segmentation: FakeSegmentation((_, _) async => resultOf([item('sofa')]))));
      await settle(tester);
      await tester.tap(find.text('Living Room'));
      await settle(tester);

      await tester.tap(find.text('Sofa'));
      await settle(tester);

      expect(find.byType(FurnitureMatchesScreen), findsOneWidget);
      expect(find.text('Sofa #1'), findsOneWidget);
    });

    testWidgets('opening the route without a room goes back to History, not a crash', (tester) async {
      usePhone(tester);
      await tester.pumpWidget(appWith(FakeDesignRepository(designs: [savedDesign('a')])));
      await settle(tester);

      router.go('/saved-design');
      await settle(tester);

      expect(tester.takeException(), isNull);
      expect(find.text('1 of 5 saved rooms'), findsOneWidget);
    });

    testWidgets('the fallback page itself returns to History (browser back/forward case)', (tester) async {
      usePhone(tester);
      final local = GoRouter(
        initialLocation: '/bare',
        routes: [
          GoRoute(path: '/bare', builder: (context, state) => buildSavedDesignPage(null)),
          GoRoute(path: '/history', builder: (context, state) => const Scaffold(body: Text('HISTORY PAGE'))),
        ],
      );
      await tester.pumpWidget(MaterialApp.router(routerConfig: local));
      await settle(tester);

      expect(tester.takeException(), isNull);
      expect(find.text('HISTORY PAGE'), findsOneWidget);
    });
  });

  group('labels', () {
    test('known ids map to the titles on the Home screen', () {
      expect(roomTypeTitle('living_room'), 'Living Room');
      expect(roomTypeTitle('home_office'), 'Home Office');
      expect(styleTitle('industrial_loft'), 'Industrial / Loft');
      expect(styleTitle('modern_luxury'), 'Modern Luxury');
    });

    test('an unknown id is still readable', () {
      expect(roomTypeTitle('sun_room'), 'Sun Room');
      expect(styleTitle('boho'), 'Boho');
    });

    test('dates read like a shop receipt', () {
      final local = DateTime(2026, 9, 7, 4, 5);
      expect(formatSavedAt(local), '7 Sep 2026, 04:05');
    });
  });
}
