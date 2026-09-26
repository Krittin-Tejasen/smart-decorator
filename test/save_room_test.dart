import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:smart_decorator/core/services/auth_service.dart';
import 'package:smart_decorator/core/services/design_repository.dart';
import 'package:smart_decorator/features/results/presentation/results_screen.dart';
import 'package:smart_decorator/shared/models/color_option.dart';
import 'package:smart_decorator/shared/models/design_style.dart';
import 'package:smart_decorator/shared/models/furniture_item.dart';
import 'package:smart_decorator/shared/models/room_type.dart';
import 'package:smart_decorator/shared/providers/app_state_provider.dart';

import 'support/fake_design_repository.dart';
import 'support/sample_images.dart';

/// A generated room's PNG as the backend sends it (data URL).
final String generatedDataUrl = kSampleCutoutDataUrl;

final roomType = RoomType(id: 'living_room', title: 'Living Room', icon: Icons.weekend_rounded);
final style = DesignStyle(id: 'japandi', title: 'Japandi', subtitle: 'Warm Minimalist', colorOptions: []);
final color = ColorOption(id: 'warm_oat_cream', title: 'Warm Oat & Cream', colors: const [Colors.white]);

/// An app state that already holds a generated room ready to be saved.
class ReadyToSave extends AppStateNotifier {
  ReadyToSave(
    FakeDesignRepository repo, {
    String photoPath = '/photos/living.JPG',
    bool withGeneratedImage = true,
    List<FurnitureItem> furniture = const [],
  }) : super(
          repositoryFactory: () => repo,
          sourceReader: (file) async => Uint8List.fromList(utf8.encode('photo:${file.path}')),
        ) {
    state = state.copyWith(
      selectedRoomType: roomType,
      selectedStyle: style,
      selectedColorOption: color,
      uploadedImage: File(photoPath),
      generatedRoomImage: withGeneratedImage ? generatedDataUrl : null,
      segmentedFurniture: furniture,
    );
  }
}

void main() {
  group('AppStateNotifier.saveCurrentRoom', () {
    test('saves the generated image and the photo with how the room was described', () async {
      final repo = FakeDesignRepository();
      final notifier = ReadyToSave(repo);

      final result = await notifier.saveCurrentRoom();

      final sent = repo.saveRequests.single;
      expect(sent.roomType, 'living_room');
      expect(sent.style, 'japandi');
      expect(sent.color, 'warm_oat_cream');
      expect(sent.generatedImage, isNotEmpty);
      expect(utf8.decode(sent.sourceImage), 'photo:/photos/living.JPG');
      expect(sent.sourceExtension, 'jpg', reason: 'the extension is lower-cased');
      expect(result.savedCount, 1);
      expect(result.limit, 5);
      expect(notifier.state.designSaved, isTrue);
      expect(notifier.state.isSavingDesign, isFalse);
    });

    test('keeps the photo\'s real extension, and falls back to jpg for odd ones', () async {
      Future<String> extensionFor(String path) async {
        final repo = FakeDesignRepository();
        await ReadyToSave(repo, photoPath: path).saveCurrentRoom();
        return repo.saveRequests.single.sourceExtension;
      }

      expect(await extensionFor('/p/room.png'), 'png');
      expect(await extensionFor('/p/room.HEIC'), 'heic');
      expect(await extensionFor('/p/room.webp'), 'webp');
      expect(await extensionFor('/p/room.jpeg'), 'jpeg');
      expect(await extensionFor('/p/room.bmp'), 'jpg');
      expect(await extensionFor('/p/noextension'), 'jpg');
    });

    test('is "saving" while it uploads', () async {
      final repo = FakeDesignRepository()..saveGate = Completer<void>();
      final notifier = ReadyToSave(repo);

      final pending = notifier.saveCurrentRoom();
      await Future<void>.delayed(Duration.zero);
      expect(notifier.state.isSavingDesign, isTrue);

      repo.saveGate!.complete();
      await pending;
      expect(notifier.state.isSavingDesign, isFalse);
    });

    test('the same room cannot be saved twice', () async {
      final repo = FakeDesignRepository();
      final notifier = ReadyToSave(repo);
      await notifier.saveCurrentRoom();

      await expectLater(notifier.saveCurrentRoom(), throwsStateError);
      expect(repo.designs, hasLength(1));
    });

    test('nothing to save yet is an error, not a silent no-op', () async {
      final repo = FakeDesignRepository();

      await expectLater(ReadyToSave(repo, withGeneratedImage: false).saveCurrentRoom(), throwsStateError);
      await expectLater(AppStateNotifier().saveCurrentRoom(), throwsStateError);
      expect(repo.saveRequests, isEmpty);
    });

    test('at the limit it throws SaveLimitReached and can be tried again later', () async {
      final repo = FakeDesignRepository(
        designs: [for (var i = 0; i < 5; i++) savedDesign('d$i')],
      );
      final notifier = ReadyToSave(repo);

      await expectLater(notifier.saveCurrentRoom(), throwsA(isA<SaveLimitReachedException>()));
      expect(notifier.state.isSavingDesign, isFalse);
      expect(notifier.state.designSaved, isFalse, reason: 'nothing was saved');

      repo.designs.removeLast(); // the user deletes one in History
      await notifier.saveCurrentRoom();
      expect(notifier.state.designSaved, isTrue);
    });

    test('a failure leaves the state ready for another try', () async {
      final repo = FakeDesignRepository()..saveError = const NotSignedInException();
      final notifier = ReadyToSave(repo);

      await expectLater(notifier.saveCurrentRoom(), throwsA(isA<NotSignedInException>()));
      expect(notifier.state.isSavingDesign, isFalse);
      expect(notifier.state.designSaved, isFalse);

      repo.saveError = null;
      await notifier.saveCurrentRoom();
      expect(notifier.state.designSaved, isTrue);
    });
  });

  group('the limit message', () {
    test('names the limit and tells the user how to make room', () {
      expect(
        const SaveLimitReachedException(5).toString(),
        'You can save up to 5 rooms. Delete one in History to save this one.',
      );
    });
  });

  group('Results screen: Save Room Data', () {
    Future<GoRouter> pump(WidgetTester tester, AppStateNotifier notifier) async {
      tester.view.physicalSize = const Size(390 * 3, 1200 * 3);
      tester.view.devicePixelRatio = 3;
      addTearDown(tester.view.reset);

      final router = GoRouter(
        initialLocation: '/results',
        routes: [
          GoRoute(path: '/results', builder: (context, state) => const ResultsScreen()),
          GoRoute(path: '/history', builder: (context, state) => const Scaffold(body: Text('HISTORY PAGE'))),
        ],
      );
      await tester.pumpWidget(ProviderScope(
        overrides: [appStateProvider.overrideWith((ref) => notifier)],
        child: MaterialApp.router(routerConfig: router),
      ));
      await tester.pump();
      return router;
    }

    testWidgets('is offered for every generated room, not only after a scan', (tester) async {
      await pump(tester, ReadyToSave(FakeDesignRepository()));

      expect(find.text('Save Room Data'), findsOneWidget);
    });

    testWidgets('is not offered before anything is generated', (tester) async {
      await pump(tester, ReadyToSave(FakeDesignRepository(), withGeneratedImage: false));

      expect(find.text('Save Room Data'), findsNothing);
    });

    testWidgets('saving shows progress, then "Saved" with how many rooms are used', (tester) async {
      final repo = FakeDesignRepository(designs: [savedDesign('old-1'), savedDesign('old-2')])
        ..saveGate = Completer<void>();
      await pump(tester, ReadyToSave(repo));

      await tester.tap(find.text('Save Room Data'));
      await tester.pump();
      expect(find.text('Saving…'), findsOneWidget);
      expect(tester.widget<FilledButton>(find.byType(FilledButton)).onPressed, isNull,
          reason: 'no double tap while saving');

      repo.saveGate!.complete();
      await tester.pump();
      await tester.pump();

      expect(find.text('Saved'), findsOneWidget);
      expect(find.text('Room data saved (3 of 5)'), findsOneWidget);
      expect(tester.widget<FilledButton>(find.byType(FilledButton)).onPressed, isNull,
          reason: 'this result is already saved');
      expect(repo.designs, hasLength(3));
    });

    testWidgets('at the limit: explains, offers History, and stays saveable', (tester) async {
      final repo = FakeDesignRepository(designs: [for (var i = 0; i < 5; i++) savedDesign('d$i')]);
      await pump(tester, ReadyToSave(repo));

      await tester.tap(find.text('Save Room Data'));
      await tester.pump();
      await tester.pump();

      expect(find.text('You can save up to 5 rooms. Delete one in History to save this one.'), findsOneWidget);
      expect(find.text('Save Room Data'), findsOneWidget, reason: 'still can save once a room is deleted');

      await tester.pump(const Duration(milliseconds: 500)); // let the SnackBar finish sliding in
      await tester.tap(find.widgetWithText(SnackBarAction, 'History'));
      await tester.pump();
      await tester.pump(const Duration(milliseconds: 400));
      expect(find.text('HISTORY PAGE'), findsOneWidget);
    });

    testWidgets('not signed in: says so plainly', (tester) async {
      final repo = FakeDesignRepository()..saveError = const NotSignedInException();
      await pump(tester, ReadyToSave(repo));

      await tester.tap(find.text('Save Room Data'));
      await tester.pump();
      await tester.pump();

      expect(find.text('Could not sign in to save your rooms.'), findsOneWidget);
      expect(find.text('Save Room Data'), findsOneWidget);
    });

    testWidgets('any other failure shows the error and lets the user retry', (tester) async {
      final repo = FakeDesignRepository()..saveError = Exception('storage exploded');
      await pump(tester, ReadyToSave(repo));

      await tester.tap(find.text('Save Room Data'));
      await tester.pump();
      await tester.pump();

      expect(find.textContaining('Save failed:'), findsOneWidget);
      expect(find.textContaining('storage exploded'), findsOneWidget);
      expect(find.text('Save Room Data'), findsOneWidget);
    });
  });
}
