import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:go_router/go_router.dart';
import 'package:smart_decorator/core/services/ai_generation_service.dart';
import 'package:smart_decorator/features/processing/presentation/processing_screen.dart';
import 'package:smart_decorator/shared/models/generation_progress.dart';
import 'package:smart_decorator/shared/providers/app_state_provider.dart';

typedef Run = Future<void> Function(
  void Function(GenerationProgress)? onProgress,
  GenerationCancelToken? cancelToken,
);

/// An app state whose generation is driven by the test instead of the network.
class FakeAppState extends AppStateNotifier {
  FakeAppState(this.run);
  final Run run;

  @override
  Future<void> generateRoomDesign({
    void Function(GenerationProgress progress)? onProgress,
    GenerationCancelToken? cancelToken,
  }) =>
      run(onProgress, cancelToken);
}

Widget appFor(FakeAppState fake) {
  final router = GoRouter(
    initialLocation: '/processing',
    routes: [
      GoRoute(path: '/processing', builder: (context, state) => const ProcessingScreen()),
      GoRoute(path: '/results', builder: (context, state) => const Scaffold(body: Text('RESULTS PAGE'))),
      GoRoute(path: '/home', builder: (context, state) => const Scaffold(body: Text('HOME PAGE'))),
    ],
  );
  return ProviderScope(
    overrides: [appStateProvider.overrideWith((ref) => fake)],
    child: MaterialApp.router(routerConfig: router),
  );
}

GenerationProgress step(String stage, String message, double progress) =>
    GenerationProgress(stage: stage, message: message, progress: progress);

/// The ring and step icons animate forever, so pumpAndSettle would never
/// return; advance time explicitly instead. Two pumps: the first builds the
/// frame after a state change / navigation (an AnimatedSwitcher only starts
/// its transition there, and GoRouter resolves routes asynchronously), the
/// second lets the transition finish.
Future<void> settle(WidgetTester tester) async {
  await tester.pump();
  await tester.pump(const Duration(milliseconds: 400));
}

void main() {
  testWidgets('shows the original line first, then whatever the backend is doing', (tester) async {
    final finish = Completer<void>();
    void Function(GenerationProgress)? push;
    await tester.pumpWidget(appFor(FakeAppState((onProgress, _) {
      push = onProgress;
      return finish.future;
    })));
    await tester.pump();

    expect(find.text('This may take a moment'), findsOneWidget);
    expect(find.text('Generating Room Design'), findsOneWidget);
    expect(find.text('Working on your new room design...'), findsOneWidget);
    for (final title in ['Scanning Room', 'Analyze & Generate', 'Finding matching products']) {
      expect(find.text(title), findsOneWidget);
    }

    push!(step('compose', 'Analyzing your room and planning the design', 0.05));
    await settle(tester);
    expect(find.text('Analyzing your room and planning the design'), findsOneWidget);
    expect(find.text('This may take a moment'), findsNothing);

    push!(step('generate', 'Generating your new room design', 0.15));
    await settle(tester);
    expect(find.text('Generating your new room design'), findsOneWidget);
    expect(find.text('Analyzing your room and planning the design'), findsNothing);

    push!(step('retry', 'Refining the design (attempt 2)', 0.7));
    await settle(tester);
    expect(find.text('Refining the design (attempt 2)'), findsOneWidget);

    finish.complete();
    await tester.pump();
    await settle(tester);
  });

  testWidgets('goes to the results page when the real generation finishes', (tester) async {
    final finish = Completer<void>();
    await tester.pumpWidget(appFor(FakeAppState((onProgress, _) => finish.future)));
    await tester.pump();
    expect(find.text('RESULTS PAGE'), findsNothing);

    finish.complete();
    await tester.pump();
    await settle(tester);

    expect(find.text('RESULTS PAGE'), findsOneWidget);
  });

  testWidgets('Cancel stops the generation and returns home without an error', (tester) async {
    final finish = Completer<void>();
    GenerationCancelToken? token;
    await tester.pumpWidget(appFor(FakeAppState((onProgress, cancelToken) {
      token = cancelToken;
      return finish.future;
    })));
    await tester.pump();
    expect(token, isNotNull);
    expect(token!.isCancelled, isFalse);

    await tester.tap(find.text('Cancel'));
    await settle(tester);

    expect(token!.isCancelled, isTrue, reason: 'the backend job must be told to stop');
    expect(find.text('HOME PAGE'), findsOneWidget);

    // The generation "notices" the cancel and throws, like the real service.
    finish.completeError(GenerationCancelled());
    await tester.pump();
    await settle(tester);
    expect(find.byType(SnackBar), findsNothing, reason: 'cancelling is not a failure');
    expect(find.text('HOME PAGE'), findsOneWidget);
  });

  testWidgets('a failed generation shows the backend message and returns home', (tester) async {
    await tester.pumpWidget(appFor(FakeAppState(
      (onProgress, _) => Future.error(Exception('Gemini did not return an image.')),
    )));
    await tester.pump();
    await settle(tester);

    expect(find.text('HOME PAGE'), findsOneWidget);
    expect(find.textContaining('Gemini did not return an image.'), findsOneWidget);
  });

  testWidgets('progress updates that arrive after Cancel are ignored safely', (tester) async {
    final finish = Completer<void>();
    void Function(GenerationProgress)? push;
    await tester.pumpWidget(appFor(FakeAppState((onProgress, _) {
      push = onProgress;
      return finish.future;
    })));
    await tester.pump();

    await tester.tap(find.text('Cancel'));
    await settle(tester);

    push!(step('critic', 'Checking doors, windows and outlets are still in place', 0.65));
    await settle(tester);

    expect(tester.takeException(), isNull);
    expect(find.text('HOME PAGE'), findsOneWidget);
    finish.complete();
    await tester.pump();
  });
}
