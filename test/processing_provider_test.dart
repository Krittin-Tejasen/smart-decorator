import 'package:flutter_test/flutter_test.dart';
import 'package:smart_decorator/features/processing/providers/processing_provider.dart';

void main() {
  group('ProcessingNotifier alongside a concurrent generate-room request', () {
    test('stays incomplete until the real request also finishes, even if it is slower', () async {
      final notifier = ProcessingNotifier();
      final slowRequest = Future<void>.delayed(const Duration(seconds: 5));

      final combined = Future.wait([notifier.startProcessing(), slowRequest]);

      await Future<void>.delayed(const Duration(milliseconds: 4200));
      expect(
        notifier.state.isCompleted,
        isFalse,
        reason:
            'the fake steps finish around 4s but the real request (5s) has not, '
            'so the screen must not show 100% yet',
      );

      await combined;
      notifier.completeProcessing();

      expect(notifier.state.isCompleted, isTrue);
      expect(notifier.state.progress, 1.0);
    });

    test('waits for the fake choreography even if the real request finishes fast', () async {
      final notifier = ProcessingNotifier();
      final fastRequest = Future<void>.delayed(const Duration(milliseconds: 100));

      final combined = Future.wait([notifier.startProcessing(), fastRequest]);

      await Future<void>.delayed(const Duration(milliseconds: 500));
      expect(
        notifier.state.isCompleted,
        isFalse,
        reason:
            'the 4s fake choreography has not finished yet even though the '
            'real request resolved in 100ms',
      );

      await combined;
      notifier.completeProcessing();

      expect(notifier.state.isCompleted, isTrue);
    });
  });
}
