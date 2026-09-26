import 'package:flutter_test/flutter_test.dart';
import 'package:smart_decorator/features/processing/providers/processing_provider.dart';
import 'package:smart_decorator/shared/models/generation_progress.dart';

GenerationProgress update(String stage, double progress, [String? message]) =>
    GenerationProgress(stage: stage, message: message ?? 'msg-$stage', progress: progress);

/// Which of the three checklist steps are done / active / pending, as a string
/// like "done,active,pending" so failures read clearly.
String steps(ProcessingNotifier n) => n.state.steps
    .map((s) => s.isCompleted ? 'done' : (s.isActive ? 'active' : 'pending'))
    .join(',');

void main() {
  group('ProcessingNotifier follows the backend, not a timer', () {
    test('starts on the original wording with nothing done and step one active', () {
      final n = ProcessingNotifier();

      expect(n.state.statusMessage, 'This may take a moment');
      expect(n.state.progress, 0);
      expect(n.state.isCompleted, isFalse);
      expect(steps(n), 'active,pending,pending');
      expect(
        n.state.steps.map((s) => s.title),
        ['Scanning Room', 'Analyze & Generate', 'Finding matching products'],
        reason: 'the step titles are existing copy and must not change',
      );
    });

    test('nothing happens by itself - no timer ticks steps over', () async {
      final n = ProcessingNotifier();

      await Future<void>.delayed(const Duration(milliseconds: 2500));

      expect(steps(n), 'active,pending,pending');
      expect(n.state.progress, 0);
    });

    test('each backend stage moves the checklist and the status line', () {
      final n = ProcessingNotifier();

      n.applyProgress(update('compose', 0.05, 'Analyzing your room and planning the design'));
      expect(steps(n), 'active,pending,pending');
      expect(n.state.statusMessage, 'Analyzing your room and planning the design');

      n.applyProgress(update('generate', 0.15));
      expect(steps(n), 'done,active,pending');

      n.applyProgress(update('critic', 0.65));
      expect(steps(n), 'done,active,pending', reason: 'critic belongs to "Analyze & Generate"');

      n.applyProgress(update('match', 0.9));
      expect(steps(n), 'done,done,active');
      expect(n.state.progress, 0.9);
    });

    test('a critic retry does not un-tick anything or move the ring back', () {
      final n = ProcessingNotifier();
      n.applyProgress(update('generate', 0.15));
      n.applyProgress(update('critic', 0.65));

      n.applyProgress(update('retry', 0.4, 'Refining the design (attempt 2)'));

      expect(steps(n), 'done,active,pending');
      expect(n.state.progress, 0.65, reason: 'progress never goes backwards');
      expect(n.state.statusMessage, 'Refining the design (attempt 2)',
          reason: 'but the status text must still say what is happening now');
    });

    test('stages arriving late or out of order never rewind the checklist', () {
      final n = ProcessingNotifier();
      n.applyProgress(update('match', 0.9));

      n.applyProgress(update('generate', 0.15));

      expect(steps(n), 'done,done,active');
    });

    test('an unknown stage only updates the status text', () {
      final n = ProcessingNotifier();
      n.applyProgress(update('generate', 0.15));

      n.applyProgress(update('some-future-stage', 0.5, 'Doing something new'));

      expect(steps(n), 'done,active,pending');
      expect(n.state.statusMessage, 'Doing something new');
    });

    test('an empty message keeps the previous status text', () {
      final n = ProcessingNotifier();
      n.applyProgress(update('generate', 0.15, 'Generating your new room design'));

      n.applyProgress(update('generate', 0.2, ''));

      expect(n.state.statusMessage, 'Generating your new room design');
    });

    test('progress can never reach 100% from the backend alone', () {
      final n = ProcessingNotifier();

      n.applyProgress(update('segment', 1.0));

      expect(n.state.progress, lessThan(1.0));
      expect(n.state.isCompleted, isFalse);
    });

    test('completeProcessing finishes everything, and later updates are ignored', () {
      final n = ProcessingNotifier();
      n.applyProgress(update('generate', 0.3));

      n.completeProcessing();
      n.applyProgress(update('compose', 0.05, 'stale update'));

      expect(n.state.isCompleted, isTrue);
      expect(n.state.progress, 1.0);
      expect(steps(n), 'done,done,done');
      expect(n.state.statusMessage, isNot('stale update'));
    });
  });

  group('stage -> checklist step mapping', () {
    test('matches what each backend stage really does', () {
      expect(ProcessingNotifier.stepIndexForStage('compose'), 0);
      for (final s in ['generate', 'critic', 'retry']) {
        expect(ProcessingNotifier.stepIndexForStage(s), 1, reason: s);
      }
      for (final s in ['match', 'segment']) {
        expect(ProcessingNotifier.stepIndexForStage(s), 2, reason: s);
      }
      expect(ProcessingNotifier.stepIndexForStage('???'), -1);
    });
  });
}
