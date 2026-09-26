import 'dart:math' as math;

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/models/generation_progress.dart';
import '../models/processing_step.dart';

/// What the screen shows before the backend has reported its first stage.
const String kDefaultProcessingStatus = 'This may take a moment';

class ProcessingState {

  final double progress;
  final List<ProcessingStep> steps;
  final bool isCompleted;

  /// The line under the steps: what the backend is doing right now.
  final String statusMessage;

  ProcessingState({
    required this.progress,
    required this.steps,
    required this.isCompleted,
    this.statusMessage = kDefaultProcessingStatus,
  });

  ProcessingState copyWith({
    double? progress,
    List<ProcessingStep>? steps,
    bool? isCompleted,
    String? statusMessage,
  }) {
    return ProcessingState(
      progress: progress ?? this.progress,
      steps: steps ?? this.steps,
      isCompleted: isCompleted ?? this.isCompleted,
      statusMessage: statusMessage ?? this.statusMessage,
    );
  }
}

class ProcessingNotifier extends StateNotifier<ProcessingState> {
  ProcessingNotifier() :
    super(
      ProcessingState(
        progress: 0,
        isCompleted: false,
        steps: [
          ProcessingStep(
            title: 'Scanning Room',
            isCompleted: false,
            isActive: true,
          ),

          ProcessingStep(
            title: 'Analyze & Generate',
            isCompleted: false,
            isActive: false,
          ),

          ProcessingStep(
            title: 'Finding matching products',
            isCompleted: false,
            isActive: false,
          ),
        ],
      ),
    );

  /// Which checklist step a backend pipeline stage belongs to, or -1 for a
  /// stage this screen doesn't know (it then only updates the status text).
  ///
  /// compose  - the AI reads the room photo and plans the design
  /// generate / critic / retry - generating the image and checking it
  /// match / segment - finding furniture and products in the result
  static int stepIndexForStage(String stage) {
    switch (stage) {
      case 'compose':
        return 0;
      case 'generate':
      case 'critic':
      case 'retry':
        return 1;
      case 'match':
      case 'segment':
        return 2;
      default:
        return -1;
    }
  }

  int _currentStep = 0;

  /// Applies a real update from the backend. Progress and the current step
  /// never move backwards (a retry can re-enter "generating" but the ring and
  /// the checklist don't un-tick), and nothing here can reach 100% - only
  /// [completeProcessing] does, once the result has actually arrived.
  void applyProgress(GenerationProgress update) {
    if (state.isCompleted) return;

    final step = stepIndexForStage(update.stage);
    if (step > _currentStep) _currentStep = step;

    state = state.copyWith(
      progress: math.max(state.progress, math.min(update.progress, 0.99)),
      statusMessage: update.message.isEmpty ? state.statusMessage : update.message,
      steps: [
        for (var i = 0; i < state.steps.length; i++)
          state.steps[i].copyWith(
            isCompleted: i < _currentStep,
            isActive: i == _currentStep,
          ),
      ],
    );
  }

  void completeProcessing() {
    state = state.copyWith(
      progress: 1,
      isCompleted: true,
      steps: [
        for (final step in state.steps)
          step.copyWith(isCompleted: true, isActive: false),
      ],
    );
  }
}

final processingProvider = StateNotifierProvider.autoDispose<ProcessingNotifier, ProcessingState>((ref) {
      return ProcessingNotifier();
});
