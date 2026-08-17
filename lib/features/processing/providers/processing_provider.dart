import 'dart:async';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/processing_step.dart';

class ProcessingState {

  final double progress;
  final List<ProcessingStep> steps;
  final bool isCompleted;

  ProcessingState({
    required this.progress,
    required this.steps,
    required this.isCompleted,
  });

  ProcessingState copyWith({
    double? progress,
    List<ProcessingStep>? steps,
    bool? isCompleted,
  }) {
    return ProcessingState(
      progress: progress ?? this.progress,
      steps: steps ?? this.steps,
      isCompleted: isCompleted ?? this.isCompleted,
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
            title: 'Reading room scan',
            isCompleted: false,
            isActive: true,
          ),

          ProcessingStep(
            title: 'Generating your room',
            isCompleted: false,
            isActive: false,
          ),

          ProcessingStep(
            title: 'Finding matching furniture',
            isCompleted: false,
            isActive: false,
          ),
        ],
      ),
    );

  /// Advances the fake step choreography up to "Finding matching products"
  /// (in progress) and then stops -- it does NOT mark processing as complete.
  /// Call [completeProcessing] once the real generate-room request actually
  /// finishes, whether that happens sooner or later than this animation.
  Future<void> startProcessing() async {

    await Future.delayed(const Duration(seconds: 2));
    state = state.copyWith(
      progress: 0.33,
      steps: [
        state.steps[0].copyWith(
          isCompleted: true,
          isActive: false,
        ),

        state.steps[1].copyWith(
          isActive: true,
        ),

        state.steps[2],
      ],
    );

    await Future.delayed(const Duration(seconds: 2));
    state = state.copyWith(
      progress: 0.66,

      steps: [
        state.steps[0],

        state.steps[1].copyWith(
          isCompleted: true,
          isActive: false,
        ),

        state.steps[2].copyWith(
          isActive: true,
        ),
      ],
    );
  }

  void completeProcessing() {
    state = state.copyWith(
      progress: 1,
      isCompleted: true,
      steps: [
        state.steps[0],

        state.steps[1],

        state.steps[2].copyWith(
          isCompleted: true,
          isActive: false,
        ),
      ],
    );
  }
}

final processingProvider = StateNotifierProvider<ProcessingNotifier, ProcessingState>((ref) {
      return ProcessingNotifier();
});