import 'dart:async';
import 'package:flutter_animate/flutter_animate.dart';
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

    await Future.delayed(const Duration(seconds: 2));
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