import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/widgets/app_icons.dart';
import '../models/processing_step.dart';

import '../providers/processing_provider.dart';

import 'package:go_router/go_router.dart';
import '../../../shared/providers/app_state_provider.dart';

class ProcessingScreen
    extends ConsumerStatefulWidget {

  const ProcessingScreen({super.key});

  @override
  ConsumerState<ProcessingScreen>
      createState() =>
          _ProcessingScreenState();
}

class _ProcessingScreenState
    extends ConsumerState<ProcessingScreen> {

  @override
  void initState() {
    super.initState();

    Future.microtask(() async {
      try {
        // Run the fake step choreography and the real backend request at the
        // same time, so the progress UI keeps reflecting "still working" for
        // however long the actual generation takes instead of sitting at
        // 100% while the network call is still in flight.
        await Future.wait([
          ref.read(processingProvider.notifier).startProcessing(),
          ref.read(appStateProvider.notifier).generateRoomDesign(),
        ]);

        ref
            .read(processingProvider.notifier)
            .completeProcessing();

        ref
            .read(appStateProvider.notifier)
            .saveToHistory();

        if (mounted) {
          context.go('/results');
        }
      } catch (error) {
        debugPrint('Generate room failed: $error');

        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Generate failed: $error'),
            ),
          );
          context.go('/home');
        }
      }
    });
  }

  @override
  Widget build(BuildContext context) {

    final processingState = ref.watch(processingProvider);

    return Scaffold(
      backgroundColor: AppColors.background,

      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 28),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Text(
                'Generating Room Design',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 22,
                  fontWeight: FontWeight.bold,
                  color: AppColors.ink,
                ),
              ),

              const SizedBox(height: 36),

              SizedBox(
                width: 160,
                height: 160,
                child: Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 160,
                      height: 160,
                      child: CircularProgressIndicator(
                        value: processingState.progress,
                        strokeWidth: 10,
                        strokeCap: StrokeCap.round,
                        backgroundColor: AppColors.sageTint,
                        valueColor: const AlwaysStoppedAnimation(AppColors.brass),
                      ),
                    ),
                    Container(
                      width: 116,
                      height: 116,
                      decoration: const BoxDecoration(
                        color: AppColors.surface,
                        shape: BoxShape.circle,
                      ),
                      child: Center(
                        child: SparkleIcon(size: 34, color: AppColors.sageDeep),
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 36),

              ...processingState.steps.map(
                (step) => _StepRow(step: step),
              ),

              const SizedBox(height: 20),

              const Text(
                'Working on your new room design...',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: AppColors.ink,
                ),
              ),

              const SizedBox(height: 6),

              const Text(
                'This may take a moment',
                textAlign: TextAlign.center,
                style: TextStyle(
                  fontSize: 12.5,
                  color: AppColors.muted,
                ),
              ),

              const Spacer(),

              TextButton(
                onPressed: () {},
                child: const Text(
                  'Cancel',
                  style: TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w600,
                    color: AppColors.sage,
                    decoration: TextDecoration.underline,
                  ),
                ),
              ),

              const SizedBox(height: 12),
            ],
          ),
        ),
      ),
    );
  }
}

class _StepRow extends StatelessWidget {
  final ProcessingStep step;
  const _StepRow({required this.step});

  @override
  Widget build(BuildContext context) {
    late final Widget iconTile;
    late final Color textColor;

    if (step.isCompleted) {
      iconTile = Container(
        width: 34,
        height: 34,
        decoration: const BoxDecoration(color: AppColors.sageDeep, shape: BoxShape.circle),
        child: const Icon(Icons.check_rounded, size: 18, color: Colors.white),
      );
      textColor = AppColors.ink;
    } else if (step.isActive) {
      iconTile = Container(
        width: 34,
        height: 34,
        decoration: BoxDecoration(
          color: AppColors.brassTint,
          shape: BoxShape.circle,
          border: Border.all(color: AppColors.brass, width: 2),
        ),
        child: Center(
          child: SparkleIcon(size: 15, color: AppColors.brassDeep),
        ),
      );
      textColor = AppColors.ink;
    } else {
      iconTile = Container(
        width: 34,
        height: 34,
        decoration: const BoxDecoration(color: AppColors.sandTint, shape: BoxShape.circle),
        child: const Icon(Icons.circle_outlined, size: 16, color: AppColors.muted),
      );
      textColor = AppColors.muted;
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Row(
        children: [
          iconTile,
          const SizedBox(width: 14),
          Text(
            step.title,
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: textColor,
            ),
          ),
        ],
      ),
    );
  }
}
