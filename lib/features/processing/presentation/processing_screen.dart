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
    extends ConsumerState<ProcessingScreen> with TickerProviderStateMixin {

  bool _cancelled = false;

  late final AnimationController _rotationController;
  late final AnimationController _pulseController;
  late final Animation<double> _pulseAnimation;

  @override
  void initState() {
    super.initState();

    _rotationController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2, milliseconds: 200),
    )..repeat();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1100),
    )..repeat(reverse: true);

    _pulseAnimation = Tween<double>(begin: 0.92, end: 1.08).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );

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

        if (_cancelled || !mounted) return;

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

        if (_cancelled || !mounted) return;

        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Generate failed: $error'),
          ),
        );
        context.go('/home');
      }
    });
  }

  void _cancel() {
    _cancelled = true;
    context.go('/home');
  }

  @override
  void dispose() {
    _rotationController.dispose();
    _pulseController.dispose();
    super.dispose();
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
                    // Static track.
                    SizedBox(
                      width: 160,
                      height: 160,
                      child: CircularProgressIndicator(
                        value: 1,
                        strokeWidth: 10,
                        backgroundColor: AppColors.sageTint,
                        valueColor: const AlwaysStoppedAnimation(AppColors.sageTint),
                      ),
                    ),

                    // Slim accent ring that keeps spinning so the screen
                    // still reads as "working" even while progress is
                    // holding steady between steps.
                    RotationTransition(
                      turns: _rotationController,
                      child: SizedBox(
                        width: 160,
                        height: 160,
                        child: CircularProgressIndicator(
                          value: 0.16,
                          strokeWidth: 3,
                          strokeCap: StrokeCap.round,
                          backgroundColor: Colors.transparent,
                          valueColor: const AlwaysStoppedAnimation(AppColors.brassTint),
                        ),
                      ),
                    ),

                    // Real progress, tweened smoothly between step values
                    // instead of snapping straight to the new value.
                    TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0, end: processingState.progress),
                      duration: const Duration(milliseconds: 650),
                      curve: Curves.easeInOutCubic,
                      builder: (context, value, _) {
                        return SizedBox(
                          width: 160,
                          height: 160,
                          child: CircularProgressIndicator(
                            value: value,
                            strokeWidth: 10,
                            strokeCap: StrokeCap.round,
                            backgroundColor: Colors.transparent,
                            valueColor: const AlwaysStoppedAnimation(AppColors.brass),
                          ),
                        );
                      },
                    ),

                    ScaleTransition(
                      scale: _pulseAnimation,
                      child: Container(
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
                onPressed: _cancel,
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
    late final Color bgColor;
    late final Color? borderColor;
    late final Widget icon;
    late final Color textColor;

    if (step.isCompleted) {
      bgColor = AppColors.sageDeep;
      borderColor = null;
      icon = const Icon(Icons.check_rounded, key: ValueKey('done'), size: 18, color: Colors.white);
      textColor = AppColors.ink;
    } else if (step.isActive) {
      bgColor = AppColors.brassTint;
      borderColor = AppColors.brass;
      icon = SparkleIcon(key: const ValueKey('active'), size: 15, color: AppColors.brassDeep);
      textColor = AppColors.ink;
    } else {
      bgColor = AppColors.sandTint;
      borderColor = null;
      icon = const Icon(Icons.circle_outlined, key: ValueKey('pending'), size: 16, color: AppColors.muted);
      textColor = AppColors.muted;
    }

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 7),
      child: Row(
        children: [
          AnimatedContainer(
            duration: const Duration(milliseconds: 320),
            curve: Curves.easeOut,
            width: 34,
            height: 34,
            decoration: BoxDecoration(
              color: bgColor,
              shape: BoxShape.circle,
              border: borderColor != null ? Border.all(color: borderColor, width: 2) : null,
            ),
            child: Center(
              child: AnimatedSwitcher(
                duration: const Duration(milliseconds: 220),
                child: icon,
              ),
            ),
          ),
          const SizedBox(width: 14),
          AnimatedDefaultTextStyle(
            duration: const Duration(milliseconds: 320),
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: textColor,
            ),
            child: Text(step.title),
          ),
        ],
      ),
    );
  }
}
