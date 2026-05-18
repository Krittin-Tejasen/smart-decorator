import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';

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
      await ref
          .read(processingProvider.notifier)
          .startProcessing();

      ref
          .read(appStateProvider.notifier)
          .setFakeResults();

      if (mounted) {
        context.go('/results');
      }
    });
  }

  @override
  Widget build(BuildContext context) {

    final processingState = ref.watch(processingProvider);

    return Scaffold(

      backgroundColor:
          const Color(0xFFE5E5E5),

      body: SafeArea(
        child: Center(
          child: Container(
            margin:
                const EdgeInsets.all(20),

            padding:
                const EdgeInsets.all(24),

            decoration: BoxDecoration(
              color: Colors.white,

              borderRadius:
                  BorderRadius.circular(40),
            ),

            child: Column(
              mainAxisSize: MainAxisSize.min,

              crossAxisAlignment:
                  CrossAxisAlignment.start,

              children: [
                const Center(
                  child: Text(
                    'Generating Room Design',

                    style: TextStyle(
                      fontSize: 32,
                      fontWeight:
                          FontWeight.bold,
                    ),
                  ),
                ),

                const SizedBox(height: 30),

                Container(
                  height: 220,
                  width: double.infinity,

                  decoration: BoxDecoration(
                    color: const Color(
                      0xFFF5F5F5,
                    ),

                    borderRadius:
                        BorderRadius.circular(
                      30,
                    ),
                  ),

                  child: const Center(
                    child: Icon(
                      Icons.chair_rounded,
                      size: 120,
                    ),
                  ),
                ),

                const SizedBox(height: 30),

                ClipRRect(
                  borderRadius:
                      BorderRadius.circular(
                    20,
                  ),

                  child:
                      LinearProgressIndicator(
                    minHeight: 14,

                    value:
                        processingState.progress,

                    backgroundColor:
                        Colors.grey.shade300,

                    valueColor:
                        const AlwaysStoppedAnimation(
                      AppColors.primary,
                    ),
                  ),
                ),

                const SizedBox(height: 30),

                ...processingState.steps.map(
                  (step) {

                    IconData icon;

                    Color color;

                    if (step.isCompleted) {

                      icon =
                          Icons.check_circle;

                      color =
                          AppColors.primary;
                    }
                    else if (step.isActive) {

                      icon =
                          Icons.radio_button_checked;

                      color =
                          AppColors.primary;
                    }
                    else {

                      icon =
                          Icons.radio_button_unchecked;

                      color = Colors.grey;
                    }

                    return Padding(
                      padding:
                          const EdgeInsets.only(
                        bottom: 18,
                      ),

                      child: Row(
                        children: [

                          Icon(
                            icon,
                            color: color,
                          ),

                          const SizedBox(
                            width: 14,
                          ),

                          Text(
                            step.title,

                            style:
                                const TextStyle(
                              fontSize: 22,
                              fontWeight:
                                  FontWeight.w500,
                            ),
                          ),
                        ],
                      ),
                    );
                  },
                ),

                const SizedBox(height: 20),

                const Center(
                  child:
                      CircularProgressIndicator(
                    color:
                        AppColors.primary,
                  ),
                ),

                const SizedBox(height: 24),

                const Center(
                  child: Text(
                    'Working on your new room design...',
                    style: TextStyle(
                      fontSize: 18,
                    ),
                  ),
                ),

                const SizedBox(height: 10),

                const Center(
                  child: Text(
                    'This may take a moment',
                    style: TextStyle(
                      fontSize: 16,
                      color: Colors.grey,
                    ),
                  ),
                ),

                const SizedBox(height: 30),

                SizedBox(
                  width: double.infinity,
                  height: 60,

                  child: ElevatedButton(
                    style:
                        ElevatedButton.styleFrom(
                      backgroundColor:
                          AppColors.primary,

                      foregroundColor:
                          Colors.white,

                      shape:
                          RoundedRectangleBorder(
                        borderRadius:
                            BorderRadius.circular(
                          20,
                        ),
                      ),
                    ),

                    onPressed: () {},

                    child: const Text(
                      'Cancel',

                      style: TextStyle(
                        fontSize: 22,
                        fontWeight:
                            FontWeight.bold,
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}