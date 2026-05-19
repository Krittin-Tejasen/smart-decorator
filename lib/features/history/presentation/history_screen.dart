import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/providers/app_state_provider.dart';

class HistoryScreen extends ConsumerWidget {

  const HistoryScreen({super.key});

  @override
  Widget build(
    BuildContext context,
    WidgetRef ref,
  ) {

    final history = ref.watch(appStateProvider).history;

    return Scaffold(

      appBar: AppBar(
        title: const Text('History'),
      ),

      body: history.isEmpty
          ? const Center(
              child: Text('No Design History Yet'),
            )

          : ListView.builder(

              padding:
                  const EdgeInsets.all(20),

              itemCount: history.length,

              itemBuilder: (
                context,
                index,
              ) {

                final item =
                    history[index];

                return Container(
                  margin:
                      const EdgeInsets.only(
                    bottom: 20,
                  ),

                  padding:
                      const EdgeInsets.all(20),

                  decoration: BoxDecoration(
                    color: Colors.white,

                    borderRadius:
                        BorderRadius.circular(
                      24,
                    ),
                  ),

                  child: Column(
                    crossAxisAlignment:
                        CrossAxisAlignment
                            .start,

                    children: [

                      Text(
                        item.roomType,

                        style:
                            const TextStyle(
                          fontSize: 24,

                          fontWeight:
                              FontWeight.bold,
                        ),
                      ),

                      const SizedBox(height: 10),

                      Text(
                        'Theme: ${item.theme}',
                      ),

                      const SizedBox(height: 10),

                      Text(
                        '${item.products.length} Products Matched',
                      ),

                      const SizedBox(height: 10),

                      Text(
                        item.createdAt
                            .toString(),
                      ),
                    ],
                  ),
                );
              },
            ),
    );
  }
}