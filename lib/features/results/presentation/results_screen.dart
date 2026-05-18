import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';

import '../../../shared/providers/app_state_provider.dart';

import '../widgets/product_card.dart';

class ResultsScreen
    extends ConsumerWidget {

  const ResultsScreen({super.key});

  @override
  Widget build(
    BuildContext context,
    WidgetRef ref,
  ) {

    final appState =
        ref.watch(appStateProvider);

    return Scaffold(

      appBar: AppBar(
        title: const Text(
          'Decorated Room',
        ),
      ),

      body: Padding(
        padding: const EdgeInsets.all(20),

        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment:
                CrossAxisAlignment.start,

            children: [

              Container(
                height: 250,
                width: double.infinity,

                decoration: BoxDecoration(
                  color:
                      const Color(0xFFEFEAEA),

                  borderRadius:
                      BorderRadius.circular(
                    30,
                  ),
                ),

                child: const Center(
                  child: Icon(
                    Icons.chair_rounded,
                    size: 140,
                  ),
                ),
              ),

              const SizedBox(height: 30),

              Container(
                padding:
                    const EdgeInsets.symmetric(
                  horizontal: 22,
                  vertical: 14,
                ),

                decoration: BoxDecoration(
                  color: AppColors.primary,

                  borderRadius:
                      BorderRadius.circular(
                    30,
                  ),
                ),

                child: Text(
                  '${appState.matchedProducts.length} Matching Product Found',

                  style: const TextStyle(
                    color: Colors.white,

                    fontSize: 20,

                    fontWeight:
                        FontWeight.bold,
                  ),
                ),
              ),

              const SizedBox(height: 30),

              ...appState.matchedProducts.map(
                (product) {

                  return ProductCard(
                    product: product,
                  );
                },
              ),
            ],
          ),
        ),
      ),
    );
  }
}