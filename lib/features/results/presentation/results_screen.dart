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
          style: TextStyle(
            height: 1.2,
            fontSize: 30,
            fontWeight: FontWeight.bold,
          ),
        ),
      ),

      body: Padding(
        padding: const EdgeInsets.all(15),

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

                alignment: Alignment.center,

                child: Text(
                  '${appState.matchedProducts.length} Matching Product Found',

                  textAlign: TextAlign.center,
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