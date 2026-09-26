import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/product_match_service.dart';
import '../../../shared/models/furniture_item.dart';
import '../../../shared/widgets/app_footer_nav.dart';
import '../providers/furniture_matches_provider.dart';
import '../widgets/furniture_image.dart';
import '../widgets/product_match_card.dart';

/// Products that match one detected piece of furniture (opened by tapping its
/// card on the results screen).
class FurnitureMatchesScreen extends ConsumerWidget {
  final FurnitureItem item;

  const FurnitureMatchesScreen({super.key, required this.item});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final matches = ref.watch(furnitureMatchesProvider(item));

    return Scaffold(
      backgroundColor: AppColors.background,

      appBar: AppBar(
        backgroundColor: AppColors.background,
        title: Text(
          titleCase(item.label),
          style: const TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: AppColors.ink,
          ),
        ),
      ),

      body: ListView(
        padding: const EdgeInsets.fromLTRB(18, 4, 18, 18),
        children: [
          _DetectedHeader(item: item),

          const SizedBox(height: 22),

          const Row(
            children: [
              Expanded(
                child: Text(
                  'Matching products',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppColors.ink,
                  ),
                ),
              ),
              _SampleBadge(),
            ],
          ),

          const SizedBox(height: 10),

          const _SampleNote(),

          const SizedBox(height: 14),

          matches.when(
            loading: () => const _MatchesLoading(),
            error: (error, stackTrace) => _MatchesError(
              onRetry: () => ref.invalidate(furnitureMatchesProvider(item)),
            ),
            data: (products) => products.isEmpty
                ? const Padding(
                    padding: EdgeInsets.symmetric(vertical: 30),
                    child: Center(
                      child: Text(
                        'No matching products found',
                        style: TextStyle(color: AppColors.muted),
                      ),
                    ),
                  )
                : Column(
                    children: [
                      for (final product in products) ...[
                        ProductMatchCard(product: product),
                        const SizedBox(height: 12),
                      ],
                    ],
                  ),
          ),
        ],
      ),

      bottomNavigationBar: const AppFooterNav(current: FooterTab.none),
    );
  }
}

/// The detected item itself: its cut-out and the colours found in it.
class _DetectedHeader extends StatelessWidget {
  final FurnitureItem item;

  const _DetectedHeader({required this.item});

  Color? _parse(String hex) {
    final digits = hex.replaceFirst('#', '');
    if (digits.length != 6) return null;
    final value = int.tryParse(digits, radix: 16);
    return value == null ? null : Color(0xFF000000 | value);
  }

  @override
  Widget build(BuildContext context) {
    final swatches = item.features.dominantColors
        .map(_parse)
        .whereType<Color>()
        .toList();

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AppColors.ink.withValues(alpha: 0.06),
            blurRadius: 14,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        children: [
          SizedBox(
            width: 96,
            height: 96,
            child: FurnitureImage(item: item, borderRadius: 12),
          ),

          const SizedBox(width: 14),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  titleCase(item.label),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppColors.ink,
                  ),
                ),

                const SizedBox(height: 2),

                const Text(
                  'Detected in your design',
                  style: TextStyle(fontSize: 12, color: AppColors.muted),
                ),

                if (swatches.isNotEmpty) ...[
                  const SizedBox(height: 10),
                  Row(
                    children: [
                      for (final color in swatches)
                        Container(
                          width: 18,
                          height: 18,
                          margin: const EdgeInsets.only(right: 6),
                          decoration: BoxDecoration(
                            color: color,
                            shape: BoxShape.circle,
                            border: Border.all(
                              color: AppColors.ink.withValues(alpha: 0.12),
                            ),
                          ),
                        ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _SampleBadge extends StatelessWidget {
  const _SampleBadge();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: AppColors.brassTint,
        borderRadius: BorderRadius.circular(999),
      ),
      child: const Text(
        'Sample data',
        style: TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.bold,
          color: AppColors.brassDeep,
        ),
      ),
    );
  }
}

/// Says plainly that this isn't real matching yet, so nobody reading the demo
/// mistakes the products, prices or scores for real results.
class _SampleNote extends StatelessWidget {
  const _SampleNote();

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.sandTint,
        borderRadius: BorderRadius.circular(12),
      ),
      child: const Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(Icons.info_outline_rounded, size: 16, color: AppColors.muted),
          SizedBox(width: 8),
          Expanded(
            child: Text(
              'These are sample results that preview the layout. '
              'Real product matching is still in development.',
              style: TextStyle(fontSize: 12, color: AppColors.muted, height: 1.35),
            ),
          ),
        ],
      ),
    );
  }
}

class _MatchesLoading extends StatelessWidget {
  const _MatchesLoading();

  @override
  Widget build(BuildContext context) {
    return Column(
      key: const ValueKey('matches-loading'),
      children: [
        for (var i = 0; i < 3; i++)
          Container(
            height: 112,
            margin: const EdgeInsets.only(bottom: 12),
            decoration: BoxDecoration(
              color: AppColors.sandTint,
              borderRadius: BorderRadius.circular(16),
            ),
          ),
      ],
    );
  }
}

class _MatchesError extends StatelessWidget {
  final VoidCallback onRetry;

  const _MatchesError({required this.onRetry});

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 24),
      child: Column(
        children: [
          const Icon(Icons.error_outline_rounded, size: 28, color: AppColors.muted),
          const SizedBox(height: 10),
          const Text(
            'Could not load matching products',
            style: TextStyle(fontSize: 13, color: AppColors.muted),
          ),
          TextButton(
            onPressed: onRetry,
            child: const Text('Try again'),
          ),
        ],
      ),
    );
  }
}
