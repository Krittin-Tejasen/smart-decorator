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
    final isSample = matches.valueOrNull?.any((m) => m.isSample) ?? false;

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

          Row(
            children: [
              const Expanded(
                child: Text(
                  'Matching products',
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: AppColors.ink,
                  ),
                ),
              ),
              if (isSample) const _SampleBadge(),
            ],
          ),

          if (isSample) ...[
            const SizedBox(height: 10),
            const _SampleNote(),
          ],

          if (!isSample) ...[
            const SizedBox(height: 10),
            _FeatureToggles(item: item),
          ],

          const SizedBox(height: 14),

          matches.when(
            loading: () => const _MatchesLoading(),
            error: (error, stackTrace) => _MatchesError(
              onRetry: () => ref.invalidate(furnitureMatchesProvider(item)),
            ),
            data: (products) => products.isEmpty
                ? const _NoMatches()
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

/// "Colour" / "Material" chips: let the matcher re-rank by the item's colour
/// and material. Unavailable when the item has none, or when the model said the
/// cut-out is not a real item.
class _FeatureToggles extends ConsumerWidget {
  final FurnitureItem item;

  const _FeatureToggles({required this.item});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final use = ref.watch(featureUseProvider);
    final features = item.features;
    final hasColour = features.usable && features.colors.isNotEmpty;
    final hasMaterial = features.usable && features.hasMaterial;

    Widget chip({
      required Key key,
      required String label,
      required bool available,
      required bool selected,
      required void Function(bool) onChanged,
    }) {
      final child = FilterChip(
        key: key,
        label: Text(label),
        selected: selected && available,
        onSelected: available ? onChanged : null,
        showCheckmark: false,
        selectedColor: AppColors.brassTint,
        backgroundColor: AppColors.surface,
        disabledColor: AppColors.sandTint,
        side: BorderSide(color: AppColors.ink.withValues(alpha: 0.12)),
        labelStyle: TextStyle(
          fontSize: 12.5,
          color: available ? (selected ? AppColors.brassDeep : AppColors.ink) : AppColors.muted,
        ),
      );
      return available ? child : Tooltip(message: 'Not detected for this item', child: child);
    }

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Wrap(
          spacing: 8,
          crossAxisAlignment: WrapCrossAlignment.center,
          children: [
            const Text('Re-rank by', style: TextStyle(fontSize: 12, color: AppColors.muted)),
            chip(
              key: const ValueKey('use-colour'),
              label: 'Colour',
              available: hasColour,
              selected: use.colour,
              onChanged: (on) => ref.read(featureUseProvider.notifier).update((s) => s.copyWith(colour: on)),
            ),
            chip(
              key: const ValueKey('use-material'),
              label: 'Material',
              available: hasMaterial,
              selected: use.material,
              onChanged: (on) => ref.read(featureUseProvider.notifier).update((s) => s.copyWith(material: on)),
            ),
          ],
        ),
        const SizedBox(height: 4),
        const Text(
          'Only changes the order of the matches; none are hidden.',
          style: TextStyle(fontSize: 11.5, color: AppColors.muted),
        ),
      ],
    );
  }
}

/// One short line about an item's material and colour, e.g. "Wood, Fabric ·
/// terracotta". Null when there is nothing to say.
String? describeTraits(FurnitureFeatures features) {
  final materials = features.materials.map((m) => titleCase(m.name)).join(', ');
  final parts = [
    if (materials.isNotEmpty) materials,
    if (features.colorName != null) features.colorName!,
  ];
  return parts.isEmpty ? null : parts.join(' · ');
}

/// Nothing in the catalog was similar enough. Said plainly rather than
/// showing weak guesses.
class _NoMatches extends StatelessWidget {
  const _NoMatches();

  @override
  Widget build(BuildContext context) {
    return const Padding(
      padding: EdgeInsets.symmetric(vertical: 30),
      child: Column(
        children: [
          Text(
            'No matching products found',
            style: TextStyle(color: AppColors.muted),
          ),
          SizedBox(height: 4),
          Text(
            'Nothing in the catalog looks close enough to this item.',
            textAlign: TextAlign.center,
            style: TextStyle(fontSize: 12, color: AppColors.muted),
          ),
        ],
      ),
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

  /// "Wood, Fabric · terracotta", from what the backend extracted (null when
  /// it extracted neither a material nor a colour name).
  String? get _traits => describeTraits(item.features);

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

                if (_traits != null) ...[
                  const SizedBox(height: 4),
                  Text(
                    _traits!,
                    key: const ValueKey('detected-traits'),
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 12, color: AppColors.ink),
                  ),
                ],

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

/// Shown only for placeholder results, so nobody reading a demo mistakes the
/// products, prices or scores for real ones.
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
              'These are sample results that preview the layout, '
              'not real catalog products.',
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
