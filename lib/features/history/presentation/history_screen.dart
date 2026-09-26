import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/design_save_service.dart';
import '../../../shared/models/saved_design.dart';
import '../../../shared/widgets/app_footer_nav.dart';

/// Designs the user has favorited (room_designs.is_saved = true), newest
/// first, each with a resolved signed URL for its generated image.
final savedDesignsProvider =
    FutureProvider.autoDispose<List<SavedDesign>>((ref) async {
  final service = DesignSaveService();
  final rows = await service.fetchSavedDesigns();

  return Future.wait(rows.map((row) async {
    final path = row['generated_image_path'] as String? ?? '';
    final url = path.isEmpty ? '' : await service.getImageUrl(path);
    return SavedDesign.fromRow(row, url);
  }));
});

class HistoryScreen extends ConsumerWidget {

  const HistoryScreen({super.key});

  @override
  Widget build(
    BuildContext context,
    WidgetRef ref,
  ) {

    final designsAsync = ref.watch(savedDesignsProvider);

    return Scaffold(
      backgroundColor: AppColors.background,

      appBar: AppBar(
        backgroundColor: AppColors.background,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_rounded, color: AppColors.ink),
          onPressed: () {
            if (context.canPop()) {
              context.pop();
            } else {
              context.go('/home');
            }
          },
        ),
        title: const Text(
          'History',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: AppColors.ink,
          ),
        ),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh_rounded, color: AppColors.ink),
            onPressed: () => ref.invalidate(savedDesignsProvider),
          ),
        ],
      ),

      body: designsAsync.when(
        data: (designs) => designs.isEmpty
            ? const _EmptyHistory()
            : ListView.builder(
                padding: const EdgeInsets.fromLTRB(18, 4, 18, 18),
                itemCount: designs.length,
                itemBuilder: (context, index) {
                  return _HistoryCard(item: designs[index]);
                },
              ),
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => _HistoryError(message: '$error'),
      ),

      bottomNavigationBar: const AppFooterNav(current: FooterTab.history),
    );
  }
}

class _HistoryCard extends StatelessWidget {
  final SavedDesign item;
  const _HistoryCard({required this.item});

  IconData get _roomIcon {
    final key = item.roomType.toLowerCase();
    if (key.contains('bed')) return Icons.bed_rounded;
    if (key.contains('dining') || key.contains('table')) return Icons.table_restaurant_rounded;
    if (key.contains('kitchen')) return Icons.kitchen_rounded;
    if (key.contains('office')) return Icons.work_rounded;
    return Icons.weekend_rounded;
  }

  Color get _tileColor {
    final key = item.style.toLowerCase();
    if (key.contains('luxury')) return const Color(0xFFEFE1C9);
    if (key.contains('industrial') || key.contains('loft')) return AppColors.lightCard;
    return AppColors.sageTint;
  }

  Color get _tileIconColor {
    final key = item.style.toLowerCase();
    if (key.contains('luxury')) return AppColors.brassDeep;
    if (key.contains('industrial') || key.contains('loft')) return AppColors.ink;
    return AppColors.sageDeep;
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(12),

      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: AppColors.ink.withValues(alpha: 0.06),
            blurRadius: 16,
            offset: const Offset(0, 8),
          ),
        ],
      ),

      child: Row(
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(14),
            child: Container(
              width: 58,
              height: 58,
              color: _tileColor,
              child: item.imageUrl.isEmpty
                  ? Icon(_roomIcon, size: 26, color: _tileIconColor)
                  : Image.network(
                      item.imageUrl,
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stackTrace) =>
                          Icon(_roomIcon, size: 26, color: _tileIconColor),
                      loadingBuilder: (context, child, progress) =>
                          progress == null
                              ? child
                              : Icon(_roomIcon, size: 26, color: _tileIconColor),
                    ),
            ),
          ),

          const SizedBox(width: 14),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  item.roomType,
                  style: const TextStyle(
                    fontSize: 14.5,
                    fontWeight: FontWeight.bold,
                    color: AppColors.ink,
                  ),
                ),

                const SizedBox(height: 6),

                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: [
                    _Chip(
                      label: 'Style: ${item.style}',
                      bg: _tileColor,
                      fg: _tileIconColor,
                    ),
                    _Chip(
                      label: '${item.furnitureCount} Furniture Detected',
                      bg: AppColors.brassTint,
                      fg: AppColors.brassDeep,
                    ),
                  ],
                ),

                const SizedBox(height: 6),

                Text(
                  item.createdAt.toString(),
                  style: const TextStyle(fontSize: 11, color: AppColors.muted),
                ),
              ],
            ),
          ),

          const Icon(Icons.chevron_right_rounded, color: AppColors.muted),
        ],
      ),
    );
  }
}

class _Chip extends StatelessWidget {
  final String label;
  final Color bg;
  final Color fg;
  const _Chip({required this.label, required this.bg, required this.fg});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 3),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 10.5,
          fontWeight: FontWeight.w600,
          color: fg,
        ),
      ),
    );
  }
}

class _EmptyHistory extends StatelessWidget {
  const _EmptyHistory();

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: const BoxDecoration(
              color: AppColors.sandTint,
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.history_rounded, size: 28, color: AppColors.muted),
          ),
          const SizedBox(height: 14),
          const Text(
            'No Design History Yet',
            style: TextStyle(
              fontSize: 15,
              fontWeight: FontWeight.bold,
              color: AppColors.ink,
            ),
          ),
          const SizedBox(height: 6),
          const Text(
            'Tap the heart on a generated design to save it here',
            style: TextStyle(fontSize: 12, color: AppColors.muted),
          ),
        ],
      ),
    );
  }
}

class _HistoryError extends StatelessWidget {
  final String message;
  const _HistoryError({required this.message});

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline_rounded, size: 28, color: AppColors.muted),
            const SizedBox(height: 10),
            Text(
              'Could not load history:\n$message',
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 12, color: AppColors.muted),
            ),
          ],
        ),
      ),
    );
  }
}
