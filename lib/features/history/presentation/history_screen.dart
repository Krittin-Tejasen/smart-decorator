import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/design_repository.dart';
import '../../../shared/models/design_labels.dart';
import '../../../shared/models/saved_design.dart';
import '../../../shared/widgets/app_footer_nav.dart';
import '../providers/history_providers.dart';

class HistoryScreen extends ConsumerWidget {

  const HistoryScreen({super.key});

  Future<void> _confirmAndDelete(
    BuildContext context,
    WidgetRef ref,
    SavedDesign design,
  ) async {
    final messenger = ScaffoldMessenger.of(context);

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete this room?'),
        content: const Text(
          'It is removed from your saved rooms, and its images are deleted.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(false),
            child: const Text('Cancel'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(true),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
    if (confirmed != true) return;

    try {
      await ref.read(designRepositoryProvider).delete(design);
      ref.invalidate(savedDesignsProvider);
      ref.invalidate(savedDesignFurnitureProvider(design));
      messenger.showSnackBar(const SnackBar(content: Text('Room deleted')));
    } catch (e) {
      messenger.showSnackBar(
        SnackBar(content: Text('Could not delete room: $e')),
      );
    }
  }

  @override
  Widget build(
    BuildContext context,
    WidgetRef ref,
  ) {

    final designsAsync = ref.watch(savedDesignsProvider);
    final limit = ref.watch(designRepositoryProvider).maxSavedDesigns;

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
                // One extra row on top: how full the saved rooms are.
                itemCount: designs.length + 1,
                itemBuilder: (context, index) {
                  if (index == 0) {
                    return Padding(
                      padding: const EdgeInsets.only(bottom: 12, left: 2),
                      child: Text(
                        '${designs.length} of $limit saved rooms',
                        key: const ValueKey('saved-count'),
                        style: const TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                          color: AppColors.muted,
                        ),
                      ),
                    );
                  }

                  final design = designs[index - 1];
                  return _HistoryCard(
                    item: design,
                    onTap: () => context.push('/saved-design', extra: design),
                    onDelete: () => _confirmAndDelete(context, ref, design),
                  );
                },
              ),
        loading: () => const Center(child: CircularProgressIndicator()),
        error: (error, _) => _HistoryError(message: '$error'),
      ),

      bottomNavigationBar: const AppFooterNav(current: FooterTab.history),
    );
  }
}

class _HistoryCard extends ConsumerWidget {
  final SavedDesign item;
  final VoidCallback onTap;
  final VoidCallback onDelete;

  const _HistoryCard({
    required this.item,
    required this.onTap,
    required this.onDelete,
  });

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
  Widget build(BuildContext context, WidgetRef ref) {
    final imageUrl = ref.watch(savedImageUrlProvider(item.generatedImagePath));
    final icon = Icon(_roomIcon, size: 26, color: _tileIconColor);

    return Container(
      margin: const EdgeInsets.only(bottom: 12),

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

      // The ripple needs a Material above the white decoration.
      child: Material(
        type: MaterialType.transparency,
        child: InkWell(
          borderRadius: BorderRadius.circular(18),
          onTap: onTap,
          child: Padding(
            padding: const EdgeInsets.fromLTRB(12, 12, 4, 12),
            child: Row(
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(14),
                  child: Container(
                    width: 58,
                    height: 58,
                    color: _tileColor,
                    child: imageUrl.when(
                      data: (url) => Image.network(
                        url,
                        fit: BoxFit.cover,
                        errorBuilder: (context, error, stackTrace) => Center(child: icon),
                        loadingBuilder: (context, child, progress) =>
                            progress == null ? child : Center(child: icon),
                      ),
                      loading: () => Center(child: icon),
                      error: (error, stackTrace) => Center(child: icon),
                    ),
                  ),
                ),

                const SizedBox(width: 14),

                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        roomTypeTitle(item.roomType),
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
                            label: 'Style: ${styleTitle(item.style)}',
                            bg: _tileColor,
                            fg: _tileIconColor,
                          ),
                        ],
                      ),

                      const SizedBox(height: 6),

                      Text(
                        formatSavedAt(item.createdAt),
                        style: const TextStyle(fontSize: 11, color: AppColors.muted),
                      ),
                    ],
                  ),
                ),

                IconButton(
                  key: ValueKey('delete-${item.id}'),
                  tooltip: 'Delete',
                  icon: const Icon(Icons.delete_outline_rounded, color: AppColors.muted),
                  onPressed: onDelete,
                ),
              ],
            ),
          ),
        ),
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
            'Tap Save Room Data on a generated design to save it here',
            textAlign: TextAlign.center,
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
