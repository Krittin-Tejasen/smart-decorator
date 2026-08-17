import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';
import '../../../shared/models/design_history.dart';
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
      backgroundColor: AppColors.background,

      appBar: AppBar(
        backgroundColor: AppColors.background,
        title: const Text(
          'History',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: AppColors.ink,
          ),
        ),
      ),

      body: history.isEmpty
          ? const _EmptyHistory()
          : ListView.builder(

              padding: const EdgeInsets.fromLTRB(18, 4, 18, 18),

              itemCount: history.length,

              itemBuilder: (
                context,
                index,
              ) {

                final item = history[index];

                return _HistoryCard(item: item);
              },
            ),
    );
  }
}

class _HistoryCard extends StatelessWidget {
  final DesignHistory item;
  const _HistoryCard({required this.item});

  IconData get _roomIcon {
    final key = item.roomType.toLowerCase();
    if (key.contains('bed')) return Icons.bed_rounded;
    if (key.contains('dining') || key.contains('table')) return Icons.table_restaurant_rounded;
    return Icons.weekend_rounded;
  }

  Color get _tileColor {
    final key = item.theme.toLowerCase();
    if (key.contains('luxury')) return const Color(0xFFEFE1C9);
    if (key.contains('minimal')) return AppColors.sandTint;
    return AppColors.sageTint;
  }

  Color get _tileIconColor {
    final key = item.theme.toLowerCase();
    if (key.contains('luxury')) return AppColors.brassDeep;
    if (key.contains('minimal')) return const Color(0xFF8A7350);
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
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              color: _tileColor,
              borderRadius: BorderRadius.circular(14),
            ),
            child: Icon(_roomIcon, size: 26, color: _tileIconColor),
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
                      label: 'Theme: ${item.theme}',
                      bg: _tileColor,
                      fg: _tileIconColor,
                    ),
                    _Chip(
                      label: '${item.products.length} Products Matched',
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
        ],
      ),
    );
  }
}
