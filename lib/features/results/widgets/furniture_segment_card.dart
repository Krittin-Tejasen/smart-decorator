import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/product_match_service.dart';
import '../../../shared/models/furniture_item.dart';
import 'furniture_image.dart';

/// One detected piece of furniture. Tapping it opens the page of products that
/// match it ([onTap] overrides that, e.g. in tests).
class FurnitureSegmentCard extends StatelessWidget {

  final FurnitureItem item;
  final VoidCallback? onTap;

  const FurnitureSegmentCard({
    super.key,
    required this.item,
    this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
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

      // The ripple needs its own Material *above* the white decoration,
      // otherwise the decoration paints over it and the tap looks dead.
      child: Material(
        type: MaterialType.transparency,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: onTap ??
              () => context.push('/furniture-matches', extra: item),
          child: Padding(
            padding: const EdgeInsets.all(10),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // The grid fixes the card's height, so the image takes what is
                // left after the label instead of forcing a square (which
                // overflowed the card by several pixels).
                Expanded(
                  child: FurnitureImage(item: item),
                ),

                const SizedBox(height: 8),

                Row(
                  children: [
                    Expanded(
                      child: Text(
                        titleCase(item.label),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                        style: const TextStyle(
                          fontSize: 11.5,
                          fontWeight: FontWeight.w600,
                          color: AppColors.ink,
                        ),
                      ),
                    ),
                    const Icon(
                      Icons.chevron_right_rounded,
                      size: 16,
                      color: AppColors.muted,
                    ),
                  ],
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
