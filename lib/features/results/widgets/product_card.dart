import 'package:flutter/material.dart';

import '../../../../core/constants/app_colors.dart';

import '../../../../shared/models/product.dart';

class ProductCard extends StatelessWidget {

  final Product product;

  const ProductCard({
    super.key,
    required this.product,
  });

  IconData get _icon {
    final key = product.imageUrl.toLowerCase();
    if (key.contains('sofa') || key.contains('chair')) return Icons.weekend_rounded;
    if (key.contains('lamp') || key.contains('light')) return Icons.emoji_objects_outlined;
    if (key.contains('table') || key.contains('desk')) return Icons.table_restaurant_rounded;
    if (key.contains('bed')) return Icons.bed_rounded;
    if (key.contains('rug') || key.contains('carpet')) return Icons.crop_square_rounded;
    return Icons.category_rounded;
  }

  Color get _tileColor {
    final index = product.id.hashCode % 3;
    switch (index) {
      case 0:
        return AppColors.sageTint;
      case 1:
        return AppColors.sandTint;
      default:
        return AppColors.brassTint;
    }
  }

  Color get _iconColor {
    final index = product.id.hashCode % 3;
    switch (index) {
      case 0:
        return AppColors.sageDeep;
      case 1:
        return AppColors.brassDeep;
      default:
        return AppColors.brassDeep;
    }
  }

  @override
  Widget build(BuildContext context) {

    return Container(
      padding: const EdgeInsets.all(10),

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

      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            height: 64,
            width: double.infinity,
            decoration: BoxDecoration(
              color: _tileColor,
              borderRadius: BorderRadius.circular(10),
            ),
            child: Icon(_icon, size: 26, color: _iconColor),
          ),

          const SizedBox(height: 8),

          Text(
            product.name,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 11.5,
              fontWeight: FontWeight.w600,
              color: AppColors.ink,
            ),
          ),

          const SizedBox(height: 2),

          Text(
            '฿${product.price.toStringAsFixed(0)}',
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.bold,
              color: AppColors.sageDeep,
            ),
          ),
        ],
      ),
    );
  }
}
