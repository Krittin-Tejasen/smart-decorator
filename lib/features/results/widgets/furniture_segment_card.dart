import 'dart:convert';

import 'package:flutter/material.dart';

import '../../../core/constants/app_colors.dart';
import '../../../shared/models/furniture_item.dart';

class FurnitureSegmentCard extends StatelessWidget {

  final FurnitureItem item;

  const FurnitureSegmentCard({
    super.key,
    required this.item,
  });

  String get _label {
    final raw = item.label.trim();
    if (raw.isEmpty) return 'Item';
    return raw
        .split(RegExp(r'[\s_]+'))
        .where((word) => word.isNotEmpty)
        .map((word) => word[0].toUpperCase() + word.substring(1))
        .join(' ');
  }

  @override
  Widget build(BuildContext context) {

    final base64Data = item.cropImage.contains(',')
        ? item.cropImage.substring(item.cropImage.indexOf(',') + 1)
        : item.cropImage;

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
          ClipRRect(
            borderRadius: BorderRadius.circular(10),
            child: AspectRatio(
              aspectRatio: 1,
              child: base64Data.isEmpty
                  ? Container(
                      color: AppColors.sandTint,
                      child: const Icon(
                        Icons.chair_rounded,
                        size: 26,
                        color: AppColors.muted,
                      ),
                    )
                  : Image.memory(
                      base64Decode(base64Data),
                      fit: BoxFit.cover,
                      errorBuilder: (context, error, stackTrace) => Container(
                        color: AppColors.sandTint,
                        child: const Icon(
                          Icons.chair_rounded,
                          size: 26,
                          color: AppColors.muted,
                        ),
                      ),
                    ),
            ),
          ),

          const SizedBox(height: 8),

          Text(
            _label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: const TextStyle(
              fontSize: 11.5,
              fontWeight: FontWeight.w600,
              color: AppColors.ink,
            ),
          ),
        ],
      ),
    );
  }
}
