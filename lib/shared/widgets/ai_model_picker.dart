import 'package:flutter/material.dart';

import '../../core/constants/app_colors.dart';
import '../models/ai_model.dart';

/// A compact "Model: X ▾" pill, in the style of Claude's model picker -
/// tapping it opens a bottom sheet to pick which AI image-generation
/// provider /generate-room should use for this request.
class AiModelPicker extends StatelessWidget {
  final AiModel selected;
  final ValueChanged<AiModel> onSelected;

  const AiModelPicker({
    super.key,
    required this.selected,
    required this.onSelected,
  });

  Future<void> _openPicker(BuildContext context) async {
    final picked = await showModalBottomSheet<AiModel>(
      context: context,
      backgroundColor: AppColors.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(22)),
      ),
      builder: (ctx) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(8, 10, 8, 8),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Center(
                  child: Container(
                    width: 36,
                    height: 4,
                    margin: const EdgeInsets.only(bottom: 14),
                    decoration: BoxDecoration(
                      color: AppColors.muted.withValues(alpha: 0.3),
                      borderRadius: BorderRadius.circular(4),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 12),
                  child: Text(
                    'AI Model',
                    style: TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w600,
                      letterSpacing: 0.6,
                      color: AppColors.ink,
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                ...AiModel.all.map((model) {
                  final isSelected = model.providerValue == selected.providerValue;
                  return InkWell(
                    borderRadius: BorderRadius.circular(14),
                    onTap: () => Navigator.of(ctx).pop(model),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 12,
                        vertical: 10,
                      ),
                      child: Row(
                        children: [
                          Container(
                            width: 34,
                            height: 34,
                            decoration: BoxDecoration(
                              color: isSelected
                                  ? AppColors.sageTint
                                  : AppColors.sandTint,
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Icon(
                              _iconFor(model),
                              size: 17,
                              color: isSelected
                                  ? AppColors.sageDeep
                                  : AppColors.brassDeep,
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                Text(
                                  model.title,
                                  style: TextStyle(
                                    fontSize: 14,
                                    fontWeight: FontWeight.w600,
                                    color: AppColors.ink,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  model.subtitle,
                                  style: const TextStyle(
                                    fontSize: 11.5,
                                    color: AppColors.muted,
                                  ),
                                ),
                              ],
                            ),
                          ),
                          if (isSelected)
                            const Icon(
                              Icons.check_rounded,
                              size: 20,
                              color: AppColors.sageDeep,
                            ),
                        ],
                      ),
                    ),
                  );
                }),
              ],
            ),
          ),
        );
      },
    );

    if (picked != null) {
      onSelected(picked);
    }
  }

  IconData _iconFor(AiModel model) {
    switch (model.providerValue) {
      case 'mock':
        return Icons.visibility_outlined;
      case 'replicate':
        return Icons.bolt_rounded;
      case 'gemini':
        return Icons.auto_awesome_rounded;
      default:
        return Icons.tune_rounded;
    }
  }

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: () => _openPicker(context),
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: AppColors.surface,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: AppColors.muted.withValues(alpha: 0.25),
            width: 1,
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(_iconFor(selected), size: 15, color: AppColors.brassDeep),
            const SizedBox(width: 6),
            Text(
              selected.title,
              style: const TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w600,
                color: AppColors.ink,
              ),
            ),
            const SizedBox(width: 4),
            const Icon(
              Icons.keyboard_arrow_down_rounded,
              size: 16,
              color: AppColors.muted,
            ),
          ],
        ),
      ),
    );
  }
}
