import 'dart:convert';
import 'dart:io';

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

    final products = appState.matchedProducts;

    return Scaffold(
      backgroundColor: AppColors.background,

      appBar: AppBar(
        backgroundColor: AppColors.background,
        title: const Text(
          'Decorated Room',
          style: TextStyle(
            fontSize: 18,
            fontWeight: FontWeight.bold,
            color: AppColors.ink,
          ),
        ),
      ),

      body: Padding(
        padding: const EdgeInsets.fromLTRB(18, 4, 18, 18),

        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment:
                CrossAxisAlignment.start,

            children: [

              _GeneratedRoomImage(
                imageData: appState.generatedRoomImage,
                originalImage: appState.uploadedImage,
              ),

              const SizedBox(height: 18),

              Container(
                padding: const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 8,
                ),

                decoration: BoxDecoration(
                  color: AppColors.brassTint,
                  borderRadius: BorderRadius.circular(999),
                ),

                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.sell_rounded, size: 15, color: AppColors.brassDeep),
                    const SizedBox(width: 6),
                    Text(
                      '${products.length} Matching Product Found',
                      style: const TextStyle(
                        color: AppColors.brassDeep,
                        fontSize: 12.5,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),

              const SizedBox(height: 18),

              if (products.isEmpty)
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 30),
                  child: Center(
                    child: Text(
                      'No matching products yet',
                      style: TextStyle(color: AppColors.muted),
                    ),
                  ),
                )
              else
                GridView.builder(
                  shrinkWrap: true,
                  physics: const NeverScrollableScrollPhysics(),
                  itemCount: products.length,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 12,
                    crossAxisSpacing: 12,
                    childAspectRatio: 0.92,
                  ),
                  itemBuilder: (context, index) {
                    return ProductCard(product: products[index]);
                  },
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _GeneratedRoomImage extends StatefulWidget {
  final String? imageData;
  final File? originalImage;

  const _GeneratedRoomImage({
    required this.imageData,
    required this.originalImage,
  });

  @override
  State<_GeneratedRoomImage> createState() => _GeneratedRoomImageState();
}

class _GeneratedRoomImageState extends State<_GeneratedRoomImage> {
  bool _showAfter = true;

  @override
  Widget build(BuildContext context) {
    final data = widget.imageData;
    final hasBefore = widget.originalImage != null;

    Widget image;
    if (_showAfter && data != null && data.startsWith('data:image')) {
      final base64Data = data.substring(data.indexOf(',') + 1);
      image = Image.memory(
        base64Decode(base64Data),
        height: 240,
        width: double.infinity,
        fit: BoxFit.cover,
      );
    } else if (!_showAfter && hasBefore) {
      image = Image.file(
        widget.originalImage!,
        height: 240,
        width: double.infinity,
        fit: BoxFit.cover,
      );
    } else {
      image = Container(
        height: 240,
        width: double.infinity,
        color: AppColors.sandTint,
        child: const Center(
          child: Icon(Icons.chair_rounded, size: 100, color: AppColors.muted),
        ),
      );
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: Stack(
        children: [
          image,

          Positioned(
            top: 10,
            right: 10,
            child: Row(
              children: [
                _RoundIconButton(icon: Icons.favorite_border_rounded, onTap: () {}),
                const SizedBox(width: 8),
                _RoundIconButton(icon: Icons.ios_share_rounded, onTap: () {}),
              ],
            ),
          ),

          if (hasBefore)
            Positioned(
              left: 10,
              bottom: 10,
              child: Container(
                padding: const EdgeInsets.all(4),
                decoration: BoxDecoration(
                  color: AppColors.ink.withValues(alpha: 0.55),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    _ToggleChip(
                      label: 'After',
                      selected: _showAfter,
                      onTap: () => setState(() => _showAfter = true),
                    ),
                    _ToggleChip(
                      label: 'Before',
                      selected: !_showAfter,
                      onTap: () => setState(() => _showAfter = false),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}

class _RoundIconButton extends StatelessWidget {
  final IconData icon;
  final VoidCallback onTap;
  const _RoundIconButton({required this.icon, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.white.withValues(alpha: 0.9),
      shape: const CircleBorder(),
      child: InkWell(
        customBorder: const CircleBorder(),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(7),
          child: Icon(icon, size: 16, color: AppColors.sageDeep),
        ),
      ),
    );
  }
}

class _ToggleChip extends StatelessWidget {
  final String label;
  final bool selected;
  final VoidCallback onTap;
  const _ToggleChip({required this.label, required this.selected, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
        decoration: BoxDecoration(
          color: selected ? AppColors.sage : Colors.transparent,
          borderRadius: BorderRadius.circular(999),
        ),
        child: Text(
          label,
          style: TextStyle(
            fontSize: 10.5,
            fontWeight: FontWeight.bold,
            color: selected ? Colors.white : const Color(0xFFE7E2D5),
          ),
        ),
      ),
    );
  }
}
