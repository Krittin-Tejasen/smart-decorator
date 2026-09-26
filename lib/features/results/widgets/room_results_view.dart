import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';
import '../../../shared/models/furniture_item.dart';
import '../../../shared/widgets/app_footer_nav.dart';
import 'furniture_segment_card.dart';

/// The "Decorated Room" page: the generated room (with a Before/After toggle
/// when the original photo is known), how much furniture was detected, and a
/// card for each piece. Used for a room just generated and for a saved room
/// opened from History, so both look and behave the same.
class RoomResultsView extends StatelessWidget {
  /// The generated room. Null shows a placeholder (or a spinner while
  /// [afterLoading]).
  final ImageProvider? afterImage;
  final bool afterLoading;

  /// The photo it was made from; null hides the Before/After toggle.
  final ImageProvider? beforeImage;

  /// The detected furniture: data, still being worked out, or failed.
  final AsyncValue<List<FurnitureItem>> furniture;
  final VoidCallback? onRetryFurniture;

  /// Extra content under the furniture, e.g. the Save Room Data button.
  final Widget? footer;

  final FooterTab footerTab;

  const RoomResultsView({
    super.key,
    required this.afterImage,
    this.afterLoading = false,
    this.beforeImage,
    required this.furniture,
    this.onRetryFurniture,
    this.footer,
    this.footerTab = FooterTab.none,
  });

  @override
  Widget build(BuildContext context) {
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
            crossAxisAlignment: CrossAxisAlignment.start,

            children: [
              _GeneratedRoomImage(
                after: afterImage,
                afterLoading: afterLoading,
                before: beforeImage,
              ),

              const SizedBox(height: 18),

              ..._furnitureSection(),

              if (footer != null) ...[
                const SizedBox(height: 24),
                footer!,
              ],
            ],
          ),
        ),
      ),

      bottomNavigationBar: AppFooterNav(current: footerTab),
    );
  }

  List<Widget> _furnitureSection() {
    return furniture.when(
      data: (items) => [
        _CountChip(label: '${items.length} Furniture Detected'),

        const SizedBox(height: 18),

        if (items.isEmpty)
          const Padding(
            padding: EdgeInsets.symmetric(vertical: 30),
            child: Center(
              child: Text(
                'No furniture detected yet',
                style: TextStyle(color: AppColors.muted),
              ),
            ),
          )
        else
          GridView.builder(
            shrinkWrap: true,
            physics: const NeverScrollableScrollPhysics(),
            itemCount: items.length,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              mainAxisSpacing: 12,
              crossAxisSpacing: 12,
              childAspectRatio: 0.92,
            ),
            itemBuilder: (context, index) {
              return FurnitureSegmentCard(item: items[index]);
            },
          ),
      ],

      // A saved room's furniture is worked out when it is opened, which takes
      // a few seconds: show where the cards will appear rather than a blank.
      loading: () => const [
        _CountChip(label: 'Detecting furniture…'),
        SizedBox(height: 18),
        _FurnitureLoading(),
      ],

      error: (error, stackTrace) => [
        Padding(
          key: const ValueKey('furniture-error'),
          padding: const EdgeInsets.symmetric(vertical: 30),
          child: Center(
            child: Column(
              children: [
                const Icon(Icons.error_outline_rounded, size: 28, color: AppColors.muted),
                const SizedBox(height: 10),
                const Text(
                  'Could not detect furniture',
                  style: TextStyle(color: AppColors.muted),
                ),
                if (onRetryFurniture != null)
                  TextButton(
                    onPressed: onRetryFurniture,
                    child: const Text('Try again'),
                  ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

class _CountChip extends StatelessWidget {
  final String label;

  const _CountChip({required this.label});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),

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
            label,
            style: const TextStyle(
              color: AppColors.brassDeep,
              fontSize: 12.5,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }
}

class _FurnitureLoading extends StatelessWidget {
  const _FurnitureLoading();

  @override
  Widget build(BuildContext context) {
    return GridView.builder(
      key: const ValueKey('furniture-loading'),
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: 4,
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        mainAxisSpacing: 12,
        crossAxisSpacing: 12,
        childAspectRatio: 0.92,
      ),
      itemBuilder: (context, index) {
        return Container(
          decoration: BoxDecoration(
            color: AppColors.surface,
            borderRadius: BorderRadius.circular(16),
          ),
          padding: const EdgeInsets.all(10),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(
                child: DecoratedBox(
                  decoration: BoxDecoration(
                    color: AppColors.sandTint,
                    borderRadius: BorderRadius.circular(10),
                  ),
                  child: const SizedBox.expand(),
                ),
              ),
              const SizedBox(height: 8),
              Container(
                height: 10,
                width: 70,
                decoration: BoxDecoration(
                  color: AppColors.sandTint,
                  borderRadius: BorderRadius.circular(6),
                ),
              ),
              const SizedBox(height: 4),
            ],
          ),
        );
      },
    );
  }
}

class _GeneratedRoomImage extends StatefulWidget {
  final ImageProvider? after;
  final bool afterLoading;
  final ImageProvider? before;

  const _GeneratedRoomImage({
    required this.after,
    required this.afterLoading,
    required this.before,
  });

  @override
  State<_GeneratedRoomImage> createState() => _GeneratedRoomImageState();
}

class _GeneratedRoomImageState extends State<_GeneratedRoomImage> {
  bool _showAfter = true;

  Widget _placeholder({bool loading = false}) {
    return Container(
      height: 240,
      width: double.infinity,
      color: AppColors.sandTint,
      child: Center(
        child: loading
            ? const CircularProgressIndicator(strokeWidth: 2)
            : const Icon(Icons.chair_rounded, size: 100, color: AppColors.muted),
      ),
    );
  }

  Widget _photo(ImageProvider provider, {Key? key}) {
    return Image(
      key: key,
      image: provider,
      height: 240,
      width: double.infinity,
      fit: BoxFit.cover,
      gaplessPlayback: true,
      // Network images (a saved room) take a moment: keep the tile, not a jump.
      loadingBuilder: (context, child, progress) =>
          progress == null ? child : _placeholder(loading: true),
      errorBuilder: (context, error, stackTrace) => _placeholder(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final hasBefore = widget.before != null;

    final Widget image;
    if (_showAfter && widget.after != null) {
      image = _photo(widget.after!, key: const ValueKey('room-after'));
    } else if (!_showAfter && hasBefore) {
      image = _photo(widget.before!, key: const ValueKey('room-before'));
    } else {
      image = _placeholder(loading: _showAfter && widget.afterLoading);
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(24),
      child: Stack(
        children: [
          image,

          // No heart here: saving is the "Save Room Data" button below. (The
          // heart used to flip is_saved; hiding it is deliberate.)
          Positioned(
            top: 10,
            right: 10,
            child: Row(
              children: [
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
