import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/room_capture_service.dart';

import '../../../shared/providers/app_state_provider.dart';
import '../../../shared/widgets/app_footer_nav.dart';

import '../widgets/furniture_segment_card.dart';

class ResultsScreen extends ConsumerStatefulWidget {
  const ResultsScreen({super.key});

  @override
  ConsumerState<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends ConsumerState<ResultsScreen> {
  bool _saving = false;
  bool _saved  = false;

  Future<void> _saveCapture() async {
    final appState = ref.read(appStateProvider);

    if (!appState.hasUnsavedCapture) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No room scan or AR photo to save')),
      );
      return;
    }

    setState(() => _saving = true);

    try {
      final service = RoomCaptureService();
      final arState = appState.arCaptureState;

      if (arState != null) {
        // Photo + spatial data together (merges LiDAR scan data if present too)
        await service.saveCapture(
          capture: arState,
          spatialData: appState.scanSpatialData ?? const {},
        );
      } else if (appState.scanSpatialData != null) {
        // LiDAR scan only, no photo was taken
        await service.saveScanOnly(spatialData: appState.scanSpatialData!);
      }

      ref.read(appStateProvider.notifier).clearCaptureBuffer();

      if (mounted) {
        setState(() { _saving = false; _saved = true; });
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('Room data saved'),
            backgroundColor: Colors.green,
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        setState(() => _saving = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Save failed: $e'),
            backgroundColor: Colors.red.shade800,
          ),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {

    final appState =
        ref.watch(appStateProvider);

    final segmentedFurniture = appState.segmentedFurniture;

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
                      '${segmentedFurniture.length} Furniture Detected',
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

              if (segmentedFurniture.isEmpty)
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
                  itemCount: segmentedFurniture.length,
                  gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
                    crossAxisCount: 2,
                    mainAxisSpacing: 12,
                    crossAxisSpacing: 12,
                    childAspectRatio: 0.92,
                  ),
                  itemBuilder: (context, index) {
                    return FurnitureSegmentCard(item: segmentedFurniture[index]);
                  },
                ),

              if (appState.hasUnsavedCapture) ...[
                const SizedBox(height: 24),
                SizedBox(
                  width: double.infinity,
                  child: FilledButton.icon(
                    style: FilledButton.styleFrom(
                      backgroundColor: _saved ? AppColors.sage : AppColors.sageDeep,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                      shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(14)),
                    ),
                    icon: _saving
                        ? const SizedBox(
                            width: 18, height: 18,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white),
                          )
                        : Icon(_saved ? Icons.check_circle : Icons.cloud_upload_rounded),
                    label: Text(
                      _saving ? 'Saving…' : (_saved ? 'Saved' : 'Save Room Data'),
                      style: const TextStyle(
                          fontSize: 16, fontWeight: FontWeight.bold),
                    ),
                    onPressed: (_saving || _saved) ? null : _saveCapture,
                  ),
                ),
              ],
            ],
          ),
        ),
      ),

      bottomNavigationBar: const AppFooterNav(current: FooterTab.none),
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
            // No heart here: saving is the "Save Room Data" button below. (The
            // heart used to flip is_saved; hiding it is deliberate.)
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
