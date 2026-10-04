import 'package:flutter/material.dart';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/auth_service.dart';
import '../../../core/services/design_repository.dart';
import '../../../core/services/room_capture_service.dart';

import '../../../shared/providers/app_state_provider.dart';
import '../../../shared/utils/data_url.dart';

import '../widgets/room_results_view.dart';

class ResultsScreen extends ConsumerStatefulWidget {
  const ResultsScreen({super.key});

  @override
  ConsumerState<ResultsScreen> createState() => _ResultsScreenState();
}

class _ResultsScreenState extends ConsumerState<ResultsScreen> {
  // Decoded once per generated image, not on every rebuild (a fresh
  // MemoryImage each time would decode and flicker on every state change).
  String? _afterSource;
  ImageProvider? _afterProvider;

  ImageProvider? _afterFor(String? dataUrl) {
    if (!identical(dataUrl, _afterSource)) {
      _afterSource = dataUrl;
      final bytes = decodeDataUrl(dataUrl);
      _afterProvider = bytes == null ? null : MemoryImage(bytes);
    }
    return _afterProvider;
  }

  /// "Save Room Data": saves the generated room and its photo to the user's
  /// saved rooms (max 5), plus the buffered LiDAR / AR scan data when there is
  /// any. Nothing is stored before this is tapped.
  Future<void> _save() async {
    final messenger = ScaffoldMessenger.of(context);
    final router = GoRouter.of(context);
    final notifier = ref.read(appStateProvider.notifier);
    final hasScanData = ref.read(appStateProvider).hasUnsavedCapture;

    try {
      final result = await notifier.saveCurrentRoom();

      String? scanProblem;
      if (hasScanData) {
        try {
          await _saveScanData();
        } catch (e) {
          scanProblem = '$e';
        }
      }

      messenger.showSnackBar(
        SnackBar(
          content: Text(
            scanProblem == null
                ? 'Room data saved (${result.savedCount} of ${result.limit})'
                : 'Room saved, but the scan data could not be saved: $scanProblem',
          ),
          backgroundColor: scanProblem == null ? Colors.green : Colors.orange.shade800,
        ),
      );
    } on SaveLimitReachedException catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text('$e'),
          action: SnackBarAction(
            label: 'History',
            onPressed: () => router.go('/history'),
          ),
        ),
      );
    } on NotSignedInException catch (e) {
      messenger.showSnackBar(SnackBar(content: Text('$e')));
    } catch (e) {
      messenger.showSnackBar(
        SnackBar(
          content: Text('Save failed: $e'),
          backgroundColor: Colors.red.shade800,
        ),
      );
    }
  }

  /// The LiDAR / AR capture data held in memory since the scan.
  Future<void> _saveScanData() async {
    final appState = ref.read(appStateProvider);
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
  }

  @override
  Widget build(BuildContext context) {
    final appState = ref.watch(appStateProvider);

    final original = appState.uploadedImage;

    return RoomResultsView(
      afterImage: _afterFor(appState.generatedRoomImage),
      beforeImage: original == null ? null : FileImage(original),
      furniture: AsyncValue.data(appState.segmentedFurniture),
      footer: appState.generatedRoomImage == null
          ? null
          : _SaveRoomButton(
              saving: appState.isSavingDesign,
              saved: appState.designSaved,
              onPressed: _save,
            ),
    );
  }
}

class _SaveRoomButton extends StatelessWidget {
  final bool saving;
  final bool saved;
  final VoidCallback onPressed;

  const _SaveRoomButton({
    required this.saving,
    required this.saved,
    required this.onPressed,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: double.infinity,
      child: FilledButton.icon(
        style: FilledButton.styleFrom(
          backgroundColor: saved ? AppColors.sage : AppColors.sageDeep,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
        ),
        icon: saving
            ? const SizedBox(
                width: 18,
                height: 18,
                child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
              )
            : Icon(saved ? Icons.check_circle : Icons.cloud_upload_rounded),
        label: Text(
          saving ? 'Saving…' : (saved ? 'Saved' : 'Save Room Data'),
          style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
        ),
        onPressed: (saving || saved) ? null : onPressed,
      ),
    );
  }
}
