import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../../core/models/ar_capture_state.dart';
import '../../core/services/ar_capture_service.dart';

/// Full-screen AR photo capture screen.
///
/// iOS:   Shows a live ARKit camera view (UiKitView) with a shutter button.
///        Tapping shutter calls `capturePhoto` on the per-view MethodChannel.
///
/// Android: Shows a loading spinner while [ARCaptureService.captureAndroid]
///        launches the native [ARCoreCaptureActivity]. Pops automatically when
///        the activity returns.
///
/// Pops with [ARCaptureState] on success, or `null` if cancelled.
class ARPhotoCaptureView extends StatefulWidget {
  const ARPhotoCaptureView({super.key});

  @override
  State<ARPhotoCaptureView> createState() => _ARPhotoCaptureViewState();
}

class _ARPhotoCaptureViewState extends State<ARPhotoCaptureView> {
  MethodChannel? _viewChannel;
  bool _capturing = false;
  String? _error;

  // ── Android: fire & forget to native Activity ─────────────────────────

  @override
  void initState() {
    super.initState();
    if (Platform.isAndroid) {
      WidgetsBinding.instance.addPostFrameCallback((_) => _launchAndroid());
    }
  }

  Future<void> _launchAndroid() async {
    try {
      final result = await ARCaptureService.captureAndroid();
      if (mounted) Navigator.of(context).pop(result);
    } catch (e) {
      if (mounted) setState(() => _error = e.toString());
    }
  }

  // ── iOS: per-view MethodChannel ───────────────────────────────────────

  void _onPlatformViewCreated(int viewId) {
    _viewChannel = MethodChannel('com.smartdeco.app/ar_camera_$viewId');
  }

  Future<void> _capturePhoto() async {
    if (_viewChannel == null || _capturing) return;
    setState(() => _capturing = true);
    try {
      final raw = await _viewChannel!.invokeMethod<Map>('capturePhoto');
      if (raw != null && mounted) {
        Navigator.of(context).pop(ARCaptureState.fromMap(raw));
      }
    } on PlatformException catch (e) {
      if (mounted) setState(() => _error = e.message ?? 'Capture failed');
    } finally {
      if (mounted) setState(() => _capturing = false);
    }
  }

  // ── Build ─────────────────────────────────────────────────────────────

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.of(context).padding.top;

    // Android: native Activity owns the UI; show minimal loading screen
    if (Platform.isAndroid) {
      return Scaffold(
        backgroundColor: Colors.black,
        body: Center(
          child: _error != null
              ? _ErrorView(_error!, onClose: () => Navigator.of(context).pop())
              : const _LaunchingView(),
        ),
      );
    }

    // iOS: live ARKit camera + shutter button overlay
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Live AR camera (reuses the same UiKitView type as the LiDAR scan screen)
          UiKitView(
            viewType: 'ar_camera_view',
            onPlatformViewCreated: _onPlatformViewCreated,
            creationParamsCodec: const StandardMessageCodec(),
          ),

          // Top bar
          Positioned(
            top: top + 8,
            left: 8,
            child: IconButton(
              icon: const Icon(Icons.close, color: Colors.white, size: 28),
              onPressed: () => Navigator.of(context).pop(null),
            ),
          ),

          // Hint
          Positioned(
            top: top + 60,
            left: 0,
            right: 0,
            child: const Center(
              child: Text(
                'Point at the room and tap the shutter',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 14,
                  fontWeight: FontWeight.w500,
                  shadows: [Shadow(blurRadius: 6, color: Colors.black54)],
                ),
              ),
            ),
          ),

          // Error banner
          if (_error != null)
            Positioned(
              top: top + 100,
              left: 24,
              right: 24,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                decoration: BoxDecoration(
                  color: Colors.red.shade900.withValues(alpha: 0.85),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Text(_error!,
                    style: const TextStyle(color: Colors.white, fontSize: 13)),
              ),
            ),

          // Shutter button
          Positioned(
            bottom: 56,
            left: 0,
            right: 0,
            child: Center(
              child: GestureDetector(
                onTap: _capturePhoto,
                child: AnimatedContainer(
                  duration: const Duration(milliseconds: 120),
                  width: 76,
                  height: 76,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _capturing
                        ? Colors.white.withValues(alpha: 0.3)
                        : Colors.white.withValues(alpha: 0.15),
                    border: Border.all(color: Colors.white, width: 4),
                  ),
                  child: _capturing
                      ? const Center(
                          child: SizedBox(
                            width: 32,
                            height: 32,
                            child: CircularProgressIndicator(
                                color: Colors.white, strokeWidth: 3),
                          ),
                        )
                      : const Icon(Icons.camera_alt_rounded,
                          color: Colors.white, size: 32),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _LaunchingView extends StatelessWidget {
  const _LaunchingView();
  @override
  Widget build(BuildContext context) => const Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          CircularProgressIndicator(color: Colors.white),
          SizedBox(height: 20),
          Text('Opening AR Camera…',
              style: TextStyle(color: Colors.white70, fontSize: 16)),
        ],
      );
}

class _ErrorView extends StatelessWidget {
  final String message;
  final VoidCallback onClose;
  const _ErrorView(this.message, {required this.onClose});

  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Icon(Icons.error_outline, color: Colors.redAccent, size: 48),
            const SizedBox(height: 16),
            Text(message,
                textAlign: TextAlign.center,
                style: const TextStyle(color: Colors.white70, fontSize: 14)),
            const SizedBox(height: 24),
            TextButton(
              onPressed: onClose,
              child:
                  const Text('Close', style: TextStyle(color: Colors.white)),
            ),
          ],
        ),
      );
}
