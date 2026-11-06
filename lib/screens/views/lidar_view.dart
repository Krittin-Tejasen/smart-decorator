import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import '../../core/models/ar_capture_state.dart';
import '../widgets/scan_widgets.dart';

// "photo" replaces the old standalone "done" step: capturing (or skipping)
// the upload photo here — in the SAME AR session as the floor/ceiling taps —
// is what finishes the scan. A separately-captured photo would have
// unrelated extrinsics (ARKit resets world tracking per session), which
// would silently break furniture-size unprojection later.
enum _ScanStep { floor, height, photo }

class LidarView extends StatefulWidget {
  const LidarView({super.key});

  @override
  State<LidarView> createState() => _LidarViewState();
}

class _LidarViewState extends State<LidarView> {
  MethodChannel? _channel;
  int _cornerCount = 0;
  double _lastDistance = 0;
  List<Offset> _floorPlanPoints = [];
  List<Map<String, dynamic>> _allPoints3D = [];

  _ScanStep _step = _ScanStep.floor;
  double? _ceilingY;
  bool _isBusy = false;

  // Holds the last captured photo so the user can retake before confirming.
  ARCaptureState? _pendingCapture;

  void _onPlatformViewCreated(int viewId) {
    _channel = MethodChannel('com.smartdeco.app/ar_camera_$viewId');
  }

  // ── Step 1: mark floor corners ──────────────────────────────────────────

  Future<void> _addCorner() async {
    if (_channel == null) return;
    try {
      final result = await _channel!.invokeMethod<Map>('addPoint');
      if (result == null) return;
      final rawPoints = result['allPoints'] as List?;
      setState(() {
        _cornerCount = (result['points'] as num?)?.toInt() ?? _cornerCount;
        _lastDistance = (result['distance'] as num?)?.toDouble() ?? 0;
        if (rawPoints != null) {
          _allPoints3D = rawPoints
              .map((p) => Map<String, dynamic>.from(p as Map))
              .toList();
          _floorPlanPoints = _allPoints3D.map((m) => Offset(
                (m['x'] as num).toDouble(),
                (m['z'] as num).toDouble(),
              )).toList();
        }
      });
    } on PlatformException catch (e) {
      debugPrint('addPoint error: ${e.message}');
    }
  }

  Future<void> _undoLastPoint() async {
    if (_channel == null || _cornerCount == 0) return;
    try {
      final result = await _channel!.invokeMethod<Map>('removeLastPoint');
      if (result == null) return;
      final rawPoints = result['allPoints'] as List?;
      setState(() {
        _cornerCount = (result['points'] as num?)?.toInt() ?? 0;
        _lastDistance = (result['distance'] as num?)?.toDouble() ?? 0;
        if (rawPoints != null) {
          _allPoints3D = rawPoints
              .map((p) => Map<String, dynamic>.from(p as Map))
              .toList();
          _floorPlanPoints = _allPoints3D.map((m) => Offset(
                (m['x'] as num).toDouble(),
                (m['z'] as num).toDouble(),
              )).toList();
        }
      });
    } on PlatformException catch (e) {
      debugPrint('removeLastPoint error: ${e.message}');
    }
  }

  // ── Step 2: mark ceiling height ─────────────────────────────────────────

  Future<void> _markHeight() async {
    if (_channel == null) return;
    setState(() => _isBusy = true);
    try {
      final result = await _channel!.invokeMethod<Map>('samplePoint');
      if (result != null) {
        setState(() {
          _ceilingY = (result['y'] as num).toDouble();
          _step = _ScanStep.photo;
        });
      }
    } on PlatformException catch (e) {
      debugPrint('samplePoint error: ${e.message}');
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  // ── Step 3: Capture ─────────────────────────────────────────────────────

  /// Shutter: capture photo in this session and show retake/confirm UI.
  Future<void> _capturePhoto() async {
    if (_channel == null || _isBusy) return;
    setState(() => _isBusy = true);
    try {
      final raw = await _channel!.invokeMethod<Map>('capturePhoto');
      if (raw != null) {
        setState(() => _pendingCapture = ARCaptureState.fromMap(raw));
      }
    } on PlatformException catch (e) {
      debugPrint('capturePhoto error: ${e.message}');
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  void _retakePhoto() => setState(() => _pendingCapture = null);

  Future<void> _confirmPhotoAndFinish() async {
    await _finish(captured: _pendingCapture);
  }

  Future<void> _skipPhotoAndFinish() async {
    if (_isBusy) return;
    setState(() => _isBusy = true);
    await _finish(captured: null);
  }

  Future<void> _finish({required ARCaptureState? captured}) async {
    if (_channel == null) return;

    Map<String, dynamic>? snapshot;
    try {
      final raw = await _channel!.invokeMethod<Map>('captureSnapshot');
      if (raw != null) snapshot = Map<String, dynamic>.from(raw);
    } on PlatformException catch (e) {
      debugPrint('captureSnapshot error: ${e.message}');
    }

    if (!mounted) return;
    setState(() => _isBusy = false);

    double? floorY;
    if (_allPoints3D.isNotEmpty) {
      final sumY = _allPoints3D.fold<double>(
          0, (s, p) => s + (p['y'] as num).toDouble());
      floorY = sumY / _allPoints3D.length;
    }

    context.push('/scan_result', extra: {
      'floorPlanPoints': _floorPlanPoints
          .map((o) => {'x': o.dx, 'y': o.dy})
          .toList(),
      'snapshot': snapshot,
      'cornerCount': _cornerCount,
      'ceilingY': _ceilingY,
      'floorY': floorY,
      'capturedPhoto': captured,
    });
  }

  // ── UI helpers ───────────────────────────────────────────────────────────

  String _fmtDistance(double m) {
    if (m < 0.01) return '—';
    if (m >= 1.0) return '${m.toStringAsFixed(2)} m';
    return '${(m * 100).toStringAsFixed(0)} cm';
  }

  String get _instruction {
    switch (_step) {
      case _ScanStep.floor:
        return _cornerCount == 0
            ? 'Point at a floor corner, then tap +'
            : '$_cornerCount corner${_cornerCount > 1 ? 's' : ''} marked · point at next corner';
      case _ScanStep.height:
        return 'Point at the ceiling or top of a wall, then tap Mark Height';
      case _ScanStep.photo:
        if (_pendingCapture != null) {
          return 'Photo captured · confirm or retake';
        }
        final h = _ceilingY != null && _allPoints3D.isNotEmpty
            ? (_ceilingY! -
                    _allPoints3D.fold<double>(
                            0, (s, p) => s + (p['y'] as num).toDouble()) /
                        _allPoints3D.length)
                .abs()
            : 0.0;
        return 'Height: ${_fmtDistance(h)} · Rotate phone left, frame the room';
    }
  }

  @override
  Widget build(BuildContext context) {
    final canMarkHeight = _cornerCount >= 3 && _step == _ScanStep.floor;
    final isPhotoStep = _step == _ScanStep.photo;

    return Stack(
      children: [
        // AR view
        Positioned.fill(
          child: UiKitView(
            viewType: 'ar_camera_view',
            onPlatformViewCreated: _onPlatformViewCreated,
            creationParamsCodec: const StandardMessageCodec(),
          ),
        ),

        // Top instruction bar — transparent in photo step
        Positioned(
          top: 0, left: 0, right: 0,
          child: Container(
            color: isPhotoStep ? Colors.black.withValues(alpha: 0.28) : Colors.black54,
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: SafeArea(
              bottom: false,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      _StepDot(active: _step == _ScanStep.floor,
                          done: _step != _ScanStep.floor, label: '1 Floor'),
                      const SizedBox(width: 4),
                      Container(width: 20, height: 2,
                          color: _step != _ScanStep.floor
                              ? Colors.cyanAccent : Colors.white24),
                      const SizedBox(width: 4),
                      _StepDot(active: _step == _ScanStep.height,
                          done: _step == _ScanStep.photo, label: '2 Height'),
                      const SizedBox(width: 4),
                      Container(width: 20, height: 2,
                          color: _step == _ScanStep.photo
                              ? Colors.cyanAccent : Colors.white24),
                      const SizedBox(width: 4),
                      _StepDot(active: _step == _ScanStep.photo,
                          done: false, label: '3 Photo'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _instruction,
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 13,
                      shadows: isPhotoStep
                          ? const [Shadow(blurRadius: 6, color: Colors.black)]
                          : null,
                    ),
                    textAlign: TextAlign.center,
                  ),
                  // Rotation guide hint for photo step
                  if (isPhotoStep && _pendingCapture == null) ...[
                    const SizedBox(height: 6),
                    Row(
                      mainAxisAlignment: MainAxisAlignment.center,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Transform.rotate(
                          angle: math.pi / 2,
                          child: const Icon(Icons.stay_current_portrait,
                              color: Colors.cyanAccent, size: 16),
                        ),
                        const SizedBox(width: 6),
                        const Icon(Icons.arrow_back_rounded,
                            color: Colors.cyanAccent, size: 14),
                        const SizedBox(width: 4),
                        const Text(
                          'Rotate left · camera button on right edge',
                          style: TextStyle(
                            color: Colors.cyanAccent,
                            fontSize: 11,
                            shadows: [Shadow(blurRadius: 4, color: Colors.black)],
                          ),
                        ),
                      ],
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),

        // Crosshair
        const Center(child: ScanCrosshair()),

        // Last distance (bottom-left) — floor step only
        if (_lastDistance > 0.01 && _step == _ScanStep.floor)
          Positioned(
            bottom: 104, left: 16,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
              decoration: BoxDecoration(
                color: Colors.black54,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  const Text('Last segment',
                      style: TextStyle(color: Colors.white60, fontSize: 11)),
                  Text(_fmtDistance(_lastDistance),
                      style: const TextStyle(
                          color: Colors.white,
                          fontSize: 22,
                          fontWeight: FontWeight.bold)),
                ],
              ),
            ),
          ),

        // Mini floor plan (bottom-right) — floor step only
        if (_step == _ScanStep.floor)
          Positioned(
            bottom: 104, right: 16,
            child: FloorPlanMini(points: _floorPlanPoints),
          ),

        // Bottom control bar — floor and height steps only
        if (!isPhotoStep)
          Positioned(
            bottom: 0, left: 0, right: 0,
            child: Container(
              color: Colors.black87,
              padding: EdgeInsets.fromLTRB(
                  16, 12, 16, 12 + MediaQuery.of(context).padding.bottom),
              child: _step == _ScanStep.floor
                  ? _floorBar(canMarkHeight)
                  : _heightBar(),
            ),
          ),

        // ── Photo step overlays ────────────────────────────────────────────
        // Right edge: shutter OR retake/confirm buttons
        if (isPhotoStep)
          Positioned(
            right: 20,
            top: MediaQuery.of(context).padding.top,
            bottom: MediaQuery.of(context).padding.bottom,
            child: Center(
              child: _pendingCapture == null
                  ? _shutterColumn()
                  : _confirmColumn(),
            ),
          ),
      ],
    );
  }

  // ── Bottom bar: step 1 (floor) ───────────────────────────────────────────
  Widget _floorBar(bool canMarkHeight) => Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          // Undo last point
          TextButton.icon(
            onPressed: _cornerCount > 0 ? _undoLastPoint : null,
            icon: Icon(Icons.undo_rounded,
                color: _cornerCount > 0 ? Colors.white60 : Colors.white24,
                size: 20),
            label: Text('Undo',
                style: TextStyle(
                    color: _cornerCount > 0 ? Colors.white60 : Colors.white24)),
          ),

          // Add corner button
          GestureDetector(
            onTap: _addCorner,
            child: Container(
              width: 64, height: 64,
              decoration: BoxDecoration(
                color: Colors.yellow,
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(
                    color: Colors.yellow.withValues(alpha: 0.45),
                    blurRadius: 18, spreadRadius: 2)],
              ),
              child: const Icon(Icons.add_rounded, color: Colors.black, size: 36),
            ),
          ),

          // Next step: mark height
          TextButton.icon(
            onPressed: canMarkHeight
                ? () => setState(() => _step = _ScanStep.height)
                : null,
            icon: Icon(Icons.height_rounded,
                color: canMarkHeight ? Colors.cyanAccent : Colors.white30,
                size: 20),
            label: Text('Next',
                style: TextStyle(
                    color: canMarkHeight ? Colors.cyanAccent : Colors.white30)),
          ),
        ],
      );

  // ── Bottom bar: step 2 (height) ──────────────────────────────────────────
  Widget _heightBar() => Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          TextButton.icon(
            onPressed: () => setState(() => _step = _ScanStep.floor),
            icon: const Icon(Icons.arrow_back_rounded, color: Colors.white60, size: 20),
            label: const Text('Back', style: TextStyle(color: Colors.white60)),
          ),

          GestureDetector(
            onTap: _isBusy ? null : _markHeight,
            child: Container(
              width: 64, height: 64,
              decoration: BoxDecoration(
                color: Colors.cyanAccent,
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(
                    color: Colors.cyanAccent.withValues(alpha: 0.45),
                    blurRadius: 18, spreadRadius: 2)],
              ),
              child: _isBusy
                  ? const Padding(
                      padding: EdgeInsets.all(18),
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.black))
                  : const Icon(Icons.arrow_upward_rounded,
                      color: Colors.black, size: 36),
            ),
          ),

          const SizedBox(width: 72),
        ],
      );

  // ── Photo step: right-edge shutter column ────────────────────────────────
  Widget _shutterColumn() => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Redo height (small icon button above shutter)
          GestureDetector(
            onTap: _isBusy ? null : () => setState(() => _step = _ScanStep.height),
            child: Container(
              width: 44, height: 44,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.black.withValues(alpha: 0.35),
                border: Border.all(color: Colors.white38, width: 1.5),
              ),
              child: const Icon(Icons.height_rounded, color: Colors.white70, size: 20),
            ),
          ),
          const SizedBox(height: 20),
          // Shutter button
          GestureDetector(
            onTap: _isBusy ? null : _capturePhoto,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 120),
              width: 72, height: 72,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: _isBusy
                    ? Colors.white.withValues(alpha: 0.1)
                    : Colors.white.withValues(alpha: 0.15),
                border: Border.all(color: Colors.white, width: 4),
                boxShadow: [BoxShadow(
                    color: Colors.white.withValues(alpha: 0.2),
                    blurRadius: 12, spreadRadius: 1)],
              ),
              child: _isBusy
                  ? const Padding(
                      padding: EdgeInsets.all(18),
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.white))
                  : const Icon(Icons.camera_alt_rounded,
                      color: Colors.white, size: 32),
            ),
          ),
          const SizedBox(height: 20),
          // Skip / upload photo
          GestureDetector(
            onTap: _isBusy ? null : _skipPhotoAndFinish,
            child: Container(
              width: 44, height: 44,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.black.withValues(alpha: 0.35),
                border: Border.all(color: Colors.white24, width: 1.5),
              ),
              child: const Icon(Icons.photo_library_outlined,
                  color: Colors.white54, size: 20),
            ),
          ),
        ],
      );

  // ── Photo step: retake / confirm column ──────────────────────────────────
  Widget _confirmColumn() => Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // Use this photo
          GestureDetector(
            onTap: _isBusy ? null : _confirmPhotoAndFinish,
            child: Container(
              width: 72, height: 72,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.greenAccent.withValues(alpha: 0.85),
                boxShadow: [BoxShadow(
                    color: Colors.greenAccent.withValues(alpha: 0.4),
                    blurRadius: 16, spreadRadius: 2)],
              ),
              child: _isBusy
                  ? const Padding(
                      padding: EdgeInsets.all(18),
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.black))
                  : const Icon(Icons.check_rounded, color: Colors.black, size: 36),
            ),
          ),
          const SizedBox(height: 16),
          // Retake
          GestureDetector(
            onTap: _isBusy ? null : _retakePhoto,
            child: Container(
              width: 52, height: 52,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: Colors.black.withValues(alpha: 0.45),
                border: Border.all(color: Colors.white54, width: 2),
              ),
              child: const Icon(Icons.replay_rounded, color: Colors.white70, size: 24),
            ),
          ),
          const SizedBox(height: 8),
          const Text('Retake',
              style: TextStyle(
                  color: Colors.white54, fontSize: 10,
                  shadows: [Shadow(blurRadius: 4, color: Colors.black)])),
        ],
      );
}

// ── Step indicator dot ────────────────────────────────────────────────────────

class _StepDot extends StatelessWidget {
  final bool active, done;
  final String label;
  const _StepDot({required this.active, required this.done, required this.label});

  @override
  Widget build(BuildContext context) {
    final color = done
        ? Colors.cyanAccent
        : active
            ? Colors.white
            : Colors.white30;
    return Column(mainAxisSize: MainAxisSize.min, children: [
      Container(
        width: 8, height: 8,
        decoration: BoxDecoration(shape: BoxShape.circle, color: color),
      ),
      const SizedBox(height: 2),
      Text(label, style: TextStyle(color: color, fontSize: 9)),
    ]);
  }
}
