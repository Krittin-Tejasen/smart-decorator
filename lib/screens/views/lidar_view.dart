import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import '../widgets/scan_widgets.dart';

enum _ScanStep { floor, height, done }

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
  double? _ceilingY;   // world-space Y of the ceiling sample
  bool _isBusy = false;

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

  Future<void> _clearPoints() async {
    await _channel?.invokeMethod('clearPoints');
    setState(() {
      _cornerCount = 0;
      _lastDistance = 0;
      _floorPlanPoints = [];
      _allPoints3D = [];
      _ceilingY = null;
      _step = _ScanStep.floor;
    });
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
          _step = _ScanStep.done;
        });
      }
    } on PlatformException catch (e) {
      debugPrint('samplePoint error: ${e.message}');
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  // ── Step 3: Done — capture snapshot and navigate ─────────────────────────

  Future<void> _onDone() async {
    if (_channel == null) return;
    setState(() => _isBusy = true);

    Map<String, dynamic>? snapshot;
    try {
      final raw = await _channel!.invokeMethod<Map>('captureSnapshot');
      if (raw != null) snapshot = Map<String, dynamic>.from(raw);
    } on PlatformException catch (e) {
      debugPrint('captureSnapshot error: ${e.message}');
    }

    if (!mounted) return;
    setState(() => _isBusy = false);

    // Compute floor average Y from 3-D points
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
        return 'Now point at the ceiling or top of a wall, then tap Mark Height';
      case _ScanStep.done:
        final h = _ceilingY != null && _allPoints3D.isNotEmpty
            ? (_ceilingY! -
                    _allPoints3D.fold<double>(
                            0, (s, p) => s + (p['y'] as num).toDouble()) /
                        _allPoints3D.length)
                .abs()
            : 0.0;
        return 'Height marked: ${_fmtDistance(h)} · Tap Done to save';
    }
  }

  @override
  Widget build(BuildContext context) {
    final canMarkHeight = _cornerCount >= 3 && _step == _ScanStep.floor;
    final canDone = _step == _ScanStep.done;

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

        // Top instruction bar
        Positioned(
          top: 0, left: 0, right: 0,
          child: Container(
            color: Colors.black54,
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: SafeArea(
              bottom: false,
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Step indicator dots
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
                          done: _step == _ScanStep.done, label: '2 Height'),
                      const SizedBox(width: 4),
                      Container(width: 20, height: 2,
                          color: _step == _ScanStep.done
                              ? Colors.cyanAccent : Colors.white24),
                      const SizedBox(width: 4),
                      _StepDot(active: _step == _ScanStep.done,
                          done: false, label: '3 Done'),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    _instruction,
                    style: const TextStyle(color: Colors.white, fontSize: 13),
                    textAlign: TextAlign.center,
                  ),
                ],
              ),
            ),
          ),
        ),

        // Crosshair
        const Center(child: ScanCrosshair()),

        // Last distance (bottom-left)
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

        // Mini floor plan (bottom-right)
        Positioned(
          bottom: 104, right: 16,
          child: FloorPlanMini(points: _floorPlanPoints),
        ),

        // Bottom control bar
        Positioned(
          bottom: 0, left: 0, right: 0,
          child: Container(
            color: Colors.black87,
            padding: EdgeInsets.fromLTRB(
                16, 12, 16, 12 + MediaQuery.of(context).padding.bottom),
            child: _step == _ScanStep.floor
                ? _floorBar(canMarkHeight)
                : _step == _ScanStep.height
                    ? _heightBar()
                    : _doneBar(canDone),
          ),
        ),
      ],
    );
  }

  // ── Bottom bar: step 1 (floor) ───────────────────────────────────────────
  Widget _floorBar(bool canMarkHeight) => Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          TextButton.icon(
            onPressed: _cornerCount > 0 ? _clearPoints : null,
            icon: const Icon(Icons.refresh_rounded, color: Colors.white60, size: 20),
            label: const Text('Reset', style: TextStyle(color: Colors.white60)),
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

          // Mark height button
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

          const SizedBox(width: 72), // balance
        ],
      );

  // ── Bottom bar: step 3 (done) ────────────────────────────────────────────
  Widget _doneBar(bool canDone) => Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          TextButton.icon(
            onPressed: () => setState(() => _step = _ScanStep.height),
            icon: const Icon(Icons.arrow_back_rounded, color: Colors.white60, size: 20),
            label: const Text('Redo Height',
                style: TextStyle(color: Colors.white60, fontSize: 12)),
          ),

          GestureDetector(
            onTap: (_isBusy || !canDone) ? null : _onDone,
            child: Container(
              width: 64, height: 64,
              decoration: BoxDecoration(
                color: Colors.greenAccent,
                shape: BoxShape.circle,
                boxShadow: [BoxShadow(
                    color: Colors.greenAccent.withValues(alpha: 0.45),
                    blurRadius: 18, spreadRadius: 2)],
              ),
              child: _isBusy
                  ? const Padding(
                      padding: EdgeInsets.all(18),
                      child: CircularProgressIndicator(
                          strokeWidth: 2, color: Colors.black))
                  : const Icon(Icons.check_rounded, color: Colors.black, size: 36),
            ),
          ),

          const SizedBox(width: 72),
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
