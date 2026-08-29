import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:go_router/go_router.dart';
import '../widgets/scan_widgets.dart';

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
  bool _isSaving = false;

  void _onPlatformViewCreated(int viewId) {
    _channel = MethodChannel('com.smartdeco.app/ar_camera_$viewId');
  }

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
          _floorPlanPoints = rawPoints.map((p) {
            final m = Map<String, dynamic>.from(p as Map);
            return Offset(
              (m['x'] as num).toDouble(),
              (m['z'] as num).toDouble(),
            );
          }).toList();
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
    });
  }

  Future<void> _onDone() async {
    if (_channel == null || _cornerCount < 3) return;
    setState(() => _isSaving = true);

    Map<String, dynamic>? snapshot;
    try {
      final raw = await _channel!.invokeMethod<Map>('captureSnapshot');
      if (raw != null) {
        snapshot = Map<String, dynamic>.from(raw);
      }
    } on PlatformException catch (e) {
      debugPrint('captureSnapshot error: ${e.message}');
    }

    if (!mounted) return;
    setState(() => _isSaving = false);

    context.push('/scan_result', extra: {
      'floorPlanPoints': _floorPlanPoints
          .map((o) => {'x': o.dx, 'y': o.dy})
          .toList(),
      'snapshot': snapshot,
      'cornerCount': _cornerCount,
    });
  }

  String _fmtDistance(double m) {
    if (m < 0.01) return '—';
    if (m >= 1.0) return '${m.toStringAsFixed(2)} m';
    return '${(m * 100).toStringAsFixed(0)} cm';
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        Positioned.fill(
          child: UiKitView(
            viewType: 'ar_camera_view',
            onPlatformViewCreated: _onPlatformViewCreated,
            creationParamsCodec: const StandardMessageCodec(),
          ),
        ),

        Positioned(
          top: 0, left: 0, right: 0,
          child: Container(
            color: Colors.black54,
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 12),
            child: SafeArea(
              bottom: false,
              child: Text(
                _cornerCount == 0
                    ? 'Aim crosshair at a room corner, then tap +'
                    : '$_cornerCount corner${_cornerCount > 1 ? 's' : ''} marked  ·  Aim at next corner',
                style: const TextStyle(color: Colors.white, fontSize: 15),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),

        const Center(child: ScanCrosshair()),

        if (_lastDistance > 0.01)
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
                  Text(
                    _fmtDistance(_lastDistance),
                    style: const TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
          ),

        Positioned(
          bottom: 104, right: 16,
          child: FloorPlanMini(points: _floorPlanPoints),
        ),

        Positioned(
          bottom: 0, left: 0, right: 0,
          child: Container(
            color: Colors.black87,
            padding: EdgeInsets.fromLTRB(
                24, 16, 24, 16 + MediaQuery.of(context).padding.bottom),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                TextButton.icon(
                  onPressed: _cornerCount > 0 ? _clearPoints : null,
                  icon: const Icon(Icons.refresh_rounded,
                      color: Colors.white60, size: 20),
                  label: const Text('Reset',
                      style: TextStyle(color: Colors.white60)),
                ),

                GestureDetector(
                  onTap: _addCorner,
                  child: Container(
                    width: 68, height: 68,
                    decoration: BoxDecoration(
                      color: Colors.yellow,
                      shape: BoxShape.circle,
                      boxShadow: [
                        BoxShadow(
                            color: Colors.yellow.withValues(alpha: 0.45),
                            blurRadius: 18,
                            spreadRadius: 2),
                      ],
                    ),
                    child: const Icon(Icons.add_rounded,
                        color: Colors.black, size: 38),
                  ),
                ),

                TextButton.icon(
                  onPressed: (_cornerCount >= 3 && !_isSaving) ? _onDone : null,
                  icon: _isSaving
                      ? const SizedBox(
                          width: 16, height: 16,
                          child: CircularProgressIndicator(
                              strokeWidth: 2, color: Colors.greenAccent))
                      : Icon(Icons.check_rounded,
                          color: _cornerCount >= 3
                              ? Colors.greenAccent
                              : Colors.white30,
                          size: 20),
                  label: Text('Done',
                      style: TextStyle(
                          color: _cornerCount >= 3
                              ? Colors.greenAccent
                              : Colors.white30)),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}
