import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import '../widgets/scan_widgets.dart';

class ARVisionView extends StatefulWidget {
  const ARVisionView({super.key});

  @override
  State<ARVisionView> createState() => _ARVisionViewState();
}

class _ARVisionViewState extends State<ARVisionView> {
  CameraController? _controller;
  bool _cameraReady = false;
  String? _cameraError;

  // Normalized [0,1] screen positions of each corner tap
  final List<Offset> _normCorners = [];

  @override
  void initState() {
    super.initState();
    _initCamera();
  }

  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) {
        setState(() => _cameraError = 'No camera found on this device.');
        return;
      }
      final back = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );
      final controller = CameraController(
        back,
        ResolutionPreset.high,
        enableAudio: false,
      );
      await controller.initialize();
      if (!mounted) return;
      setState(() {
        _controller = controller;
        _cameraReady = true;
      });
    } catch (e) {
      if (mounted) setState(() => _cameraError = 'Camera error: $e');
    }
  }

  @override
  void dispose() {
    _controller?.dispose();
    super.dispose();
  }

  void _onTap(TapDownDetails details, BoxConstraints constraints) {
    final norm = Offset(
      details.localPosition.dx / constraints.maxWidth,
      details.localPosition.dy / constraints.maxHeight,
    );
    setState(() => _normCorners.add(norm));
  }

  void _undoLast() {
    if (_normCorners.isNotEmpty) setState(() => _normCorners.removeLast());
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Camera preview layer
        Positioned.fill(child: _buildCameraLayer()),

        // Tap overlay — draws dots + lines and captures taps
        Positioned.fill(
          child: LayoutBuilder(
            builder: (ctx, constraints) => GestureDetector(
              behavior: HitTestBehavior.translucent,
              onTapDown: (d) => _onTap(d, constraints),
              child: CustomPaint(
                painter: _CornerOverlayPainter(
                  normCorners: _normCorners,
                  canvasSize: Size(constraints.maxWidth, constraints.maxHeight),
                ),
              ),
            ),
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
              child: Text(
                _normCorners.isEmpty
                    ? 'Tap on each room corner to mark it'
                    : '${_normCorners.length} corner${_normCorners.length > 1 ? 's' : ''} marked  ·  Tap to add more',
                style: const TextStyle(color: Colors.white, fontSize: 15),
                textAlign: TextAlign.center,
              ),
            ),
          ),
        ),

        // Center crosshair
        const Center(child: ScanCrosshair()),

        // Corner count badge (bottom-left)
        if (_normCorners.isNotEmpty)
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
                  const Text('Corners', style: TextStyle(color: Colors.white60, fontSize: 11)),
                  Text(
                    '${_normCorners.length}',
                    style: const TextStyle(color: Colors.white, fontSize: 22, fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
          ),

        // Mini floor plan (bottom-right) — uses normalized positions as 2D plan
        Positioned(
          bottom: 104, right: 16,
          child: FloorPlanMini(
            points: _normCorners,
            lineColor: Colors.orangeAccent,
            dotColor: Colors.orange,
          ),
        ),

        // Bottom control bar
        Positioned(
          bottom: 0, left: 0, right: 0,
          child: Container(
            color: Colors.black87,
            padding: EdgeInsets.fromLTRB(
              24, 16, 24,
              16 + MediaQuery.of(context).padding.bottom,
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                // Undo last corner
                TextButton.icon(
                  onPressed: _normCorners.isNotEmpty ? _undoLast : null,
                  icon: const Icon(Icons.undo_rounded, color: Colors.white60, size: 20),
                  label: const Text('Undo', style: TextStyle(color: Colors.white60)),
                ),

                // Visual hint — tap anywhere to add corner
                Container(
                  width: 68, height: 68,
                  decoration: BoxDecoration(
                    color: Colors.orange,
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(color: Colors.orange.withValues(alpha: 0.45), blurRadius: 18, spreadRadius: 2),
                    ],
                  ),
                  child: const Icon(Icons.touch_app_rounded, color: Colors.white, size: 34),
                ),

                // Done (needs ≥ 3 corners)
                TextButton.icon(
                  onPressed: _normCorners.length >= 3
                      ? () => Navigator.of(context).pop(_normCorners)
                      : null,
                  icon: Icon(Icons.check_rounded,
                      color: _normCorners.length >= 3 ? Colors.greenAccent : Colors.white30, size: 20),
                  label: Text('Done',
                      style: TextStyle(
                          color: _normCorners.length >= 3 ? Colors.greenAccent : Colors.white30)),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildCameraLayer() {
    if (_cameraError != null) {
      return ColoredBox(
        color: Colors.black,
        child: Center(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Text(_cameraError!, style: const TextStyle(color: Colors.white70)),
          ),
        ),
      );
    }
    if (!_cameraReady || _controller == null) {
      return const ColoredBox(
        color: Colors.black,
        child: Center(child: CircularProgressIndicator(color: Colors.orange)),
      );
    }
    return CameraPreview(_controller!);
  }
}

class _CornerOverlayPainter extends CustomPainter {
  final List<Offset> normCorners;
  final Size canvasSize;

  const _CornerOverlayPainter({required this.normCorners, required this.canvasSize});

  @override
  void paint(Canvas canvas, Size size) {
    if (normCorners.isEmpty) return;

    final pts = normCorners
        .map((n) => Offset(n.dx * canvasSize.width, n.dy * canvasSize.height))
        .toList();

    final linePaint = Paint()
      ..color = Colors.orangeAccent
      ..strokeWidth = 2.5
      ..style = PaintingStyle.stroke;

    final dotFill = Paint()..color = Colors.orange..style = PaintingStyle.fill;
    final dotStroke = Paint()
      ..color = Colors.white
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;

    // Draw connecting lines
    for (int i = 1; i < pts.length; i++) {
      canvas.drawLine(pts[i - 1], pts[i], linePaint);
    }
    if (pts.length >= 3) {
      canvas.drawLine(pts.last, pts.first, linePaint..color = Colors.orangeAccent.withValues(alpha: 0.35));
    }

    // Draw corner dots with number labels
    final labelStyle = const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.bold);
    for (int i = 0; i < pts.length; i++) {
      canvas.drawCircle(pts[i], 10, dotFill);
      canvas.drawCircle(pts[i], 10, dotStroke);

      final span = TextSpan(text: '${i + 1}', style: labelStyle);
      final tp = TextPainter(text: span, textDirection: TextDirection.ltr)..layout();
      tp.paint(canvas, pts[i] - Offset(tp.width / 2, tp.height / 2));
    }
  }

  @override
  bool shouldRepaint(covariant _CornerOverlayPainter old) =>
      old.normCorners != normCorners;
}
