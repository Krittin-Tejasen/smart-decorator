import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:go_router/go_router.dart';
import 'package:path_provider/path_provider.dart';

class ScanResultScreen extends StatefulWidget {
  final Map<String, dynamic> data;
  const ScanResultScreen({super.key, required this.data});

  @override
  State<ScanResultScreen> createState() => _ScanResultScreenState();
}

class _ScanResultScreenState extends State<ScanResultScreen> {
  bool _saved = false;
  String? _savedPath;

  // ── Compute room dimensions from XZ floor-plan points ──────────────────

  _RoomDimensions _computeDimensions(List<Offset> pts) {
    if (pts.length < 2) return _RoomDimensions(0, 0, 0);

    double minX = pts.first.dx, maxX = pts.first.dx;
    double minZ = pts.first.dy, maxZ = pts.first.dy;
    for (final p in pts) {
      if (p.dx < minX) minX = p.dx;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy < minZ) minZ = p.dy;
      if (p.dy > maxZ) maxZ = p.dy;
    }

    // Perimeter
    double perimeter = 0;
    for (int i = 0; i < pts.length; i++) {
      final a = pts[i], b = pts[(i + 1) % pts.length];
      perimeter += sqrt(pow(b.dx - a.dx, 2) + pow(b.dy - a.dy, 2));
    }

    // Shoelace area
    double area = 0;
    for (int i = 0; i < pts.length; i++) {
      final a = pts[i], b = pts[(i + 1) % pts.length];
      area += a.dx * b.dy - b.dx * a.dy;
    }
    area = area.abs() / 2;

    return _RoomDimensions(
      (maxX - minX).abs(),
      (maxZ - minZ).abs(),
      area,
    );
  }

  // ── Save scan to local JSON file ────────────────────────────────────────

  Future<void> _saveLocally() async {
    final points = (widget.data['floorPlanPoints'] as List?)
            ?.cast<Map>()
            .toList() ??
        [];
    final snapshot = widget.data['snapshot'] as Map<String, dynamic>?;

    final saveData = {
      'savedAt': DateTime.now().toIso8601String(),
      'cornerCount': widget.data['cornerCount'],
      'captureMode': snapshot?['captureMode'] ?? 'standard',
      'floorPlanPoints': points,
      'hasDepthMap': snapshot?['depthMapPng'] != null,
      'hasMeshAnchors': (snapshot?['meshAnchors'] as List?)?.isNotEmpty == true,
      'snapshot': snapshot,
    };

    final dir = await getApplicationDocumentsDirectory();
    final ts = DateTime.now().millisecondsSinceEpoch;
    final file = File('${dir.path}/scan_$ts.json');
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(saveData));

    setState(() {
      _saved = true;
      _savedPath = file.path;
    });
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  String _fmt(double m) {
    if (m >= 1.0) return '${m.toStringAsFixed(2)} m';
    return '${(m * 100).toStringAsFixed(0)} cm';
  }

  List<Offset> get _points {
    final raw = widget.data['floorPlanPoints'] as List? ?? [];
    return raw.map((p) {
      final m = Map<String, dynamic>.from(p as Map);
      return Offset((m['x'] as num).toDouble(), (m['y'] as num).toDouble());
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final pts = _points;
    final dims = _computeDimensions(pts);
    final snapshot = widget.data['snapshot'] as Map<String, dynamic>?;
    final captureMode = snapshot?['captureMode'] as String? ?? 'standard';
    final isLidar = captureMode == 'lidar';
    final hasDepth = snapshot?['depthMapPng'] != null;
    final meshAnchors = snapshot?['meshAnchors'] as List?;
    final cornerCount = widget.data['cornerCount'] as int? ?? pts.length;

    return Scaffold(
      backgroundColor: const Color(0xFF111318),
      appBar: AppBar(
        backgroundColor: const Color(0xFF1A1D24),
        foregroundColor: Colors.white,
        title: const Text('Scan Result', style: TextStyle(color: Colors.white)),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => context.go('/home'),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [

            // ── Capture mode badge ─────────────────────────────────────
            Row(
              children: [
                _Badge(
                  icon: isLidar ? Icons.radar : Icons.camera_alt,
                  label: isLidar ? 'LiDAR Scan' : 'AR Scan',
                  color: isLidar ? Colors.cyanAccent : Colors.orangeAccent,
                ),
                const SizedBox(width: 8),
                if (hasDepth)
                  _Badge(
                    icon: Icons.layers,
                    label: 'Depth Map',
                    color: Colors.purpleAccent,
                  ),
                if ((meshAnchors?.length ?? 0) > 0)
                  Padding(
                    padding: const EdgeInsets.only(left: 8),
                    child: _Badge(
                      icon: Icons.grid_3x3,
                      label: '${meshAnchors!.length} mesh anchors',
                      color: Colors.greenAccent,
                    ),
                  ),
              ],
            ),

            const SizedBox(height: 20),

            // ── Room dimensions ────────────────────────────────────────
            _SectionTitle('Room Dimensions'),
            const SizedBox(height: 12),
            Row(
              children: [
                Expanded(child: _DimCard('Width', _fmt(dims.width), Icons.swap_horiz)),
                const SizedBox(width: 10),
                Expanded(child: _DimCard('Depth', _fmt(dims.depth), Icons.swap_vert)),
                const SizedBox(width: 10),
                Expanded(child: _DimCard('Area', '${dims.area.toStringAsFixed(1)} m²', Icons.square_foot)),
              ],
            ),

            const SizedBox(height: 20),

            // ── Floor plan preview ─────────────────────────────────────
            _SectionTitle('Floor Plan  ($cornerCount corners)'),
            const SizedBox(height: 12),
            Container(
              height: 200,
              decoration: BoxDecoration(
                color: const Color(0xFF1A1D24),
                borderRadius: BorderRadius.circular(16),
              ),
              child: pts.length >= 2
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(16),
                      child: CustomPaint(
                        painter: _FloorPlanPainter(pts),
                        child: const SizedBox.expand(),
                      ),
                    )
                  : const Center(
                      child: Text('Not enough points',
                          style: TextStyle(color: Colors.white38))),
            ),

            // ── Depth map preview (LiDAR only) ─────────────────────────
            if (hasDepth) ...[
              const SizedBox(height: 20),
              _SectionTitle('Depth Map (LiDAR)'),
              const SizedBox(height: 12),
              ClipRRect(
                borderRadius: BorderRadius.circular(16),
                child: Image.memory(
                  Uint8List.fromList(
                      base64Decode(snapshot!['depthMapPng'] as String)),
                  fit: BoxFit.contain,
                  width: double.infinity,
                ),
              ),
            ],

            // ── Raw data summary ───────────────────────────────────────
            const SizedBox(height: 20),
            _SectionTitle('Captured Data'),
            const SizedBox(height: 12),
            _InfoRow('Capture mode', captureMode),
            _InfoRow('Corners marked', '$cornerCount'),
            _InfoRow('Depth map', hasDepth ? 'Yes (LiDAR)' : 'No'),
            _InfoRow('Mesh anchors', '${meshAnchors?.length ?? 0}'),
            if (snapshot?['depthMinMeters'] != null)
              _InfoRow('Depth range',
                '${(snapshot!['depthMinMeters'] as num).toStringAsFixed(2)} m  –  '
                '${(snapshot['depthMaxMeters'] as num).toStringAsFixed(2)} m'),

            // ── Save button ────────────────────────────────────────────
            const SizedBox(height: 28),
            if (!_saved)
              SizedBox(
                width: double.infinity,
                child: FilledButton.icon(
                  style: FilledButton.styleFrom(
                    backgroundColor: Colors.cyanAccent,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(vertical: 16),
                    shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(14)),
                  ),
                  icon: const Icon(Icons.save_alt),
                  label: const Text('Save to Device',
                      style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                  onPressed: _saveLocally,
                ),
              )
            else ...[
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.green.withValues(alpha: 0.15),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: Colors.greenAccent.withValues(alpha: 0.4)),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Row(children: [
                      Icon(Icons.check_circle, color: Colors.greenAccent, size: 20),
                      SizedBox(width: 8),
                      Text('Saved to device',
                          style: TextStyle(
                              color: Colors.greenAccent,
                              fontWeight: FontWeight.bold)),
                    ]),
                    const SizedBox(height: 6),
                    Text(
                      _savedPath ?? '',
                      style: const TextStyle(color: Colors.white54, fontSize: 11),
                    ),
                  ],
                ),
              ),
            ],

            const SizedBox(height: 16),
            SizedBox(
              width: double.infinity,
              child: OutlinedButton(
                style: OutlinedButton.styleFrom(
                  foregroundColor: Colors.white70,
                  side: const BorderSide(color: Colors.white24),
                  padding: const EdgeInsets.symmetric(vertical: 14),
                  shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(14)),
                ),
                onPressed: () => context.go('/home'),
                child: const Text('Back to Home'),
              ),
            ),
            const SizedBox(height: 32),
          ],
        ),
      ),
    );
  }
}

// ── Small helpers ─────────────────────────────────────────────────────────────

class _RoomDimensions {
  final double width, depth, area;
  _RoomDimensions(this.width, this.depth, this.area);
}

class _SectionTitle extends StatelessWidget {
  final String text;
  const _SectionTitle(this.text);
  @override
  Widget build(BuildContext context) => Text(text,
      style: const TextStyle(
          color: Colors.white70, fontSize: 13, fontWeight: FontWeight.w600,
          letterSpacing: 0.8));
}

class _Badge extends StatelessWidget {
  final IconData icon;
  final String label;
  final Color color;
  const _Badge({required this.icon, required this.label, required this.color});
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.15),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: color.withValues(alpha: 0.5)),
        ),
        child: Row(mainAxisSize: MainAxisSize.min, children: [
          Icon(icon, color: color, size: 14),
          const SizedBox(width: 5),
          Text(label, style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w600)),
        ]),
      );
}

class _DimCard extends StatelessWidget {
  final String label, value;
  final IconData icon;
  const _DimCard(this.label, this.value, this.icon);
  @override
  Widget build(BuildContext context) => Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: const Color(0xFF1A1D24),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Column(children: [
          Icon(icon, color: Colors.white38, size: 18),
          const SizedBox(height: 6),
          Text(value,
              style: const TextStyle(
                  color: Colors.white, fontSize: 16, fontWeight: FontWeight.bold)),
          const SizedBox(height: 2),
          Text(label, style: const TextStyle(color: Colors.white38, fontSize: 11)),
        ]),
      );
}

class _InfoRow extends StatelessWidget {
  final String label, value;
  const _InfoRow(this.label, this.value);
  @override
  Widget build(BuildContext context) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 5),
        child: Row(children: [
          Text(label, style: const TextStyle(color: Colors.white54, fontSize: 13)),
          const Spacer(),
          Text(value,
              style: const TextStyle(
                  color: Colors.white, fontSize: 13, fontWeight: FontWeight.w500)),
        ]),
      );
}

class _FloorPlanPainter extends CustomPainter {
  final List<Offset> points;
  _FloorPlanPainter(this.points);

  @override
  void paint(Canvas canvas, Size size) {
    if (points.length < 2) return;

    double minX = points.first.dx, maxX = points.first.dx;
    double minY = points.first.dy, maxY = points.first.dy;
    for (final p in points) {
      if (p.dx < minX) minX = p.dx;
      if (p.dx > maxX) maxX = p.dx;
      if (p.dy < minY) minY = p.dy;
      if (p.dy > maxY) maxY = p.dy;
    }

    final rangeX = (maxX - minX).abs().clamp(0.001, double.infinity);
    final rangeY = (maxY - minY).abs().clamp(0.001, double.infinity);
    const pad = 24.0;

    Offset toCanvas(Offset p) => Offset(
          pad + (p.dx - minX) / rangeX * (size.width - pad * 2),
          pad + (p.dy - minY) / rangeY * (size.height - pad * 2),
        );

    final fill = Paint()
      ..color = Colors.cyanAccent.withValues(alpha: 0.08)
      ..style = PaintingStyle.fill;
    final stroke = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2;
    final dot = Paint()..color = Colors.yellowAccent;

    final path = Path()..moveTo(toCanvas(points.first).dx, toCanvas(points.first).dy);
    for (final p in points.skip(1)) {
      final c = toCanvas(p);
      path.lineTo(c.dx, c.dy);
    }
    path.close();

    canvas.drawPath(path, fill);
    canvas.drawPath(path, stroke);

    for (final p in points) {
      canvas.drawCircle(toCanvas(p), 5, dot);
    }
  }

  @override
  bool shouldRepaint(_FloorPlanPainter old) => old.points != points;
}
