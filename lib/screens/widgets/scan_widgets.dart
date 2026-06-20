import 'dart:math';
import 'package:flutter/material.dart';

class ScanCrosshair extends StatelessWidget {
  const ScanCrosshair({super.key});

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 48,
      height: 48,
      child: CustomPaint(painter: _CrosshairPainter()),
    );
  }
}

class _CrosshairPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.yellow
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    final cx = size.width / 2;
    final cy = size.height / 2;
    canvas.drawLine(Offset(cx - 14, cy), Offset(cx + 14, cy), paint);
    canvas.drawLine(Offset(cx, cy - 14), Offset(cx, cy + 14), paint);
    canvas.drawCircle(Offset(cx, cy), 5, paint);
  }

  @override
  bool shouldRepaint(covariant CustomPainter old) => false;
}

class FloorPlanMini extends StatelessWidget {
  final List<Offset> points;
  final Color lineColor;
  final Color dotColor;

  const FloorPlanMini({
    super.key,
    required this.points,
    this.lineColor = Colors.cyanAccent,
    this.dotColor = Colors.yellow,
  });

  @override
  Widget build(BuildContext context) {
    if (points.isEmpty) return const SizedBox.shrink();
    return Container(
      width: 120,
      height: 120,
      decoration: BoxDecoration(
        color: Colors.black87,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white24),
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: CustomPaint(
          painter: FloorPlanPainter(
            points: points,
            lineColor: lineColor,
            dotColor: dotColor,
          ),
        ),
      ),
    );
  }
}

class FloorPlanPainter extends CustomPainter {
  final List<Offset> points;
  final Color lineColor;
  final Color dotColor;

  const FloorPlanPainter({
    required this.points,
    this.lineColor = Colors.cyanAccent,
    this.dotColor = Colors.yellow,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (points.isEmpty) return;

    const pad = 14.0;
    final xs = points.map((p) => p.dx).toList();
    final ys = points.map((p) => p.dy).toList();
    final minX = xs.reduce(min);
    final maxX = xs.reduce(max);
    final minY = ys.reduce(min);
    final maxY = ys.reduce(max);
    final rangeX = maxX - minX;
    final rangeY = maxY - minY;

    final scaleX = rangeX > 0.001 ? (size.width - pad * 2) / rangeX : 50.0;
    final scaleY = rangeY > 0.001 ? (size.height - pad * 2) / rangeY : 50.0;
    final scale = min(scaleX, scaleY);

    final drawW = rangeX > 0.001 ? rangeX * scale : 0.0;
    final drawH = rangeY > 0.001 ? rangeY * scale : 0.0;
    final offsetX = pad + (size.width - pad * 2 - drawW) / 2;
    final offsetY = pad + (size.height - pad * 2 - drawH) / 2;

    Offset toCanvas(Offset p) => Offset(
          offsetX + (p.dx - minX) * scale,
          offsetY + (p.dy - minY) * scale,
        );

    final canvasPoints = points.map(toCanvas).toList();

    final linePaint = Paint()
      ..color = lineColor
      ..strokeWidth = 1.5
      ..style = PaintingStyle.stroke;

    final dotPaint = Paint()
      ..color = dotColor
      ..style = PaintingStyle.fill;

    for (int i = 1; i < canvasPoints.length; i++) {
      canvas.drawLine(canvasPoints[i - 1], canvasPoints[i], linePaint);
    }
    if (canvasPoints.length >= 3) {
      canvas.drawLine(
        canvasPoints.last,
        canvasPoints.first,
        linePaint..color = lineColor.withValues(alpha: 0.35),
      );
    }
    for (final p in canvasPoints) {
      canvas.drawCircle(p, 3.5, dotPaint);
    }
  }

  @override
  bool shouldRepaint(covariant FloorPlanPainter old) =>
      old.points != points || old.lineColor != lineColor || old.dotColor != dotColor;
}
