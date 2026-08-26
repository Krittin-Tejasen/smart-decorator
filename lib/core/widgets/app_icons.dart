import 'package:flutter/material.dart';

/// Four-point sparkle/spark glyph, used on the generating screen.
class SparkleIcon extends StatelessWidget {
  final double size;
  final Color color;

  const SparkleIcon({super.key, this.size = 20, required this.color});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size(size, size),
      painter: _SparklePainter(color),
    );
  }
}

class _SparklePainter extends CustomPainter {
  final Color color;
  _SparklePainter(this.color);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.fill
      ..isAntiAlias = true;

    final w = size.width;
    final h = size.height;

    final path = Path()
      ..moveTo(w * 0.5, h * 0.0)
      ..cubicTo(w * 0.525, h * 0.317, w * 0.683, h * 0.45, w * 1.0, h * 0.5)
      ..cubicTo(w * 0.683, h * 0.55, w * 0.525, h * 0.683, w * 0.5, h * 1.0)
      ..cubicTo(w * 0.475, h * 0.683, w * 0.317, h * 0.55, w * 0.0, h * 0.5)
      ..cubicTo(w * 0.317, h * 0.45, w * 0.475, h * 0.317, w * 0.5, h * 0.0)
      ..close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _SparklePainter oldDelegate) =>
      oldDelegate.color != color;
}

/// Four L-shaped viewfinder brackets, used to represent an AR/LiDAR scan target.
class ScanBracketIcon extends StatelessWidget {
  final double size;
  final Color color;
  final double strokeWidth;

  const ScanBracketIcon({
    super.key,
    this.size = 20,
    required this.color,
    this.strokeWidth = 1.8,
  });

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size(size, size),
      painter: _ScanBracketPainter(color, strokeWidth),
    );
  }
}

class _ScanBracketPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;
  _ScanBracketPainter(this.color, this.strokeWidth);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..isAntiAlias = true;

    final w = size.width;
    final h = size.height;
    final arm = w * 0.28;
    const inset = 0.0;

    void bracket(Offset corner, Offset toH, Offset toV) {
      final path = Path()
        ..moveTo(toH.dx, toH.dy)
        ..lineTo(corner.dx, corner.dy)
        ..lineTo(toV.dx, toV.dy);
      canvas.drawPath(path, paint);
    }

    bracket(
      Offset(inset, inset),
      Offset(inset + arm, inset),
      Offset(inset, inset + arm),
    );
    bracket(
      Offset(w - inset, inset),
      Offset(w - inset - arm, inset),
      Offset(w - inset, inset + arm),
    );
    bracket(
      Offset(w - inset, h - inset),
      Offset(w - inset - arm, h - inset),
      Offset(w - inset, h - inset - arm),
    );
    bracket(
      Offset(inset, h - inset),
      Offset(inset + arm, h - inset),
      Offset(inset, h - inset - arm),
    );
  }

  @override
  bool shouldRepaint(covariant _ScanBracketPainter oldDelegate) =>
      oldDelegate.color != color || oldDelegate.strokeWidth != strokeWidth;
}

/// A simple phone-with-viewfinder graphic for the LiDAR scan card. Deliberately
/// plain (rect + rounded corners + the ScanBracketIcon above) rather than an
/// illustrated scene - compound hand-drawn shapes are the ones that render
/// unreliably across devices; flat geometric shapes hold up.
class LidarScanGraphic extends StatelessWidget {
  final double size;
  final Color caseColor;
  final Color screenColor;
  final Color bracketColor;

  const LidarScanGraphic({
    super.key,
    this.size = 64,
    this.caseColor = const Color(0xFF17140F),
    this.screenColor = const Color(0xFFEEF3F1),
    required this.bracketColor,
  });

  @override
  Widget build(BuildContext context) {
    final screenInset = size * 0.09;
    return Container(
      width: size * 0.62,
      height: size,
      padding: EdgeInsets.all(screenInset),
      decoration: BoxDecoration(
        color: caseColor,
        borderRadius: BorderRadius.circular(size * 0.16),
      ),
      child: Container(
        decoration: BoxDecoration(
          color: screenColor,
          borderRadius: BorderRadius.circular(size * 0.05),
        ),
        padding: EdgeInsets.all(size * 0.1),
        child: ScanBracketIcon(
          size: size * 0.6,
          color: bracketColor,
          strokeWidth: 1.6,
        ),
      ),
    );
  }
}
