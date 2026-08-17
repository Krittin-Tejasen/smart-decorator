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
      ..style = PaintingStyle.fill;

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
      ..strokeJoin = StrokeJoin.round;

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

/// A hand holding a phone, reaching up from the bottom of its box - the LiDAR
/// scan hero illustration on the home screen.
class LidarHandIllustration extends StatelessWidget {
  final double width;
  final double height;
  final Color handColor;
  final Color phoneCaseColor;
  final Color phoneScreenColor;
  final Color bracketColor;

  const LidarHandIllustration({
    super.key,
    this.width = 260,
    this.height = 100,
    this.handColor = const Color(0xFFE7C9A0),
    this.phoneCaseColor = const Color(0xFF17140F),
    this.phoneScreenColor = const Color(0xFFEEF3F1),
    required this.bracketColor,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: width,
      height: height,
      child: CustomPaint(
        size: Size(width, height),
        painter: _LidarHandPainter(
          handColor: handColor,
          phoneCaseColor: phoneCaseColor,
          phoneScreenColor: phoneScreenColor,
          bracketColor: bracketColor,
        ),
      ),
    );
  }
}

class _LidarHandPainter extends CustomPainter {
  final Color handColor;
  final Color phoneCaseColor;
  final Color phoneScreenColor;
  final Color bracketColor;

  _LidarHandPainter({
    required this.handColor,
    required this.phoneCaseColor,
    required this.phoneScreenColor,
    required this.bracketColor,
  });

  // Reference art was drawn against a 260x92 canvas - scale everything from there.
  static const _refW = 260.0;
  static const _refH = 92.0;

  @override
  void paint(Canvas canvas, Size size) {
    final sx = size.width / _refW;
    final sy = size.height / _refH;
    canvas.save();
    canvas.scale(sx, sy);

    final armPaint = Paint()
      ..color = handColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 15
      ..strokeCap = StrokeCap.round;

    final arm = Path()
      ..moveTo(100, 92)
      ..cubicTo(100, 66, 108, 52, 126, 52)
      ..cubicTo(144, 52, 132, 64, 156, 64)
      ..cubicTo(166, 64, 166, 56, 180, 56)
      ..cubicTo(194, 56, 200, 66, 200, 80);
    canvas.drawPath(arm, armPaint);

    final caseRect = RRect.fromRectAndRadius(
      const Rect.fromLTWH(150, 8, 30, 52),
      const Radius.circular(7),
    );
    canvas.drawRRect(caseRect, Paint()..color = phoneCaseColor);

    final screenRect = RRect.fromRectAndRadius(
      const Rect.fromLTWH(153, 13, 24, 38),
      const Radius.circular(2),
    );
    canvas.drawRRect(screenRect, Paint()..color = phoneScreenColor);

    canvas.save();
    canvas.translate(158, 19);
    final bracketPaint = Paint()
      ..color = bracketColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.6
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    const armLen = 5.0;
    const box = 17.0;
    void bracket(Offset c, Offset h, Offset v) {
      canvas.drawPath(
        Path()
          ..moveTo(h.dx, h.dy)
          ..lineTo(c.dx, c.dy)
          ..lineTo(v.dx, v.dy),
        bracketPaint,
      );
    }

    bracket(const Offset(0, 0), const Offset(armLen, 0), const Offset(0, armLen));
    bracket(const Offset(box, 0), Offset(box - armLen, 0), Offset(box, armLen));
    bracket(Offset(box, box), Offset(box - armLen, box), Offset(box, box - armLen));
    bracket(Offset(0, box), Offset(armLen, box), Offset(0, box - armLen));
    canvas.restore();

    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant _LidarHandPainter oldDelegate) =>
      oldDelegate.handColor != handColor ||
      oldDelegate.phoneCaseColor != phoneCaseColor ||
      oldDelegate.phoneScreenColor != phoneScreenColor ||
      oldDelegate.bracketColor != bracketColor;
}
