import 'dart:convert';

/// Spatial metadata captured at the exact moment a photo is taken via AR camera.
/// Carries everything needed to ray-cast AI-generated furniture back into 3-D space.
class ARCaptureState {
  /// Absolute path to the saved JPEG on device.
  final String imagePath;

  /// 4×4 camera-to-world transform, stored as 16 doubles in **column-major** order.
  /// Column 3 (indices 12-15) is the camera position: [tx, ty, tz, 1].
  /// ARKit: ARFrame.camera.transform  |  ARCore: Camera.pose.toMatrix()
  final List<double> extrinsics;

  /// 3×3 camera intrinsics, stored as 9 doubles in **row-major** order:
  ///   [fx,  0, cx,
  ///    0,  fy, cy,
  ///    0,   0,  1]
  /// ARKit: ARFrame.camera.intrinsics  |  ARCore: Camera.imageIntrinsics
  final List<double> intrinsics;

  final int imageWidth;
  final int imageHeight;

  /// "ios" | "android"
  final String platform;

  final String capturedAt;

  const ARCaptureState({
    required this.imagePath,
    required this.extrinsics,
    required this.intrinsics,
    required this.imageWidth,
    required this.imageHeight,
    required this.platform,
    required this.capturedAt,
  });

  // ── Convenience accessors ──────────────────────────────────────────────

  /// Focal lengths in pixels: (fx, fy)
  (double, double) get focalLength => (intrinsics[0], intrinsics[4]);

  /// Principal point in pixels: (cx, cy)
  (double, double) get principalPoint => (intrinsics[2], intrinsics[5]);

  /// Camera world position extracted from extrinsics column 3.
  (double, double, double) get cameraPosition =>
      (extrinsics[12], extrinsics[13], extrinsics[14]);

  /// Approximate camera height above world origin (Y axis, metres).
  double get cameraHeightMeters => extrinsics[13];

  // ── Serialisation ──────────────────────────────────────────────────────

  Map<String, dynamic> toJson() => {
        'imagePath': imagePath,
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
        'imageWidth': imageWidth,
        'imageHeight': imageHeight,
        'platform': platform,
        'capturedAt': capturedAt,
      };

  String toJsonString() => const JsonEncoder.withIndent('  ').convert(toJson());

  factory ARCaptureState.fromMap(Map<dynamic, dynamic> map) => ARCaptureState(
        imagePath: map['imagePath'] as String,
        extrinsics: (map['extrinsics'] as List).map((e) => (e as num).toDouble()).toList(),
        intrinsics: (map['intrinsics'] as List).map((e) => (e as num).toDouble()).toList(),
        imageWidth: (map['imageWidth'] as num).toInt(),
        imageHeight: (map['imageHeight'] as num).toInt(),
        platform: map['platform'] as String? ?? 'unknown',
        capturedAt: map['capturedAt'] as String? ?? DateTime.now().toIso8601String(),
      );
}
