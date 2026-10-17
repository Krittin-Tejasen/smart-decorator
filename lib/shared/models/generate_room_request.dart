class GenerateRoomRequest {

  final String roomType;
  final String style;
  final String color;
  final String imagePath;
  final String? provider;

  // ── Spatial data for furniture-size estimation ───────────────────────────
  // Only present when the uploaded photo came from a LiDAR scan session
  // (see AppState.arCaptureState / scanSpatialData). All null otherwise —
  // the backend computes no dimensions in that case, exactly like today.

  /// 4×4 camera-to-world transform, 16 doubles, column-major.
  final List<double>? extrinsics;

  /// 3×3 camera intrinsics, 9 doubles, row-major.
  final List<double>? intrinsics;

  /// Base64 greyscale depth map PNG (closer = brighter).
  final String? depthMapPng;
  final double? depthMinMeters;
  final double? depthMaxMeters;

  /// Floor-plan corner points (world X/Z), used to align furniture bounding
  /// boxes to the room's own axes rather than raw world axes.
  final List<Map<String, dynamic>>? floorPlanPoints;

  GenerateRoomRequest({
    required this.roomType,
    required this.style,
    required this.color,
    required this.imagePath,
    this.provider,
    this.extrinsics,
    this.intrinsics,
    this.depthMapPng,
    this.depthMinMeters,
    this.depthMaxMeters,
    this.floorPlanPoints,
  });

  /// True when there's enough spatial data to ask the backend for furniture
  /// dimensions alongside segmentation.
  bool get hasSpatialData =>
      extrinsics != null &&
      intrinsics != null &&
      depthMapPng != null &&
      depthMinMeters != null &&
      depthMaxMeters != null &&
      floorPlanPoints != null;

  Map<String, dynamic> toJson() {
    return {
      'room_type': roomType,
      'style': style,
      'color': color,
      'image_path': imagePath,
      if (provider != null) 'provider': provider,
      if (extrinsics != null) 'camera_extrinsics': extrinsics,
      if (intrinsics != null) 'camera_intrinsics': intrinsics,
      if (depthMapPng != null) 'depth_map_png': depthMapPng,
      if (depthMinMeters != null) 'depth_min_m': depthMinMeters,
      if (depthMaxMeters != null) 'depth_max_m': depthMaxMeters,
      if (floorPlanPoints != null) 'floor_plan_points': floorPlanPoints,
    };
  }
}
