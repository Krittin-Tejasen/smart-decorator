import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

/// Coordinates the native iOS channels and persists scan data to Supabase.
class RoomScanService {
  static const _toolsChannel = MethodChannel('com.smartdeco.app/room_tools');

  final _db = Supabase.instance.client;

  // ── Native calls ──────────────────────────────────────────────────────────

  /// Captures the current AR frame state (planes + camera).
  /// Call this BEFORE the user navigates away from the LiDAR view.
  Future<Map<String, dynamic>> captureARSnapshot(MethodChannel arChannel) async {
    final result = await arChannel.invokeMethod<Map>('captureSnapshot');
    if (result == null) throw Exception('captureSnapshot returned null');
    return Map<String, dynamic>.from(result);
  }

  /// Presents the native RoomPlan scanner. Resolves when the user taps "Done".
  /// Requires iOS 16 + LiDAR. Returns null on unsupported devices.
  Future<Map<String, dynamic>?> runRoomPlanScan() async {
    try {
      final result = await _toolsChannel.invokeMethod<Map>('startRoomPlanScan');
      return result != null ? Map<String, dynamic>.from(result) : null;
    } on PlatformException catch (e) {
      if (e.code == 'UNSUPPORTED') return null;
      rethrow;
    }
  }

  /// Measures a furniture item from a 2-D bbox (pixel coords) and a saved snapshot.
  Future<FurnitureSizeResult?> measureFurniture({
    required double xMin,
    required double yMin,
    required double xMax,
    required double yMax,
    required Map<String, dynamic> snapshotMap,
  }) async {
    try {
      final result = await _toolsChannel.invokeMethod<Map>('measureFurniture', {
        'xMin': xMin,
        'yMin': yMin,
        'xMax': xMax,
        'yMax': yMax,
        'snapshotJson': jsonEncode(snapshotMap),
      });
      if (result == null) return null;
      return FurnitureSizeResult.fromMap(Map<String, dynamic>.from(result));
    } on PlatformException catch (e) {
      if (e.code == 'NO_FLOOR') return null;
      rethrow;
    }
  }

  /// Renders a ControlNet depth map. Returns a data:image/png;base64,… string.
  Future<String?> renderDepthMap({
    required Map<String, dynamic> roomPlanMap,
    required Map<String, dynamic> snapshotMap,
    int width = 512,
    int height = 512,
  }) async {
    return _toolsChannel.invokeMethod<String>('renderDepthMap', {
      'roomPlanJson': jsonEncode(roomPlanMap),
      'snapshotJson': jsonEncode(snapshotMap),
      'width': width,
      'height': height,
    });
  }

  /// Returns true if the given box collides with any of the existing boxes.
  Future<bool> checkCollision({
    required Map<String, dynamic> newBox,
    required List<Map<String, dynamic>> existingBoxes,
  }) async {
    final result = await _toolsChannel.invokeMethod<bool>('checkCollision', {
      'newBox': newBox,
      'existingBoxes': existingBoxes,
    });
    return result ?? false;
  }

  // ── Supabase persistence ──────────────────────────────────────────────────

  /// Saves a complete design session. Returns the created record id.
  Future<String> saveDesign({
    required String roomType,
    required String theme,
    required Uint8List sourceImageBytes,
    required Uint8List generatedImageBytes,
    Map<String, dynamic>? captureState,
    Map<String, dynamic>? roomPlan,
    String? depthMapDataUrl,              // data:image/png;base64,…
    List<SavedFurnitureSize> furnitureSizes = const [],
  }) async {
    final ts = DateTime.now().millisecondsSinceEpoch;

    // Upload images to Supabase Storage
    final sourcePath    = 'designs/$ts/source.jpg';
    final generatedPath = 'designs/$ts/generated.png';

    await _db.storage.from('room-images')
        .uploadBinary(sourcePath, sourceImageBytes,
            fileOptions: const FileOptions(contentType: 'image/jpeg'));

    await _db.storage.from('room-images')
        .uploadBinary(generatedPath, generatedImageBytes,
            fileOptions: const FileOptions(contentType: 'image/png'));

    // Optionally upload the depth map PNG
    String? depthPath;
    if (depthMapDataUrl != null) {
      final pngBytes = base64Decode(depthMapDataUrl.split(',').last);
      depthPath = 'designs/$ts/depth.png';
      await _db.storage.from('room-images')
          .uploadBinary(depthPath, pngBytes,
              fileOptions: const FileOptions(contentType: 'image/png'));
    }

    final row = await _db
        .from('room_designs')
        .insert({
          'room_type':            roomType,
          'theme':                theme,
          'source_image_path':    sourcePath,
          'generated_image_path': generatedPath,
          'depth_map_path':       depthPath,
          'capture_state':        captureState,
          'room_plan':            roomPlan,
          'furniture_sizes':      furnitureSizes.map((s) => s.toJson()).toList(),
        })
        .select('id')
        .single();

    return row['id'] as String;
  }

  /// Fetches all designs for the current user, newest first.
  Future<List<Map<String, dynamic>>> fetchHistory() async {
    final rows = await _db
        .from('room_designs')
        .select()
        .order('created_at', ascending: false);
    return List<Map<String, dynamic>>.from(rows as List);
  }

  /// Returns a signed URL for a stored image (valid 1 hour).
  Future<String> getImageUrl(String storagePath) async {
    return _db.storage
        .from('room-images')
        .createSignedUrl(storagePath, 3600);
  }

  /// Re-runs measureFurniture using the snapshot already saved in Supabase.
  Future<FurnitureSizeResult?> remeasureFromHistory({
    required String designId,
    required double xMin,
    required double yMin,
    required double xMax,
    required double yMax,
  }) async {
    final row = await _db
        .from('room_designs')
        .select('capture_state')
        .eq('id', designId)
        .single();

    final captureState = row['capture_state'] as Map<String, dynamic>?;
    if (captureState == null) return null;

    return measureFurniture(
        xMin: xMin, yMin: yMin, xMax: xMax, yMax: yMax,
        snapshotMap: captureState);
  }
}

// ── Value types ───────────────────────────────────────────────────────────────

class FurnitureSizeResult {
  final double widthMeters;
  final double heightMeters;
  final double depthMeters;

  const FurnitureSizeResult({
    required this.widthMeters,
    required this.heightMeters,
    required this.depthMeters,
  });

  factory FurnitureSizeResult.fromMap(Map<String, dynamic> m) =>
      FurnitureSizeResult(
        widthMeters:  (m['widthMeters']  as num).toDouble(),
        heightMeters: (m['heightMeters'] as num).toDouble(),
        depthMeters:  (m['depthMeters']  as num).toDouble(),
      );

  @override
  String toString() =>
      'W: ${widthMeters.toStringAsFixed(2)}m  '
      'H: ${heightMeters.toStringAsFixed(2)}m  '
      'D: ${depthMeters.toStringAsFixed(2)}m';
}

class SavedFurnitureSize {
  final String label;
  final double widthMeters;
  final double heightMeters;
  final double depthMeters;
  final Map<String, double> bbox;   // { xMin, yMin, xMax, yMax } in pixels

  const SavedFurnitureSize({
    required this.label,
    required this.widthMeters,
    required this.heightMeters,
    required this.depthMeters,
    required this.bbox,
  });

  Map<String, dynamic> toJson() => {
    'label':        label,
    'widthMeters':  widthMeters,
    'heightMeters': heightMeters,
    'depthMeters':  depthMeters,
    'bbox':         bbox,
  };
}
