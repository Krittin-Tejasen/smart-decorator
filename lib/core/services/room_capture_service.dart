import 'dart:io';
import 'package:supabase_flutter/supabase_flutter.dart';
import '../models/ar_capture_state.dart';

/// Result returned after a successful save to Supabase.
class RoomCaptureUploadResult {
  final String id;
  final String imageUrl;
  const RoomCaptureUploadResult({required this.id, required this.imageUrl});
}

/// Saves AR photo captures (JPEG + spatial metadata) to Supabase Storage +
/// the `room_captures` table. Rolls back the uploaded image if the DB insert
/// fails, so no orphaned files are left in Storage.
class RoomCaptureService {
  static const _bucket = 'room-images';
  static const _table  = 'room_captures';

  SupabaseClient get _db => Supabase.instance.client;

  // ── Public API ───────────────────────────────────────────────────────────

  /// Full save: upload JPEG → get public URL → insert row.
  ///
  /// [capture]      — the ARCaptureState produced by the AR camera.
  /// [spatialData]  — optional extra: floor plan points, ceiling Y, mesh
  ///                  anchors, LiDAR snapshot, etc.  Pass whatever you have.
  ///
  /// Throws [RoomCaptureServiceException] on failure (with a human-readable
  /// message). The upload and insert are automatically rolled back.
  Future<RoomCaptureUploadResult> saveCapture({
    required ARCaptureState capture,
    Map<String, dynamic> spatialData = const {},
  }) async {
    final storagePath = _buildStoragePath(capture.capturedAt);

    // ── Step 1: Upload the JPEG ───────────────────────────────────────────
    final imageFile = File(capture.imagePath);
    if (!imageFile.existsSync()) {
      throw RoomCaptureServiceException(
          'Image file not found at ${capture.imagePath}');
    }

    final bytes = await imageFile.readAsBytes();

    try {
      await _db.storage.from(_bucket).uploadBinary(
        storagePath,
        bytes,
        fileOptions: const FileOptions(
          contentType: 'image/jpeg',
          upsert: false,
        ),
      );
    } catch (e) {
      throw RoomCaptureServiceException('Image upload failed: $e');
    }

    // ── Step 2: Public URL (no expiry since bucket is public) ─────────────
    final imageUrl = _db.storage.from(_bucket).getPublicUrl(storagePath);

    // ── Step 3: Insert database row ───────────────────────────────────────
    try {
      final row = await _db.from(_table).insert({
        'user_id': _db.auth.currentUser?.id,   // null if not signed in
        'image_url': imageUrl,
        'platform': capture.platform,

        // Store as { "matrix": [...16 doubles...] }
        // Column-major 4×4: extrinsics[12..14] = camera world position
        'camera_extrinsics': {
          'matrix': capture.extrinsics,
          'description': 'camera-to-world transform, column-major 4x4',
        },

        // Store as { "matrix": [...9 doubles...], "image_width": N, "image_height": N }
        // Row-major 3×3: [fx,0,cx, 0,fy,cy, 0,0,1]
        'camera_intrinsics': {
          'matrix': capture.intrinsics,
          'image_width': capture.imageWidth,
          'image_height': capture.imageHeight,
          'fx': capture.focalLength.$1,
          'fy': capture.focalLength.$2,
          'cx': capture.principalPoint.$1,
          'cy': capture.principalPoint.$2,
        },

        // Everything else: floor plan, ceiling height, LiDAR data, etc.
        'spatial_data': spatialData,
      }).select('id').single();

      return RoomCaptureUploadResult(
        id: row['id'] as String,
        imageUrl: imageUrl,
      );
    } catch (e) {
      // Rollback: remove the just-uploaded image so Storage stays clean
      await _db.storage
          .from(_bucket)
          .remove([storagePath])
          .catchError((_) => <FileObject>[]);

      throw RoomCaptureServiceException(
          'Database insert failed (image rolled back): $e');
    }
  }

  /// Save a LiDAR scan with no accompanying photo (identity matrix used as a
  /// placeholder for extrinsics/intrinsics since no camera frame was captured).
  Future<RoomCaptureUploadResult> saveScanOnly({
    required Map<String, dynamic> spatialData,
    String platform = 'ios',
  }) async {
    const identityMatrix = [
      1.0, 0.0, 0.0, 0.0,
      0.0, 1.0, 0.0, 0.0,
      0.0, 0.0, 1.0, 0.0,
      0.0, 0.0, 0.0, 1.0,
    ];

    try {
      final row = await _db.from(_table).insert({
        'user_id': _db.auth.currentUser?.id,
        'image_url': null,
        'platform': platform,
        'camera_extrinsics': {'matrix': identityMatrix, 'note': 'lidar_scan_no_photo'},
        'camera_intrinsics': {'matrix': List.filled(9, 0.0), 'note': 'lidar_scan_no_photo'},
        'spatial_data': spatialData,
      }).select('id').single();

      return RoomCaptureUploadResult(id: row['id'] as String, imageUrl: '');
    } catch (e) {
      throw RoomCaptureServiceException('Database insert failed: $e');
    }
  }

  /// Fetch all captures for the current user, newest first.
  Future<List<RoomCaptureSummary>> fetchHistory() async {
    final rows = await _db
        .from(_table)
        .select('id, image_url, platform, created_at, '
                'camera_intrinsics, spatial_data')
        .order('created_at', ascending: false);

    return (rows as List)
        .map((r) => RoomCaptureSummary.fromMap(Map<String, dynamic>.from(r)))
        .toList();
  }

  /// Delete a capture and its image from Storage.
  Future<void> deleteCapture(String id) async {
    // Fetch the row to get the storage path from the URL
    final row = await _db
        .from(_table)
        .select('image_url')
        .eq('id', id)
        .single();

    final imageUrl = row['image_url'] as String;
    final storagePath = _pathFromPublicUrl(imageUrl);

    // Delete the DB row first (RLS enforces ownership)
    await _db.from(_table).delete().eq('id', id);

    // Then remove the file (best-effort — DB row is already gone)
    if (storagePath != null) {
      await _db.storage
          .from(_bucket)
          .remove([storagePath])
          .catchError((_) => <FileObject>[]);
    }
  }

  // ── Helpers ─────────────────────────────────────────────────────────────

  String _buildStoragePath(String capturedAt) {
    final ms = DateTime.tryParse(capturedAt)?.millisecondsSinceEpoch
        ?? DateTime.now().millisecondsSinceEpoch;
    final uid = _db.auth.currentUser?.id ?? 'anonymous';
    return 'captures/$uid/$ms.jpg';
  }

  /// Extracts the object path from a Supabase public URL.
  /// URL format: .../storage/v1/object/public/room-images/{path}
  String? _pathFromPublicUrl(String url) {
    final marker = '/room-images/';
    final idx = url.indexOf(marker);
    if (idx == -1) return null;
    return url.substring(idx + marker.length);
  }
}

// ── Supporting types ───────────────────────────────────────────────────────

class RoomCaptureSummary {
  final String id;
  final String imageUrl;
  final String platform;
  final DateTime createdAt;
  final Map<String, dynamic> cameraIntrinsics;
  final Map<String, dynamic> spatialData;

  const RoomCaptureSummary({
    required this.id,
    required this.imageUrl,
    required this.platform,
    required this.createdAt,
    required this.cameraIntrinsics,
    required this.spatialData,
  });

  factory RoomCaptureSummary.fromMap(Map<String, dynamic> m) =>
      RoomCaptureSummary(
        id: m['id'] as String,
        imageUrl: m['image_url'] as String,
        platform: m['platform'] as String? ?? 'unknown',
        createdAt: DateTime.parse(m['created_at'] as String),
        cameraIntrinsics:
            Map<String, dynamic>.from(m['camera_intrinsics'] as Map? ?? {}),
        spatialData:
            Map<String, dynamic>.from(m['spatial_data'] as Map? ?? {}),
      );
}

/// Thrown by [RoomCaptureService] on any failure.
class RoomCaptureServiceException implements Exception {
  final String message;
  const RoomCaptureServiceException(this.message);
  @override
  String toString() => 'RoomCaptureServiceException: $message';
}
