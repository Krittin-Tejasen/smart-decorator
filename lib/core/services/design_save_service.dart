import 'supabase_service.dart';

/// The backend already persists every generation to Supabase (via
/// backend/db.py, called from /generate-room) — this service only flips
/// the `is_saved` flag the results screen's favorite button controls, and
/// reads back the designs that flag has been set on for the history screen.
class DesignSaveService {
  final _db = SupabaseService().supabase;

  Future<void> markDesignSaved(String designId) async {
    await _db
        .from('room_designs')
        .update({'is_saved': true})
        .eq('id', designId);
  }

  /// Saved designs (is_saved=true), newest first, with each design's
  /// furniture count via a PostgREST embedded resource count.
  Future<List<Map<String, dynamic>>> fetchSavedDesigns() async {
    final rows = await _db
        .from('room_designs')
        .select('id, room_type, style, color, generated_image_path, created_at, design_furniture(count)')
        .eq('is_saved', true)
        .order('created_at', ascending: false);
    return List<Map<String, dynamic>>.from(rows as List);
  }

  /// Signed URL (1 hour) for a path in the room-images bucket.
  Future<String> getImageUrl(String storagePath) async {
    return _db.storage.from('room-images').createSignedUrl(storagePath, 3600);
  }
}
