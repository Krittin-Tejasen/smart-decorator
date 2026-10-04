import 'dart:typed_data';

import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';

import '../../shared/models/saved_design.dart';
import 'auth_service.dart';

/// Thrown when the user already has the maximum number of saved rooms.
class SaveLimitReachedException implements Exception {
  final int limit;
  const SaveLimitReachedException(this.limit);

  @override
  String toString() =>
      'You can save up to $limit rooms. Delete one in History to save this one.';
}

/// What gets saved when the user taps "Save Room Data": the generated design,
/// the photo it was made from, and how it was described. Nothing else --
/// furniture and products are worked out again when the room is opened.
class DesignToSave {
  final Uint8List generatedImage; // PNG
  final Uint8List sourceImage;
  final String sourceExtension; // jpg, png, ...
  final String roomType;
  final String style;
  final String color;

  const DesignToSave({
    required this.generatedImage,
    required this.sourceImage,
    required this.sourceExtension,
    required this.roomType,
    required this.style,
    required this.color,
  });
}

/// The user's saved rooms. The Supabase implementation is the only code that
/// talks to the `room_designs` table and the `room-images` bucket.
abstract class DesignRepository {
  /// How many rooms one user may keep (enforced by the database as well).
  int get maxSavedDesigns;

  /// Saves the room; throws [SaveLimitReachedException] at the limit and
  /// [NotSignedInException] when there is no user to save under.
  Future<SavedDesign> save(DesignToSave design);

  /// The user's saved rooms, newest first.
  Future<List<SavedDesign>> list();

  /// Deletes the room and its images.
  Future<void> delete(SavedDesign design);

  /// A temporary (1 hour) URL for a file in the private bucket.
  Future<String> signedImageUrl(String storagePath);
}

class SupabaseDesignRepository implements DesignRepository {
  SupabaseDesignRepository({required AuthService auth, SupabaseClient? client})
      : _auth = auth,
        _injected = client;

  /// Must match public.max_saved_designs() in the Supabase migration.
  static const int savedDesignLimit = 5;
  static const _bucket = 'room-images';
  static const _table = 'room_designs';
  static const _limitMessage = 'save_limit_reached';

  final AuthService _auth;
  final SupabaseClient? _injected;

  SupabaseClient get _db => _injected ?? Supabase.instance.client;

  @override
  int get maxSavedDesigns => savedDesignLimit;

  @override
  Future<SavedDesign> save(DesignToSave design) async {
    final userId = await _auth.ensureSignedIn();

    // Cheap check first so a user at the limit doesn't upload two images just
    // to have the database refuse the row. The database check is the real one.
    if ((await list()).length >= savedDesignLimit) {
      throw const SaveLimitReachedException(savedDesignLimit);
    }

    // Files live under the user's own folder; storage policies only allow that.
    final folder = '$userId/${DateTime.now().millisecondsSinceEpoch}';
    final generatedPath = '$folder/generated.png';
    final sourcePath = '$folder/source.${design.sourceExtension}';

    final storage = _db.storage.from(_bucket);
    await storage.uploadBinary(
      generatedPath,
      design.generatedImage,
      fileOptions: const FileOptions(contentType: 'image/png', upsert: false),
    );

    try {
      await storage.uploadBinary(
        sourcePath,
        design.sourceImage,
        fileOptions: FileOptions(
          contentType: _contentTypeFor(design.sourceExtension),
          upsert: false,
        ),
      );

      final row = await _db
          .from(_table)
          .insert({
            'room_type': design.roomType,
            'style': design.style,
            'color': design.color,
            'generated_image_path': generatedPath,
            'source_image_path': sourcePath,
            'is_saved': true,
          })
          .select()
          .single();

      return SavedDesign.fromRow(Map<String, dynamic>.from(row));
    } on PostgrestException catch (error) {
      await _removeQuietly([generatedPath, sourcePath]);
      if (error.message.contains(_limitMessage)) {
        throw const SaveLimitReachedException(savedDesignLimit);
      }
      rethrow;
    } catch (_) {
      // An upload failed part-way: don't leave the first image orphaned.
      await _removeQuietly([generatedPath, sourcePath]);
      rethrow;
    }
  }

  @override
  Future<List<SavedDesign>> list() async {
    await _auth.ensureSignedIn();

    final rows = await _db
        .from(_table)
        .select('id, room_type, style, color, created_at, generated_image_path, source_image_path')
        .order('created_at', ascending: false);

    return [
      for (final row in rows as List)
        SavedDesign.fromRow(Map<String, dynamic>.from(row as Map)),
    ];
  }

  @override
  Future<void> delete(SavedDesign design) async {
    await _auth.ensureSignedIn();

    // Row first: if that fails nothing has been lost. Files second, best
    // effort: an orphaned file is only wasted space, a row pointing at
    // deleted files would be a broken entry in the user's history.
    await _db.from(_table).delete().eq('id', design.id);

    await _removeQuietly([
      design.generatedImagePath,
      if (design.sourceImagePath != null) design.sourceImagePath!,
    ]);
  }

  @override
  Future<String> signedImageUrl(String storagePath) async {
    await _auth.ensureSignedIn();
    return _db.storage.from(_bucket).createSignedUrl(storagePath, 3600);
  }

  Future<void> _removeQuietly(List<String> paths) async {
    try {
      await _db.storage.from(_bucket).remove(paths);
    } catch (_) {
      // Best effort; see delete().
    }
  }

  static String _contentTypeFor(String extension) {
    switch (extension.toLowerCase()) {
      case 'png':
        return 'image/png';
      case 'webp':
        return 'image/webp';
      case 'heic':
        return 'image/heic';
      default:
        return 'image/jpeg';
    }
  }
}

final designRepositoryProvider = Provider<DesignRepository>(
  (ref) => SupabaseDesignRepository(auth: ref.watch(authServiceProvider)),
);
