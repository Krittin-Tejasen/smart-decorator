import 'dart:async';

import 'package:smart_decorator/core/services/design_repository.dart';
import 'package:smart_decorator/shared/models/saved_design.dart';

/// An in-memory stand-in for the Supabase-backed repository, with the same
/// rules that matter: a per-user limit and newest-first listing.
class FakeDesignRepository implements DesignRepository {
  FakeDesignRepository({
    List<SavedDesign> designs = const [],
    this.limit = 5,
  }) : designs = [...designs];

  final List<SavedDesign> designs;
  final int limit;

  /// Every [DesignToSave] that reached [save], in order.
  final List<DesignToSave> saveRequests = [];
  final List<SavedDesign> deleted = [];

  /// When set, [save] waits for it (to observe the "saving" state).
  Completer<void>? saveGate;

  /// When set, the next [save] / [delete] / [list] throws it.
  Object? saveError;
  Object? deleteError;
  Object? listError;

  int listCalls = 0;

  @override
  int get maxSavedDesigns => limit;

  @override
  Future<SavedDesign> save(DesignToSave design) async {
    saveRequests.add(design);
    if (saveGate != null) await saveGate!.future;
    if (saveError != null) throw saveError!;
    if (designs.length >= limit) throw SaveLimitReachedException(limit);

    final saved = SavedDesign(
      id: 'design-${designs.length + 1}',
      roomType: design.roomType,
      style: design.style,
      color: design.color,
      createdAt: DateTime.utc(2026, 9, 27, 14, designs.length),
      generatedImagePath: 'user/${designs.length + 1}/generated.png',
      sourceImagePath: 'user/${designs.length + 1}/source.${design.sourceExtension}',
    );
    designs.insert(0, saved);
    return saved;
  }

  @override
  Future<List<SavedDesign>> list() async {
    listCalls++;
    if (listError != null) throw listError!;
    return [...designs];
  }

  @override
  Future<void> delete(SavedDesign design) async {
    if (deleteError != null) throw deleteError!;
    designs.removeWhere((d) => d.id == design.id);
    deleted.add(design);
  }

  @override
  Future<String> signedImageUrl(String storagePath) async =>
      'https://storage.invalid/$storagePath?token=t';
}

SavedDesign savedDesign(
  String id, {
  String roomType = 'living_room',
  String style = 'japandi',
  DateTime? createdAt,
  String? sourceImagePath = 'user/source.jpg',
}) {
  return SavedDesign(
    id: id,
    roomType: roomType,
    style: style,
    color: 'warm_oat_cream',
    createdAt: createdAt ?? DateTime.utc(2026, 9, 27, 14, 5),
    generatedImagePath: 'user/$id/generated.png',
    sourceImagePath: sourceImagePath,
  );
}
