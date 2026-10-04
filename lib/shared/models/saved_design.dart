/// A room the user saved: one row of Supabase's `room_designs` table.
///
/// Only the description and where the two images live; the images themselves
/// are fetched with signed URLs, and the furniture is detected again when the
/// room is opened (nothing about it is stored).
class SavedDesign {
  final String id;

  /// Ids, as chosen on the Home screen (e.g. `living_room`, `japandi`).
  final String roomType;
  final String style;
  final String color;
  final DateTime createdAt;

  /// Paths in the private `room-images` bucket, `<user id>/<timestamp>/…`.
  final String generatedImagePath;
  final String? sourceImagePath;

  const SavedDesign({
    required this.id,
    required this.roomType,
    required this.style,
    required this.color,
    required this.createdAt,
    required this.generatedImagePath,
    this.sourceImagePath,
  });

  factory SavedDesign.fromRow(Map<String, dynamic> row) {
    final source = row['source_image_path'] as String?;
    return SavedDesign(
      id: row['id'] as String,
      roomType: row['room_type'] as String? ?? '',
      style: row['style'] as String? ?? '',
      color: row['color'] as String? ?? '',
      createdAt: DateTime.parse(row['created_at'] as String),
      generatedImagePath: row['generated_image_path'] as String? ?? '',
      sourceImagePath: source == null || source.isEmpty ? null : source,
    );
  }

  // Identity is the row: lets a SavedDesign key a Riverpod family.
  @override
  bool operator ==(Object other) => other is SavedDesign && other.id == id;

  @override
  int get hashCode => id.hashCode;
}
