/// A row from Supabase's `room_designs` table where is_saved=true, with a
/// resolved signed URL for its generated image and its furniture count
/// from the linked `design_furniture` rows.
class SavedDesign {
  final String id;
  final String roomType;
  final String style;
  final String color;
  final DateTime createdAt;
  final int furnitureCount;
  final String imageUrl;

  SavedDesign({
    required this.id,
    required this.roomType,
    required this.style,
    required this.color,
    required this.createdAt,
    required this.furnitureCount,
    required this.imageUrl,
  });

  factory SavedDesign.fromRow(Map<String, dynamic> row, String imageUrl) {
    final furnitureRows = row['design_furniture'];
    int count = 0;
    if (furnitureRows is List && furnitureRows.isNotEmpty) {
      count = (furnitureRows.first['count'] as num?)?.toInt() ?? 0;
    }

    return SavedDesign(
      id: row['id'] as String,
      roomType: row['room_type'] as String? ?? '',
      style: row['style'] as String? ?? '',
      color: row['color'] as String? ?? '',
      createdAt: DateTime.parse(row['created_at'] as String),
      furnitureCount: count,
      imageUrl: imageUrl,
    );
  }
}
