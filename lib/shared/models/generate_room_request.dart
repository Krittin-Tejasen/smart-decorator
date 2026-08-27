class GenerateRoomRequest {

  final String roomType;
  final String style;
  final String color;
  final String imagePath;

  GenerateRoomRequest({
    required this.roomType,
    required this.style,
    required this.color,
    required this.imagePath,
  });

  Map<String, dynamic> toJson() {
    return {
      'room_type': roomType,
      'style': style,
      'color': color,
      'image_path': imagePath,
    };
  }
}
