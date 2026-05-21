class GenerateRoomRequest {

  final String roomType;
  final String theme;
  final String imagePath;

  GenerateRoomRequest({
    required this.roomType,
    required this.theme,
    required this.imagePath,
  });

  Map<String, dynamic> toJson() {
    return {
      'room_type': roomType,
      'theme': theme,
      'image_path': imagePath,
    };
  }
}