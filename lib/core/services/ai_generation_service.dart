import 'package:dio/dio.dart';

import '../../shared/models/generate_room_request.dart';
import '../../shared/models/generate_room_response.dart';
import 'api_service.dart';

class AIGenerationService {
  final Dio _dio;

  AIGenerationService({
    Dio? dio,
  }) : _dio = dio ?? ApiService().dio;

  Future<GenerateRoomResponse> generateRoom(
    GenerateRoomRequest request,
  ) async {
    final formData = FormData.fromMap({
      'room_type': request.roomType,
      'style': request.style,
      'color': request.color,
      // Omitted entirely (not sent as a literal "null" string) when the
      // user hasn't picked a specific model - the backend then falls back
      // to its own AI_IMAGE_PROVIDER env setting.
      if (request.provider != null) 'provider': request.provider,
      'segment': 'true',
      'image': await MultipartFile.fromFile(
        request.imagePath,
        filename: request.imagePath.split(RegExp(r'[\\/]')).last,
      ),
    });

    final Response<Map<String, dynamic>> response;

    try {
      response = await _dio.post<Map<String, dynamic>>(
        '/generate-room',
        data: formData,
      );
    } on DioException catch (error) {
      final data = error.response?.data;
      final detail = data is Map<String, dynamic> ? data['detail'] : null;

      throw Exception(
        detail?.toString() ?? error.message ?? 'Generate room request failed.',
      );
    }

    final data = response.data;
    if (data == null) {
      throw Exception('Generate room API returned an empty response.');
    }

    return GenerateRoomResponse.fromJson(data);
  }
}
