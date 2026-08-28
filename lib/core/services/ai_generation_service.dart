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
      'provider': request.provider,
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
