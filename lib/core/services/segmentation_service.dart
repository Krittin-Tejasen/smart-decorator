import 'dart:typed_data';

import 'package:dio/dio.dart';

import '../../shared/models/furniture_item.dart';
import 'api_service.dart';

/// Finds the furniture in an image already generated -- used when a saved
/// room is opened again, since only the images are stored, not the detection.
class SegmentationService {
  final Dio _dio;

  SegmentationService({Dio? dio}) : _dio = dio ?? ApiService().dio;

  Future<SegmentationResult> segment(
    Uint8List imageBytes, {
    String filename = 'room.png',
  }) async {
    // Dio picks the content type from the file extension.
    final form = FormData.fromMap({
      'image': MultipartFile.fromBytes(imageBytes, filename: filename),
    });

    try {
      final response = await _dio.post<Map<String, dynamic>>(
        '/segment-furniture',
        data: form,
      );
      final data = response.data;
      if (data == null) {
        throw Exception('Segmentation API returned an empty response.');
      }
      return SegmentationResult.fromJson(data);
    } on DioException catch (error) {
      final body = error.response?.data;
      final detail = body is Map<String, dynamic> ? body['detail'] : null;
      throw Exception(
        detail?.toString() ?? error.message ?? 'Furniture detection failed.',
      );
    }
  }
}
