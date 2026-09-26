import 'dart:async';

import 'package:dio/dio.dart';

import '../../shared/models/generate_room_request.dart';
import '../../shared/models/generate_room_response.dart';
import '../../shared/models/generation_progress.dart';
import 'api_service.dart';

/// Lets the UI stop a generation that is being polled. Cancelling also tells
/// the backend to stop, so an abandoned request doesn't keep paying for
/// retries nobody is waiting for.
class GenerationCancelToken {
  bool _cancelled = false;
  bool get isCancelled => _cancelled;
  void cancel() => _cancelled = true;
}

class GenerationCancelled implements Exception {
  @override
  String toString() => 'Generation cancelled';
}

/// A snapshot of a backend generation job (GET /generate-room/jobs/{id}).
class GenerationJobStatus {
  final String status; // running | done | error | cancelled
  final String stage;
  final String message;
  final double progress;
  final GenerateRoomResponse? result;
  final String? error;

  const GenerationJobStatus({
    required this.status,
    required this.stage,
    required this.message,
    required this.progress,
    this.result,
    this.error,
  });

  factory GenerationJobStatus.fromJson(Map<String, dynamic> json) {
    final result = json['result'];
    return GenerationJobStatus(
      status: json['status'] as String? ?? 'running',
      stage: json['stage'] as String? ?? '',
      message: json['message'] as String? ?? '',
      progress: (json['progress'] as num?)?.toDouble() ?? 0,
      result: result is Map<String, dynamic>
          ? GenerateRoomResponse.fromJson(result)
          : null,
      error: json['error'] as String?,
    );
  }
}

class AIGenerationService {
  final Dio _dio;

  AIGenerationService({
    Dio? dio,
  }) : _dio = dio ?? ApiService().dio;

  Future<FormData> _buildForm(GenerateRoomRequest request) async {
    return FormData.fromMap({
      'room_type': request.roomType,
      'style': request.style,
      'color': request.color,
      // Omitted entirely (not sent as a literal "null" string) when the
      // user hasn't picked a specific model - the backend then falls back
      // to its own AI_IMAGE_PROVIDER env setting.
      if (request.provider != null) 'provider': request.provider,
      'image': await MultipartFile.fromFile(
        request.imagePath,
        filename: request.imagePath.split(RegExp(r'[\\/]')).last,
      ),
    });
  }

  /// The backend's own message for a failed request, if it sent one.
  static String _errorDetail(DioException error, String fallback) {
    final data = error.response?.data;
    final detail = data is Map<String, dynamic> ? data['detail'] : null;
    return detail?.toString() ?? error.message ?? fallback;
  }

  Future<GenerateRoomResponse> generateRoom(
    GenerateRoomRequest request,
  ) async {
    final formData = await _buildForm(request);

    final Response<Map<String, dynamic>> response;

    try {
      response = await _dio.post<Map<String, dynamic>>(
        '/generate-room',
        data: formData,
      );
    } on DioException catch (error) {
      throw Exception(_errorDetail(error, 'Generate room request failed.'));
    }

    final data = response.data;
    if (data == null) {
      throw Exception('Generate room API returned an empty response.');
    }

    return GenerateRoomResponse.fromJson(data);
  }

  // ── Job API: start, poll for real progress, cancel ────────────────────────

  /// Starts a generation on the backend and returns its job id right away.
  Future<String> startGenerateRoomJob(GenerateRoomRequest request) async {
    final formData = await _buildForm(request);

    try {
      final response = await _dio.post<Map<String, dynamic>>(
        '/generate-room/jobs',
        data: formData,
      );
      final jobId = response.data?['job_id'];
      if (jobId is! String) {
        throw Exception('Generate room API did not return a job id.');
      }
      return jobId;
    } on DioException catch (error) {
      throw Exception(_errorDetail(error, 'Generate room request failed.'));
    }
  }

  Future<GenerationJobStatus> getGenerateRoomJob(String jobId) async {
    final response = await _dio.get<Map<String, dynamic>>(
      '/generate-room/jobs/$jobId',
    );
    final data = response.data;
    if (data == null) {
      throw Exception('Generate room API returned an empty response.');
    }
    return GenerationJobStatus.fromJson(data);
  }

  Future<void> cancelGenerateRoomJob(String jobId) async {
    await _dio.delete<Map<String, dynamic>>('/generate-room/jobs/$jobId');
  }

  /// Runs a generation as a backend job and reports what the backend is
  /// really doing through [onProgress] (called on every poll), then returns
  /// the final result.
  ///
  /// Throws [GenerationCancelled] if [cancelToken] is cancelled (the backend
  /// job is cancelled too), and an [Exception] carrying the backend's message
  /// if the job fails, is lost (server restart), or takes longer than
  /// [timeout]. A few dropped polls in a row (flaky Wi-Fi) are tolerated.
  Future<GenerateRoomResponse> generateRoomWithProgress(
    GenerateRoomRequest request, {
    void Function(GenerationProgress progress)? onProgress,
    GenerationCancelToken? cancelToken,
    Duration pollInterval = const Duration(seconds: 1),
    Duration timeout = const Duration(minutes: 6),
    int maxConsecutivePollFailures = 5,
  }) async {
    final jobId = await startGenerateRoomJob(request);
    final deadline = DateTime.now().add(timeout);
    var consecutiveFailures = 0;

    while (true) {
      if (cancelToken?.isCancelled ?? false) {
        await _cancelQuietly(jobId);
        throw GenerationCancelled();
      }

      if (DateTime.now().isAfter(deadline)) {
        await _cancelQuietly(jobId);
        throw Exception('Generation took too long. Please try again.');
      }

      GenerationJobStatus status;
      try {
        status = await getGenerateRoomJob(jobId);
        consecutiveFailures = 0;
      } on DioException catch (error) {
        if (error.response?.statusCode == 404) {
          throw Exception(
            'The server lost this generation (it may have restarted). '
            'Please try again.',
          );
        }
        consecutiveFailures++;
        if (consecutiveFailures >= maxConsecutivePollFailures) {
          throw Exception(_errorDetail(error, 'Lost connection to the server.'));
        }
        await Future<void>.delayed(pollInterval);
        continue;
      }

      switch (status.status) {
        case 'done':
          final result = status.result;
          if (result == null) {
            throw Exception('Generate room API returned no result.');
          }
          return result;
        case 'error':
          throw Exception(status.error ?? 'Generate room request failed.');
        case 'cancelled':
          throw GenerationCancelled();
        default:
          onProgress?.call(
            GenerationProgress(
              stage: status.stage,
              message: status.message,
              progress: status.progress,
            ),
          );
      }

      await Future<void>.delayed(pollInterval);
    }
  }

  Future<void> _cancelQuietly(String jobId) async {
    try {
      await cancelGenerateRoomJob(jobId);
    } catch (_) {
      // Best effort: if the server is unreachable the job can't be stopped
      // from here anyway, and the caller is already leaving.
    }
  }
}
