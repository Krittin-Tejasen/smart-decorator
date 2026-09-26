import 'dart:math' as math;

import 'package:dio/dio.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:smart_decorator/core/services/ai_generation_service.dart';
import 'package:smart_decorator/shared/models/generate_room_request.dart';
import 'package:smart_decorator/shared/models/generation_progress.dart';

/// Plays back a scripted sequence of job polls: each item is either a
/// [GenerationJobStatus] (returned) or an [Exception] (thrown). The last item
/// repeats forever.
class ScriptedService extends AIGenerationService {
  ScriptedService(this.script) : super(dio: Dio());

  final List<Object> script;
  int polls = 0;
  final List<String> cancelledJobs = [];

  @override
  Future<String> startGenerateRoomJob(GenerateRoomRequest request) async => 'job-1';

  @override
  Future<GenerationJobStatus> getGenerateRoomJob(String jobId) async {
    final next = script[math.min(polls++, script.length - 1)];
    if (next is Exception) throw next;
    return next as GenerationJobStatus;
  }

  @override
  Future<void> cancelGenerateRoomJob(String jobId) async => cancelledJobs.add(jobId);
}

GenerationJobStatus running(String stage, String message, double progress) =>
    GenerationJobStatus(status: 'running', stage: stage, message: message, progress: progress);

GenerationJobStatus done() => GenerationJobStatus.fromJson({
      'status': 'done',
      'stage': 'done',
      'message': 'Your room design is ready',
      'progress': 1.0,
      'result': {
        'generated_image': 'data:image/png;base64,AAAA',
        'products': [
          {'id': '1', 'name': 'Modern Sofa', 'imageUrl': 'sofa', 'price': 12990},
        ],
      },
    });

DioException http(int? status, {DioExceptionType type = DioExceptionType.badResponse}) {
  final options = RequestOptions(path: '/generate-room/jobs/job-1');
  return DioException(
    requestOptions: options,
    type: type,
    response: status == null ? null : Response(requestOptions: options, statusCode: status),
  );
}

final request = GenerateRoomRequest(
  roomType: 'bedroom',
  style: 'japandi',
  color: 'warm_oat_cream',
  imagePath: 'room.jpg',
);

const fast = Duration(milliseconds: 1);

void main() {
  test('reports what the backend says on every poll, then returns the result', () async {
    final service = ScriptedService([
      running('compose', 'Analyzing your room and planning the design', 0.05),
      running('generate', 'Generating your new room design', 0.15),
      running('critic', 'Checking doors, windows and outlets are still in place', 0.65),
      done(),
    ]);
    final updates = <GenerationProgress>[];

    final result = await service.generateRoomWithProgress(
      request,
      onProgress: updates.add,
      pollInterval: fast,
    );

    expect(updates.map((u) => u.stage), ['compose', 'generate', 'critic']);
    expect(updates.map((u) => u.message).first, 'Analyzing your room and planning the design');
    expect(updates.map((u) => u.progress), [0.05, 0.15, 0.65]);
    expect(result.generatedImage, startsWith('data:image/png'));
    expect(result.products.single.name, 'Modern Sofa');
  });

  test('a failed job surfaces the backend message', () async {
    final service = ScriptedService([
      running('generate', 'Generating your new room design', 0.15),
      GenerationJobStatus.fromJson({
        'status': 'error',
        'stage': 'generate',
        'message': 'x',
        'progress': 0.2,
        'error': 'Gemini did not return an image.',
      }),
    ]);

    await expectLater(
      service.generateRoomWithProgress(request, pollInterval: fast),
      throwsA(predicate((e) => e.toString().contains('Gemini did not return an image.'))),
    );
  });

  test('a job cancelled on the server ends as GenerationCancelled', () async {
    final service = ScriptedService([
      GenerationJobStatus.fromJson({'status': 'cancelled', 'stage': 'generate', 'message': '', 'progress': 0.2}),
    ]);

    await expectLater(
      service.generateRoomWithProgress(request, pollInterval: fast),
      throwsA(isA<GenerationCancelled>()),
    );
  });

  test('cancelling the token stops polling and cancels the backend job', () async {
    final service = ScriptedService([running('generate', 'Generating your new room design', 0.15)]);
    final token = GenerationCancelToken();
    final updates = <GenerationProgress>[];

    final run = service.generateRoomWithProgress(
      request,
      cancelToken: token,
      onProgress: (u) {
        updates.add(u);
        if (updates.length == 2) token.cancel();
      },
      pollInterval: fast,
    );

    await expectLater(run, throwsA(isA<GenerationCancelled>()));
    expect(service.cancelledJobs, ['job-1'], reason: 'the server must be told to stop, not just the app');
    final pollsAtCancel = service.polls;
    await Future<void>.delayed(const Duration(milliseconds: 20));
    expect(service.polls, pollsAtCancel, reason: 'no more polling after cancel');
  });

  test('a couple of dropped polls (flaky Wi-Fi) are tolerated', () async {
    final service = ScriptedService([
      running('generate', 'Generating your new room design', 0.15),
      http(null, type: DioExceptionType.connectionError),
      http(503),
      running('critic', 'Checking doors, windows and outlets are still in place', 0.65),
      done(),
    ]);
    final updates = <GenerationProgress>[];

    final result = await service.generateRoomWithProgress(
      request,
      onProgress: updates.add,
      pollInterval: fast,
    );

    expect(updates.map((u) => u.stage), ['generate', 'critic']);
    expect(result.generatedImage, isNotEmpty);
  });

  test('too many dropped polls in a row gives up with an error', () async {
    final service = ScriptedService([http(null, type: DioExceptionType.connectionError)]);

    await expectLater(
      service.generateRoomWithProgress(request, pollInterval: fast, maxConsecutivePollFailures: 3),
      throwsA(isA<Exception>()),
    );
    expect(service.polls, 3);
  });

  test('a success in between resets the failure count', () async {
    final service = ScriptedService([
      http(503),
      http(503),
      running('generate', 'Generating your new room design', 0.15),
      http(503),
      http(503),
      done(),
    ]);

    final result = await service.generateRoomWithProgress(
      request,
      pollInterval: fast,
      maxConsecutivePollFailures: 3,
    );

    expect(result.generatedImage, isNotEmpty);
  });

  test('a 404 means the server lost the job (restart) and fails immediately', () async {
    final service = ScriptedService([http(404)]);

    await expectLater(
      service.generateRoomWithProgress(request, pollInterval: fast),
      throwsA(predicate((e) => e.toString().contains('lost this generation'))),
    );
    expect(service.polls, 1, reason: 'retrying a 404 would just wait for nothing');
  });

  test('gives up after the timeout and cancels the backend job', () async {
    final service = ScriptedService([running('generate', 'Generating your new room design', 0.15)]);

    await expectLater(
      service.generateRoomWithProgress(
        request,
        pollInterval: fast,
        timeout: const Duration(milliseconds: 40),
      ),
      throwsA(predicate((e) => e.toString().contains('took too long'))),
    );
    expect(service.cancelledJobs, ['job-1']);
  });

  test('status JSON with missing fields still parses', () {
    final status = GenerationJobStatus.fromJson({});

    expect(status.status, 'running');
    expect(status.progress, 0);
    expect(status.result, isNull);
  });
}
