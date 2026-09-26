import 'dart:io' show Platform;

import 'package:dio/dio.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

class ApiService {
  /// Resolution order:
  /// 1. `--dart-define=API_BASE_URL=...` (CI / one-off overrides)
  /// 2. `API_BASE_URL` in assets/.env (gitignored — each machine sets its
  ///    own, e.g. a LAN IP for testing on a physical device)
  /// 3. The loopback alias for whichever emulator/simulator is running on
  ///    the same machine as the backend — works with zero config, but not
  ///    for a physical device or a backend on a different machine.
  static String get baseUrl {
    const defineOverride = String.fromEnvironment('API_BASE_URL');
    if (defineOverride.isNotEmpty) return defineOverride;

    final envOverride = dotenv.env['API_BASE_URL'];
    if (envOverride != null && envOverride.isNotEmpty) return envOverride;

    if (Platform.isAndroid) return 'http://10.0.2.2:8000';
    return 'http://127.0.0.1:8000';
  }

  final Dio dio = Dio(
    BaseOptions(
      baseUrl: baseUrl,
      connectTimeout: const Duration(seconds: 20),
      receiveTimeout: const Duration(minutes: 3),
    ),
  );
}
