import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../../shared/models/furniture_item.dart';

class DatasetService {
  static Future<Directory> _docsDir() async {
    return getApplicationDocumentsDirectory();
  }

  /// Saves one test session to `Documents/scan_<unix_ms>.json` —
  /// same directory and naming pattern as `ar_photo_<unix_ms>.jpg`.
  /// Returns the absolute file path so the UI can show it.
  static Future<String> saveSession({
    required String mode, // "lidar" | "estimated"
    required Map<String, dynamic> scanSummary,
    required List<FurnitureItem> furniture,
    Map<String, dynamic>? arCaptureJson,
  }) async {
    final now = DateTime.now();
    final ts = now.millisecondsSinceEpoch;

    final furnitureJson = furniture.map((item) {
      final d = item.dimensionsM;
      return {
        'label': item.label,
        'confidence': item.confidence,
        'computed_width_m': d?.widthM,
        'computed_depth_m': d?.depthM,
        'computed_height_m': d?.heightM,
        'ground_truth_width_m': null,
        'ground_truth_depth_m': null,
        'ground_truth_height_m': null,
      };
    }).toList();

    final data = {
      'timestamp': now.toIso8601String(),
      'mode': mode,
      'scan': scanSummary,
      if (arCaptureJson != null) 'ar_capture': arCaptureJson,
      'furniture': furnitureJson,
    };

    final dir = await _docsDir();
    final file = File('${dir.path}/scan_$ts.json');
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(data));
    return file.path;
  }

  /// Lists all saved scan sessions, newest first.
  static Future<List<File>> listSessions() async {
    final dir = await _docsDir();
    if (!await dir.exists()) return [];
    final files = await dir
        .list()
        .where((e) => e is File && (e.path.endsWith('.json')) &&
            e.path.split('/').last.startsWith('scan_'))
        .cast<File>()
        .toList();
    files.sort((a, b) => b.path.compareTo(a.path));
    return files;
  }
}
