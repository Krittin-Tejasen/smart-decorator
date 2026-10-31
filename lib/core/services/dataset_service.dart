import 'dart:convert';
import 'dart:io';

import 'package:path_provider/path_provider.dart';

import '../../shared/models/furniture_item.dart';

class DatasetService {
  static Future<Directory> _testDir() async {
    final docs = await getApplicationDocumentsDirectory();
    final dir = Directory('${docs.path}/smart_decorator_tests');
    await dir.create(recursive: true);
    return dir;
  }

  /// Saves one test session to `Documents/smart_decorator_tests/test_<ts>.json`.
  /// Returns the absolute file path so the UI can show it.
  static Future<String> saveSession({
    required String mode, // "lidar" | "estimated"
    required Map<String, dynamic> scanSummary,
    required List<FurnitureItem> furniture,
  }) async {
    final now = DateTime.now();
    final ts = '${now.year}'
        '${now.month.toString().padLeft(2, '0')}'
        '${now.day.toString().padLeft(2, '0')}'
        '_${now.hour.toString().padLeft(2, '0')}'
        '${now.minute.toString().padLeft(2, '0')}'
        '${now.second.toString().padLeft(2, '0')}';
    final id = 'test_$ts';

    final furnitureJson = furniture.map((item) {
      final d = item.dimensionsM;
      return {
        'label': item.label,
        'confidence': item.confidence,
        'computed_width_m': d?.widthM,
        'computed_depth_m': d?.depthM,
        'computed_height_m': d?.heightM,
        // Fill these in manually after measuring with a tape measure
        'ground_truth_width_m': null,
        'ground_truth_depth_m': null,
        'ground_truth_height_m': null,
      };
    }).toList();

    final data = {
      'id': id,
      'timestamp': now.toIso8601String(),
      'mode': mode,
      'scan': scanSummary,
      'furniture': furnitureJson,
    };

    final dir = await _testDir();
    final file = File('${dir.path}/$id.json');
    await file.writeAsString(const JsonEncoder.withIndent('  ').convert(data));
    return file.path;
  }

  /// Lists all saved test sessions, newest first.
  static Future<List<File>> listSessions() async {
    final dir = await _testDir();
    if (!await dir.exists()) return [];
    final files = await dir
        .list()
        .where((e) => e is File && e.path.endsWith('.json'))
        .cast<File>()
        .toList();
    files.sort((a, b) => b.path.compareTo(a.path));
    return files;
  }
}
