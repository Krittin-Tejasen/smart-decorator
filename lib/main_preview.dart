// Developer preview: opens the product-matches page for the saved room crops in
// backend/crops/ without uploading a photo, generating a room, or signing in.
// Runs in Chrome, so it needs no Visual Studio / Android toolchain.
//
//   cd backend && python -m uvicorn main:app                    (matching needs the backend)
//   cd backend/crops && python -m http.server 8001               (serves the crop images)
//   flutter run -d chrome -t lib/main_preview.dart \
//     --dart-define=API_BASE_URL=http://127.0.0.1:8000 \
//     --web-browser-flag="--disable-web-security"                (backend has no CORS; dev only)
//
// The real app starts from lib/main.dart; this file is not part of it.

import 'dart:convert';
import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter/material.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'core/theme/app_theme.dart';
import 'features/results/presentation/furniture_matches_screen.dart';
import 'shared/models/furniture_item.dart';

const _cropsUrl = 'http://127.0.0.1:8001';

void main() {
  dotenv.testLoad(fileInput: '');
  runApp(const ProviderScope(child: _PreviewApp()));
}

class _PreviewApp extends StatelessWidget {
  const _PreviewApp();

  @override
  Widget build(BuildContext context) => MaterialApp(
        debugShowCheckedModeBanner: false,
        title: 'Product matches preview',
        theme: AppTheme.lightTheme,
        home: const _CropPicker(),
      );
}

/// "room2_17_chandelier.png" -> "chandelier".
String _labelOf(String fileName) {
  final match = RegExp(r'^[^_]+_\d+_(.+)\.png$').firstMatch(fileName);
  return (match?.group(1) ?? fileName).replaceAll('_', ' ');
}

FurnitureItem _itemFrom(String name, Uint8List bytes) => FurnitureItem(
      id: name,
      label: _labelOf(name),
      confidence: 1,
      bbox: BoundingBox(xMin: 0, yMin: 0, xMax: 1, yMax: 1),
      // These crops are plain rectangles from the room, so, like a real item
      // without a precise mask, the matcher gets the crop itself.
      cropImage: 'data:image/png;base64,${base64Encode(bytes)}',
      maskImage: '',
      maskPrecise: false,
      features: FurnitureFeatures(dominantColors: const [], areaPct: 0, aspectRatio: 1),
    );

Future<List<String>> _listCrops() async {
  final response = await Dio().get<String>('$_cropsUrl/');
  final names = RegExp(r'href="([^"]+\.png)"').allMatches(response.data ?? '').map((m) => Uri.decodeFull(m.group(1)!)).toList();
  names.sort();
  return names;
}

Future<Uint8List> _loadCrop(String name) async {
  final response = await Dio().get<List<int>>('$_cropsUrl/$name', options: Options(responseType: ResponseType.bytes));
  return Uint8List.fromList(response.data!);
}

class _CropPicker extends StatefulWidget {
  const _CropPicker();

  @override
  State<_CropPicker> createState() => _CropPickerState();
}

class _CropPickerState extends State<_CropPicker> {
  late final Future<List<String>> _crops = _listCrops();

  Future<void> _open(String name) async {
    final bytes = await _loadCrop(name);
    if (!mounted) return;
    await Navigator.of(context).push(
      MaterialPageRoute<void>(builder: (_) => FurnitureMatchesScreen(item: _itemFrom(name, bytes))),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Preview: pick a detected item')),
      body: FutureBuilder<List<String>>(
        future: _crops,
        builder: (context, snapshot) {
          if (snapshot.hasError) {
            return const Center(child: Text('Start the crop server: cd backend/crops && python -m http.server 8001'));
          }
          final names = snapshot.data;
          if (names == null) return const Center(child: CircularProgressIndicator());
          return ListView(
            children: [
              for (final name in names)
                ListTile(
                  leading: Image.network('$_cropsUrl/$name', width: 56, height: 56, fit: BoxFit.contain),
                  title: Text(_labelOf(name)),
                  subtitle: Text(name),
                  trailing: const Icon(Icons.chevron_right),
                  onTap: () => _open(name),
                ),
            ],
          );
        },
      ),
    );
  }
}
