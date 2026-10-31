import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/services/ai_generation_service.dart';
import '../../../core/services/dataset_service.dart';
import '../../../shared/models/furniture_item.dart';
import '../../../shared/models/generate_room_request.dart';
import '../../../shared/providers/app_state_provider.dart';

class TestScreen extends ConsumerStatefulWidget {
  const TestScreen({super.key});

  @override
  ConsumerState<TestScreen> createState() => _TestScreenState();
}

class _TestScreenState extends ConsumerState<TestScreen> {
  bool _simulateUpload = false; // false = LiDAR mode, true = monocular simulation
  bool _isRunning = false;
  String? _error;
  List<FurnitureItem> _results = [];
  String? _depthSource;
  String? _savedPath;

  Future<void> _runAnalysis() async {
    final appState = ref.read(appStateProvider);
    final image = appState.uploadedImage;
    if (image == null) {
      setState(() => _error = 'No image available. Take a photo on the home screen first.');
      return;
    }

    setState(() {
      _isRunning = true;
      _error = null;
      _results = [];
      _savedPath = null;
    });

    try {
      final ar = appState.arCaptureState;
      final rawFloorPlan = appState.scanSpatialData?['floorPlanPoints'] as List?;
      final hasSpatial = !_simulateUpload && ar != null && ar.hasDepth && rawFloorPlan != null;
      final hasFloorPlan = _simulateUpload && rawFloorPlan != null;

      final parsedFloorPlan = rawFloorPlan
          ?.map((p) => Map<String, dynamic>.from(p as Map))
          .toList();

      final request = GenerateRoomRequest(
        roomType: appState.selectedRoomType?.id ?? 'living_room',
        style: appState.selectedStyle?.id ?? 'modern',
        color: appState.selectedColorOption?.id ?? 'neutral',
        imagePath: image.path,
        extrinsics: hasSpatial ? ar.extrinsics : null,
        intrinsics: hasSpatial ? ar.intrinsics : null,
        depthMapPng: hasSpatial ? ar.depthMapPng : null,
        depthMinMeters: hasSpatial ? ar.depthMinMeters : null,
        depthMaxMeters: hasSpatial ? ar.depthMaxMeters : null,
        floorPlanPoints: (hasSpatial || hasFloorPlan) ? parsedFloorPlan : null,
        roomHeightMeters: hasFloorPlan
            ? (appState.scanSpatialData?['heightMeters'] as num?)?.toDouble()
            : null,
        directSegment: true,
      );

      final response = await AIGenerationService().generateRoom(request);

      setState(() {
        _results = response.furnitureSegments?.items ?? [];
        _depthSource = response.furnitureSegments?.depthSource;
        _isRunning = false;
      });
    } catch (e) {
      setState(() {
        _error = e.toString();
        _isRunning = false;
      });
    }
  }

  Future<void> _saveDataset() async {
    final appState = ref.read(appStateProvider);
    final scan = appState.scanSpatialData;

    final floorPlanPoints = scan?['floorPlanPoints'] as List? ?? [];
    final heightM = (scan?['heightMeters'] as num?)?.toDouble();
    final cornerCount = floorPlanPoints.length;

    double areaM2 = 0;
    if (floorPlanPoints.length >= 3) {
      // Shoelace formula on XZ floor plan points
      final pts = floorPlanPoints
          .map((p) => (p as Map<String, dynamic>))
          .toList();
      double area = 0;
      final n = pts.length;
      for (var i = 0; i < n; i++) {
        final j = (i + 1) % n;
        final xi = (pts[i]['x'] as num).toDouble();
        final zi = (pts[i]['z'] as num).toDouble();
        final xj = (pts[j]['x'] as num).toDouble();
        final zj = (pts[j]['z'] as num).toDouble();
        area += xi * zj - xj * zi;
      }
      areaM2 = (area.abs() / 2).toDouble();
      areaM2 = double.parse(areaM2.toStringAsFixed(2));
    }

    try {
      final path = await DatasetService.saveSession(
        mode: _simulateUpload ? 'estimated' : 'lidar',
        scanSummary: {
          'floor_plan_points': floorPlanPoints,
          'height_m': heightM,
          'area_m2': areaM2,
          'corner_count': cornerCount,
        },
        furniture: _results,
      );
      setState(() => _savedPath = path);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Saved: $path')),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Save failed: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final appState = ref.watch(appStateProvider);
    final image = appState.uploadedImage;
    final scan = appState.scanSpatialData;
    final floorPlanPoints = scan?['floorPlanPoints'] as List?;
    final heightM = (scan?['heightMeters'] as num?)?.toDouble();
    final hasScan = floorPlanPoints != null && heightM != null;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Size Extraction Test'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _SectionHeader('Step 1 — Source data'),
          const SizedBox(height: 8),
          _ScanSummaryCard(image: image, scan: scan, hasScan: hasScan),
          const SizedBox(height: 16),
          _SectionHeader('Step 2 — Mode'),
          const SizedBox(height: 8),
          _ModeToggle(
            simulateUpload: _simulateUpload,
            hasSpatial: appState.arCaptureState?.hasDepth == true,
            hasScan: hasScan,
            onChanged: (v) => setState(() {
              _simulateUpload = v;
              _results = [];
              _depthSource = null;
              _savedPath = null;
            }),
          ),
          const SizedBox(height: 24),
          ElevatedButton.icon(
            onPressed: (image == null || _isRunning) ? null : _runAnalysis,
            icon: _isRunning
                ? const SizedBox(
                    width: 18,
                    height: 18,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.play_arrow_rounded),
            label: Text(_isRunning ? 'Analysing…' : 'Run Analysis'),
            style: ElevatedButton.styleFrom(
              minimumSize: const Size.fromHeight(48),
            ),
          ),
          if (_error != null) ...[
            const SizedBox(height: 12),
            Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
          ],
          if (_results.isNotEmpty) ...[
            const SizedBox(height: 24),
            _SectionHeader(
              'Results (${_results.length} items'
              '${_depthSource != null ? ' · depth: $_depthSource' : ''})',
            ),
            const SizedBox(height: 8),
            ..._results.map((item) => _FurnitureResultTile(item: item)),
            const SizedBox(height: 16),
            OutlinedButton.icon(
              onPressed: _saveDataset,
              icon: const Icon(Icons.save_alt_rounded),
              label: const Text('Save Dataset'),
              style: OutlinedButton.styleFrom(
                minimumSize: const Size.fromHeight(44),
              ),
            ),
            if (_savedPath != null) ...[
              const SizedBox(height: 8),
              Text(
                'Saved to Files app:\nOn My iPhone › Smart Decorator',
                style: Theme.of(context).textTheme.bodySmall,
                textAlign: TextAlign.center,
              ),
            ],
          ],
        ],
      ),
    );
  }
}

class _SectionHeader extends StatelessWidget {
  final String title;
  const _SectionHeader(this.title);

  @override
  Widget build(BuildContext context) => Text(
        title,
        style: Theme.of(context)
            .textTheme
            .labelLarge
            ?.copyWith(color: Theme.of(context).colorScheme.primary),
      );
}

class _ScanSummaryCard extends StatelessWidget {
  final File? image;
  final Map<String, dynamic>? scan;
  final bool hasScan;

  const _ScanSummaryCard({
    required this.image,
    required this.scan,
    required this.hasScan,
  });

  @override
  Widget build(BuildContext context) {
    final floorPlanPoints = scan?['floorPlanPoints'] as List?;
    final heightM = (scan?['heightMeters'] as num?)?.toDouble();
    final cornerCount = floorPlanPoints?.length ?? 0;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (image != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(8),
                child: Image.file(image!, width: 80, height: 80, fit: BoxFit.cover),
              )
            else
              Container(
                width: 80,
                height: 80,
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.surfaceVariant,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.image_not_supported_outlined),
              ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    image != null ? 'Photo ready' : 'No photo — take one on home screen',
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  const SizedBox(height: 4),
                  if (hasScan) ...[
                    Text(
                      'Height: ${heightM?.toStringAsFixed(2)} m',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                    Text(
                      'Corners: $cornerCount',
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ] else
                    Text(
                      'No scan data — scan the room first',
                      style: Theme.of(context).textTheme.bodySmall?.copyWith(
                            color: Theme.of(context).colorScheme.error,
                          ),
                    ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _ModeToggle extends StatelessWidget {
  final bool simulateUpload;
  final bool hasSpatial;
  final bool hasScan;
  final ValueChanged<bool> onChanged;

  const _ModeToggle({
    required this.simulateUpload,
    required this.hasSpatial,
    required this.hasScan,
    required this.onChanged,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        RadioListTile<bool>(
          value: false,
          groupValue: simulateUpload,
          onChanged: hasSpatial ? (_) => onChanged(false) : null,
          title: const Text('LiDAR mode'),
          subtitle: Text(
            hasSpatial
                ? 'Uses real depth map + camera pose from the scan'
                : 'Not available — requires a photo captured during scan',
          ),
          contentPadding: EdgeInsets.zero,
        ),
        RadioListTile<bool>(
          value: true,
          groupValue: simulateUpload,
          onChanged: hasScan ? (_) => onChanged(true) : null,
          title: const Text('Upload simulation'),
          subtitle: Text(
            hasScan
                ? 'Strips extrinsics/depth map, uses monocular depth + floor plan'
                : 'Not available — requires scan data (floor plan + height)',
          ),
          contentPadding: EdgeInsets.zero,
        ),
      ],
    );
  }
}

class _FurnitureResultTile extends StatelessWidget {
  final FurnitureItem item;

  const _FurnitureResultTile({required this.item});

  @override
  Widget build(BuildContext context) {
    final d = item.dimensionsM;
    final hasSize = d != null;

    return Card(
      margin: const EdgeInsets.only(bottom: 8),
      child: ListTile(
        leading: item.cropImage.isNotEmpty
            ? ClipRRect(
                borderRadius: BorderRadius.circular(6),
                child: Image.memory(
                  base64Decode(item.cropImage.contains(',')
                      ? item.cropImage.split(',').last
                      : item.cropImage),
                  width: 56,
                  height: 56,
                  fit: BoxFit.cover,
                  errorBuilder: (_, __, ___) => const Icon(Icons.chair),
                ),
              )
            : const Icon(Icons.chair),
        title: Text(
          '${item.label}  (${(item.confidence * 100).toStringAsFixed(0)}%)',
        ),
        subtitle: hasSize
            ? Text(
                'W ${d.widthM.toStringAsFixed(2)} m  '
                'D ${d.depthM.toStringAsFixed(2)} m  '
                'H ${d.heightM.toStringAsFixed(2)} m',
              )
            : const Text('Size not computed'),
        trailing: hasSize
            ? const Icon(Icons.straighten_rounded, size: 18)
            : const Icon(Icons.help_outline_rounded, size: 18),
      ),
    );
  }
}
