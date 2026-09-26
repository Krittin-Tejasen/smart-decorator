import 'dart:typed_data';

import 'package:dio/dio.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/services/design_repository.dart';
import '../../../core/services/segmentation_service.dart';
import '../../../shared/models/furniture_item.dart';
import '../../../shared/models/saved_design.dart';

/// The user's saved rooms, newest first.
final savedDesignsProvider = FutureProvider.autoDispose<List<SavedDesign>>(
  (ref) => ref.watch(designRepositoryProvider).list(),
);

/// A temporary URL for one file in the private bucket (thumbnails, the room
/// photo). Cached per path while something is watching it.
final savedImageUrlProvider =
    FutureProvider.autoDispose.family<String, String>(
  (ref, storagePath) =>
      ref.watch(designRepositoryProvider).signedImageUrl(storagePath),
);

final segmentationServiceProvider = Provider<SegmentationService>(
  (ref) => SegmentationService(),
);

/// Downloads the bytes behind a URL. A provider so tests can replace it.
typedef ImageBytesLoader = Future<Uint8List> Function(String url);

final imageBytesLoaderProvider = Provider<ImageBytesLoader>((ref) {
  return (url) async {
    final response = await Dio().get<List<int>>(
      url,
      options: Options(responseType: ResponseType.bytes),
    );
    return Uint8List.fromList(response.data ?? const []);
  };
});

/// The furniture in a saved room. Only the images are stored, so it is found
/// again when the room is opened: fetch the generated image, run it through
/// the backend's segmentation. Kept for the session so reopening the same room
/// doesn't run the (slow) detection again; deleting the room drops it.
final savedDesignFurnitureProvider = FutureProvider.autoDispose
    .family<List<FurnitureItem>, SavedDesign>((ref, design) async {
  ref.keepAlive();

  final url = await ref
      .watch(designRepositoryProvider)
      .signedImageUrl(design.generatedImagePath);
  final bytes = await ref.watch(imageBytesLoaderProvider)(url);
  if (bytes.isEmpty) {
    throw Exception('Could not load the saved image.');
  }

  final result = await ref.watch(segmentationServiceProvider).segment(bytes);
  return result.items;
});
