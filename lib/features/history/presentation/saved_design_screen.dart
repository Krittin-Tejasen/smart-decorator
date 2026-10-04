import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../shared/models/saved_design.dart';
import '../../../shared/widgets/app_footer_nav.dart';
import '../../results/widgets/room_results_view.dart';
import '../providers/history_providers.dart';

/// A saved room opened from History. It looks and behaves like the Results
/// screen: the generated room (with Before/After from the saved photo) and its
/// furniture cards -- which are worked out again now, since only the images
/// were stored, so the cards show a loading state first.
class SavedDesignScreen extends ConsumerWidget {
  final SavedDesign design;

  const SavedDesignScreen({super.key, required this.design});

  ImageProvider? _imageFor(AsyncValue<String> url) =>
      url.maybeWhen(data: NetworkImage.new, orElse: () => null);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final afterUrl = ref.watch(savedImageUrlProvider(design.generatedImagePath));
    final sourcePath = design.sourceImagePath;
    final beforeUrl = sourcePath == null ? null : ref.watch(savedImageUrlProvider(sourcePath));

    return RoomResultsView(
      afterImage: _imageFor(afterUrl),
      afterLoading: afterUrl.isLoading,
      beforeImage: beforeUrl == null ? null : _imageFor(beforeUrl),
      furniture: ref.watch(savedDesignFurnitureProvider(design)),
      onRetryFurniture: () => ref.invalidate(savedDesignFurnitureProvider(design)),
      footerTab: FooterTab.history,
    );
  }
}
