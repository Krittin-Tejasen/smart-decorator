import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/services/product_match_service.dart';
import '../../../shared/models/furniture_item.dart';
import '../../../shared/models/product_match.dart';

/// Which matcher the app uses. Sample data until real product matching is
/// merged — then swap this one line (or override it in tests).
final productMatchServiceProvider = Provider<ProductMatchService>(
  (ref) => const SampleProductMatchService(),
);

/// The products proposed for one detected piece of furniture.
final furnitureMatchesProvider = FutureProvider.autoDispose
    .family<List<ProductMatch>, FurnitureItem>((ref, item) {
  return ref.watch(productMatchServiceProvider).matchesFor(item);
});
