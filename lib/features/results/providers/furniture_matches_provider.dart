import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/services/product_match_service.dart';
import '../../../shared/models/furniture_item.dart';
import '../../../shared/models/product_match.dart';

/// Which matcher the app uses (override it in tests).
final productMatchServiceProvider = Provider<ProductMatchService>(
  (ref) => BackendProductMatchService(),
);

/// The products proposed for one detected piece of furniture.
///
/// There is no price filter in the UI for now. The matcher still accepts a
/// `PriceRange` (and the backend filters on it), so bringing it back is a
/// screen change plus a family key that carries the range.
final furnitureMatchesProvider = FutureProvider.autoDispose
    .family<List<ProductMatch>, FurnitureItem>((ref, item) {
  return ref.watch(productMatchServiceProvider).matchesFor(item);
});
