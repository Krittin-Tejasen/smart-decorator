import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import '../../../core/constants/app_colors.dart';
import '../../../shared/models/product_match.dart';

/// Opens a product's shop page outside the app. Replaceable in tests.
typedef OpenProductLink = Future<bool> Function(Uri url);

Future<bool> _openInBrowser(Uri url) => launchUrl(url, mode: LaunchMode.externalApplication);

/// One proposed product for a detected piece of furniture: photo (or a
/// category icon when there is none), name, price, size, colour/material, how
/// closely it matches, and a link to buy it.
class ProductMatchCard extends StatelessWidget {
  final ProductMatch product;
  final OpenProductLink openLink;

  const ProductMatchCard({super.key, required this.product, this.openLink = _openInBrowser});

  Uri? get _shopUrl {
    final uri = Uri.tryParse(product.url ?? '');
    return uri != null && uri.hasScheme && uri.host.isNotEmpty ? uri : null;
  }

  Future<void> _open(BuildContext context, Uri url) async {
    final messenger = ScaffoldMessenger.maybeOf(context);
    final opened = await openLink(url).catchError((Object _) => false);
    if (!opened) {
      messenger?.showSnackBar(const SnackBar(content: Text('Could not open the store page')));
    }
  }

  IconData get _icon {
    switch (product.category) {
      case 'Sofa':
        return Icons.weekend_rounded;
      case 'Bed':
        return Icons.bed_rounded;
      case 'Chair':
        return Icons.chair_alt_rounded;
      case 'Table':
      case 'Bedside table':
        return Icons.table_restaurant_rounded;
      case 'Storage':
        return Icons.shelves;
      case 'Lighting':
        return Icons.emoji_objects_outlined;
      case 'Rug':
        return Icons.crop_square_rounded;
      case 'Curtain':
        return Icons.curtains_outlined;
      case 'Plant':
        return Icons.local_florist_rounded;
      case 'Wall decor':
        return Icons.image_outlined;
      default:
        return Icons.category_rounded;
    }
  }

  Widget _photo() {
    final url = product.imageUrl;
    final placeholder = Center(
      child: Icon(_icon, size: 30, color: AppColors.brassDeep),
    );

    return ClipRRect(
      borderRadius: BorderRadius.circular(12),
      child: ColoredBox(
        color: AppColors.sandTint,
        child: SizedBox(
          width: 88,
          height: 88,
          child: url == null || url.isEmpty
              ? placeholder
              : Image.network(
                  url,
                  fit: BoxFit.cover,
                  errorBuilder: (context, error, stackTrace) => placeholder,
                ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final shopUrl = _shopUrl;
    final dimensions = product.dimensionsLabel;
    final details = [
      if (product.colorName != null) product.colorName!,
      if (product.material != null) product.material!,
    ].join(' · ');

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: AppColors.surface,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: AppColors.ink.withValues(alpha: 0.06),
            blurRadius: 14,
            offset: const Offset(0, 6),
          ),
        ],
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _photo(),

          const SizedBox(width: 12),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  product.name,
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: AppColors.ink,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  formatBaht(product.price),
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                    color: AppColors.sageDeep,
                  ),
                ),

                if (dimensions != null) ...[
                  const SizedBox(height: 4),
                  Text(
                    dimensions,
                    style: const TextStyle(fontSize: 11.5, color: AppColors.muted),
                  ),
                ],

                if (details.isNotEmpty) ...[
                  const SizedBox(height: 2),
                  Text(
                    details,
                    maxLines: 1,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 11.5, color: AppColors.muted),
                  ),
                ],

                const SizedBox(height: 8),

                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: [
                    _Chip(
                      label: product.scoreLabel,
                      background: AppColors.brassTint,
                      foreground: AppColors.brassDeep,
                    ),
                    _Chip(
                      label: product.inStock ? 'In stock' : 'Out of stock',
                      background: AppColors.sageTint,
                      foreground: AppColors.sageDeep,
                    ),
                    if (product.isSample)
                      const _Chip(
                        label: 'Sample',
                        background: AppColors.sandTint,
                        foreground: AppColors.muted,
                      ),
                  ],
                ),

                if (shopUrl != null)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: TextButton.icon(
                      key: const ValueKey('product-store-link'),
                      onPressed: () => _open(context, shopUrl),
                      icon: const Icon(Icons.open_in_new_rounded, size: 16),
                      label: Text('View at ${_storeName(shopUrl)}'),
                      style: TextButton.styleFrom(
                        foregroundColor: AppColors.sageDeep,
                        padding: EdgeInsets.zero,
                        minimumSize: const Size(0, 32),
                        textStyle: const TextStyle(fontSize: 12.5, fontWeight: FontWeight.w600),
                      ),
                    ),
                  ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

/// "IKEA" for ikea.com links, otherwise the site's host.
String _storeName(Uri url) {
  final host = url.host.replaceFirst(RegExp(r'^www\.'), '');
  return host.startsWith('ikea.') ? 'IKEA' : host;
}

class _Chip extends StatelessWidget {
  final String label;
  final Color background;
  final Color foreground;

  const _Chip({
    required this.label,
    required this.background,
    required this.foreground,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 3),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(999),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 10.5,
          fontWeight: FontWeight.w600,
          color: foreground,
        ),
      ),
    );
  }
}
