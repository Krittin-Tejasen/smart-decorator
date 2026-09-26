import 'product.dart';
import 'furniture_item.dart';

class GenerateRoomResponse {

  final String generatedImage;
  final List<Product> products;
  final SegmentationResult? furnitureSegments;

  GenerateRoomResponse({
    required this.generatedImage,
    required this.products,
    this.furnitureSegments,
  });

  factory GenerateRoomResponse.fromJson(Map<String, dynamic> json) {
    final productItems = json['products'];
    final segments = json['furniture_segments'];

    return GenerateRoomResponse(
      generatedImage: json['generated_image'] as String? ?? '',
      products: productItems is List
          ? productItems
              .map((item) => Product.fromJson(item as Map<String, dynamic>))
              .toList()
          : [],
      furnitureSegments: segments is Map<String, dynamic>
          ? SegmentationResult.fromJson(segments)
          : null,
    );
  }
}
