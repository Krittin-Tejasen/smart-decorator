import 'product.dart';

class GenerateRoomResponse {

  final String generatedImage;
  final List<Product> products;

  GenerateRoomResponse({
    required this.generatedImage,
    required this.products,
  });

  factory GenerateRoomResponse.fromJson(Map<String, dynamic> json) {
    final productItems = json['products'];

    return GenerateRoomResponse(
      generatedImage: json['generated_image'] as String? ?? '',
      products: productItems is List
          ? productItems
              .map((item) => Product.fromJson(item as Map<String, dynamic>))
              .toList()
          : [],
    );
  }
}
