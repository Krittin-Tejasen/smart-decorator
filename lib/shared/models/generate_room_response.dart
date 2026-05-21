import 'product.dart';

class GenerateRoomResponse {

  final String generatedImage;
  final List<Product> products;

  GenerateRoomResponse({
    required this.generatedImage,
    required this.products,
  });
}