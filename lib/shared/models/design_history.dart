import 'product.dart';

class DesignHistory {
  final String id;
  final String roomType;
  final String style;
  final String imagePath;
  final List<Product> products;
  final DateTime createdAt;

  DesignHistory({
    required this.id,
    required this.roomType,
    required this.style,
    required this.imagePath,
    required this.products,
    required this.createdAt,
  });
}