class Furniture {

  final String sku;

  final String title;

  final String imageUrl;

  final double price;

  final String category;

  Furniture({
    required this.sku,
    required this.title,
    required this.imageUrl,
    required this.price,
    required this.category,
  });

  factory Furniture.fromJson(
    Map<String, dynamic> json,
  ) {

    return Furniture(

      sku: json['sku'],

      title: json['main_title'],

      imageUrl:
          json['main_image'] ?? '',

      price:
          (json['thb_price'] ?? 0)
              .toDouble(),

      category:
          json['category'] ?? '',
    );
  }
}