import 'package:flutter/material.dart';

import '../../../../core/constants/app_colors.dart';

import '../../../../shared/models/product.dart';

class ProductCard extends StatelessWidget {

  final Product product;

  const ProductCard({
    super.key,
    required this.product,
  });

  @override
  Widget build(BuildContext context) {

    return Container(
      margin: const EdgeInsets.only(
        bottom: 20,
      ),

      padding: const EdgeInsets.all(18),

      decoration: BoxDecoration(
        color: Colors.white,

        borderRadius:
            BorderRadius.circular(28),

        boxShadow: [

          BoxShadow(
            color: Colors.black.withValues(alpha: 0.04),
            blurRadius: 10,
            offset: const Offset(0,4,),
          ),
        ],
      ),

      child: Row(
        children: [

          Container(
            width: 110,
            height: 110,

            decoration: BoxDecoration(
              color: Colors.grey.shade600,
              borderRadius: BorderRadius.circular(24),
            ),

            child: const Center(
              child: Text(
                'Product\nImage',

                textAlign:
                    TextAlign.center,

                style: TextStyle(
                  color: Colors.white,

                  fontSize: 20,
                ),
              ),
            ),
          ),

          const SizedBox(width: 20),

          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  product.name,
                  style:
                      const TextStyle(
                    fontSize: 26,
                    fontWeight:
                        FontWeight.bold,
                  ),
                ),

                const SizedBox(height: 12),

                Text(
                  '฿ ${product.price.toStringAsFixed(0)}',

                  style: const TextStyle(fontSize: 18,),
                ),

                const SizedBox(height: 20),

                Align(
                  alignment: Alignment.centerRight,

                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor:AppColors.primary,
                      foregroundColor:Colors.white,
                    ),
                    onPressed: () {},
                    child: const Text(
                      'View Product',
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