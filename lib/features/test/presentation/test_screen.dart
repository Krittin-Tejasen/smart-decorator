import 'package:flutter/material.dart';
import '../../../core/services/furniture_service.dart';
import '../../../shared/models/furniture.dart';


class TestScreen
    extends StatefulWidget {

  const TestScreen({super.key});

  @override
  State<TestScreen> createState() =>
      _TestScreenState();
}

class _TestScreenState
    extends State<TestScreen> {

  List<Furniture> furniture = [];

  bool isLoading = true;

  @override
  void initState() {
    super.initState();

    loadFurniture();
  }

  Future<void> loadFurniture() async {

    final result =
        await FurnitureService()
            .getFurniture();

    setState(() {

      furniture = result;

      isLoading = false;
    });
  }

  @override
  Widget build(BuildContext context) {

    return Scaffold(

      appBar: AppBar(
        title: const Text(
          'Supabase Test',
        ),
      ),

      body: isLoading

          ? const Center(
              child:
                  CircularProgressIndicator(),
            )

          : ListView.builder(

              itemCount: furniture.length,

              itemBuilder: (
                context,
                index,
              ) {

                final item =
                    furniture[index];

                return ListTile(

                  leading:
                      item.imageUrl.isNotEmpty

                          ? Image.network(
                              item.imageUrl,
                              width: 60,
                              fit: BoxFit.cover,
                            )

                          : const Icon(
                              Icons.chair,
                            ),

                  title: Text(
                    item.title,
                  ),

                  subtitle: Text(
                    '฿ ${item.price}',
                  ),
                );
              },
            ),
    );
  }
}