import '../../shared/models/furniture.dart';

import 'supabase_service.dart';

class FurnitureService {

  final supabase =
      SupabaseService().supabase;

  Future<List<Furniture>>
      getFurniture() async {

    final response =
        await supabase

            .from('ikea_furniture')

            .select()

            .limit(10);

    return (response as List)

        .map(
          (item) =>
              Furniture.fromJson(item),
        )

        .toList();
  }
}