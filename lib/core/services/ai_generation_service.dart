import '../../shared/models/generate_room_request.dart';
import '../../shared/models/generate_room_response.dart';
import '../../shared/models/product.dart';

class AIGenerationService {

  Future<GenerateRoomResponse> generateRoom(
    GenerateRoomRequest request,
  ) async {

    await Future.delayed(
      const Duration(seconds: 2),
    );

    return GenerateRoomResponse(

      generatedImage:
          'fake_generated_room',

      products: [

        Product(
          id: '1',
          name: 'Modern Sofa',
          imageUrl: 'sofa',
          price: 12990,
        ),

        Product(
          id: '2',
          name: 'Minimal Lamp',
          imageUrl: 'lamp',
          price: 2490,
        ),

        Product(
          id: '3',
          name: 'Wooden Table',
          imageUrl: 'table',
          price: 7990,
        ),
      ],
    );
  }
}