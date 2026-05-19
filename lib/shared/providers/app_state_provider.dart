import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'dart:io';

import '../models/design_theme.dart';
import '../models/room_type.dart';
import '../models/product.dart';
import '../models/design_history.dart';

class AppState {

  final RoomType? selectedRoomType;

  final DesignTheme? selectedTheme;

  final String? generatedRoomImage;
  final List<Product> matchedProducts;

  final List<DesignHistory> history;

  final File? uploadedImage;


  AppState({
    this.selectedRoomType,
    this.selectedTheme,
    this.generatedRoomImage,
    this.matchedProducts = const [],
    this.history = const [],
    this.uploadedImage,
  });

  AppState copyWith({
    RoomType? selectedRoomType,
    DesignTheme? selectedTheme,
    File? uploadedImage,
    String? generatedRoomImage,
    List<DesignHistory>? history,
    List<Product>? matchedProducts,
  }) {
    return AppState(
      selectedRoomType:
          selectedRoomType ?? this.selectedRoomType,

      selectedTheme:
          selectedTheme ?? this.selectedTheme,

      generatedRoomImage:
          generatedRoomImage ?? this.generatedRoomImage,
          
      matchedProducts:
          matchedProducts ?? this.matchedProducts,
      
      history:
          history ?? this.history,

      uploadedImage: 
          uploadedImage ?? this.uploadedImage,
    );
  }

  bool get canGenerateDesign {
    return 
          selectedRoomType != null 
          && selectedTheme != null 
          // && uploadedImage != null
          ;
  }

}

class AppStateNotifier 
  extends StateNotifier<AppState> {

  AppStateNotifier(): 
    super(
      AppState(),
    );

  void selectRoomType(RoomType roomType) {
    state = state.copyWith(
      selectedRoomType: roomType,
    );
  }

  void selectTheme(DesignTheme theme) {
    state = state.copyWith(
      selectedTheme: theme,
    );
  }

  void setUploadedImage(File image) {
    state = state.copyWith(
      uploadedImage: image,
    );
  }

  void saveToHistory() {
    if(
      state.selectedRoomType == null ||
      state.selectedTheme == null ||
      state.generatedRoomImage == null
    ){
      return ;
    }
    final historyItem = DesignHistory(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      roomType: state.selectedRoomType!.title,
      theme: state.selectedTheme!.title,
      imagePath: state.generatedRoomImage!,
      products: state.matchedProducts,
      createdAt: DateTime.now(),
    );

    state = state.copyWith(
      history: [historyItem, ...state.history],
    );
  }

  void setFakeResults() {
    state = state.copyWith(
      generatedRoomImage:
        'fake_generated_room',
      
      matchedProducts: [
        Product(
          id: '1',
          name: 'Modern Sofa',
          imageUrl: 'https://example.com/sofa.jpg',
          price: 12990.00,
        ),

        Product(
          id: '2',
          name: 'Minimal Lamp',
          imageUrl: 'https://example.com/lamp.jpg',
          price: 2490,
        ),

        Product(
          id: '3',
          name: 'Wooden Table',
          imageUrl: 'https://example.com/table.jpg',
          price: 7990,
        ),

      ]
    );
  }
}

final appStateProvider = StateNotifierProvider<AppStateNotifier, AppState>((ref) {
  return AppStateNotifier();
});