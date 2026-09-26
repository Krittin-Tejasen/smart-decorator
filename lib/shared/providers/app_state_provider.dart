import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'dart:io';

import '../../core/models/ar_capture_state.dart';
import '../models/design_style.dart';
import '../models/color_option.dart';
import '../models/room_type.dart';
import '../models/product.dart';
import '../models/design_history.dart';
import '../models/generate_room_request.dart';
import '../models/ai_model.dart';

import '../../core/services/ai_generation_service.dart';


class AppState {

  final RoomType? selectedRoomType;

  final DesignStyle? selectedStyle;

  final ColorOption? selectedColorOption;

  final String? generatedRoomImage;
  final List<Product> matchedProducts;

  final List<DesignHistory> history;

  final File? uploadedImage;

  final AiModel selectedAiModel;

  /// Spatial metadata from the last AR photo capture.
  /// Set alongside [uploadedImage] when the user takes a photo via AR Camera.
  final ARCaptureState? arCaptureState;

  /// Spatial metadata from the last LiDAR room scan (floor plan, ceiling
  /// height, mesh anchors, etc). Held in memory only — nothing is written to
  /// Supabase until the user explicitly saves on the results screen.
  final Map<String, dynamic>? scanSpatialData;

  /// True after the user successfully completes a room scan (buffered, not
  /// yet necessarily persisted).
  final bool scanCompleted;

  AppState({
    this.selectedRoomType,
    this.selectedStyle,
    this.selectedColorOption,
    this.generatedRoomImage,
    this.matchedProducts = const [],
    this.history = const [],
    this.uploadedImage,
    this.selectedAiModel = AiModel.serverDefault,
    this.arCaptureState,
    this.scanSpatialData,
    this.scanCompleted = false,
  });

  /// True if there's any captured room data (photo spatial metadata or LiDAR
  /// scan) sitting in the buffer, ready to be saved.
  bool get hasUnsavedCapture => arCaptureState != null || scanSpatialData != null;

  AppState copyWith({
    RoomType? selectedRoomType,
    DesignStyle? selectedStyle,
    ColorOption? selectedColorOption,
    File? uploadedImage,
    String? generatedRoomImage,
    List<DesignHistory>? history,
    List<Product>? matchedProducts,
    AiModel? selectedAiModel,
    ARCaptureState? arCaptureState,
    Map<String, dynamic>? scanSpatialData,
    bool? scanCompleted,
  }) {
    return AppState(
      selectedRoomType:    selectedRoomType    ?? this.selectedRoomType,
      selectedStyle:       selectedStyle       ?? this.selectedStyle,
      selectedColorOption: selectedColorOption ?? this.selectedColorOption,
      generatedRoomImage:  generatedRoomImage  ?? this.generatedRoomImage,
      matchedProducts:     matchedProducts     ?? this.matchedProducts,
      history:             history             ?? this.history,
      uploadedImage:       uploadedImage       ?? this.uploadedImage,
      selectedAiModel:     selectedAiModel     ?? this.selectedAiModel,
      arCaptureState:      arCaptureState      ?? this.arCaptureState,
      scanSpatialData:     scanSpatialData     ?? this.scanSpatialData,
      scanCompleted:       scanCompleted       ?? this.scanCompleted,
    );
  }

  bool get canGenerateDesign {
    return
          selectedRoomType != null
          && selectedStyle != null
          && selectedColorOption != null
          && uploadedImage != null
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

  void selectStyle(DesignStyle style) {
    // A style's color options are its own — a color picked under the
    // previous style may not exist here, so clear it rather than let a
    // stale selection silently carry over.
    state = AppState(
      selectedRoomType: state.selectedRoomType,
      selectedStyle: style,
      selectedColorOption: null,
      generatedRoomImage: state.generatedRoomImage,
      matchedProducts: state.matchedProducts,
      history: state.history,
      uploadedImage: state.uploadedImage,
      selectedAiModel: state.selectedAiModel,
    );
  }

  void selectColorOption(ColorOption colorOption) {
    state = state.copyWith(
      selectedColorOption: colorOption,
    );
  }

  void selectAiModel(AiModel model) {
    state = state.copyWith(
      selectedAiModel: model,
    );
  }

  void setUploadedImage(File image) {
    state = state.copyWith(uploadedImage: image);
  }

  void setARCaptureState(ARCaptureState arState) {
    state = state.copyWith(arCaptureState: arState);
  }

  /// Buffers LiDAR scan spatial data in memory. Nothing is sent to Supabase
  /// here — the results screen's Save button does that once, at the end.
  void setScanSpatialData(Map<String, dynamic> data) {
    state = state.copyWith(scanSpatialData: data);
  }

  void setScanCompleted() {
    state = state.copyWith(scanCompleted: true);
  }

  /// Clears the buffered capture data after it's been successfully saved.
  void clearCaptureBuffer() {
    state = AppState(
      selectedRoomType:    state.selectedRoomType,
      selectedStyle:       state.selectedStyle,
      selectedColorOption: state.selectedColorOption,
      generatedRoomImage:  state.generatedRoomImage,
      matchedProducts:     state.matchedProducts,
      history:             state.history,
      uploadedImage:       state.uploadedImage,
      selectedAiModel:     state.selectedAiModel,
      scanCompleted:       state.scanCompleted,
      // arCaptureState + scanSpatialData intentionally dropped
    );
  }

  void clearUploadedImage() {
    state = AppState(
      selectedRoomType:    state.selectedRoomType,
      selectedStyle:       state.selectedStyle,
      selectedColorOption: state.selectedColorOption,
      generatedRoomImage:  state.generatedRoomImage,
      matchedProducts:     state.matchedProducts,
      history:             state.history,
      selectedAiModel:     state.selectedAiModel,
      // arCaptureState intentionally cleared alongside image
    );
  }

  void saveToHistory() {
    if(
      state.selectedRoomType == null ||
      state.selectedStyle == null ||
      state.generatedRoomImage == null
    ){
      return ;
    }
    final historyItem = DesignHistory(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      roomType: state.selectedRoomType!.title,
      style: state.selectedStyle!.title,
      imagePath: state.generatedRoomImage!,
      products: state.matchedProducts,
      createdAt: DateTime.now(),
    );

    state = state.copyWith(
      history: [historyItem, ...state.history],
    );
  }

  Future<void> generateRoomDesign() async {
    if (
      state.selectedRoomType == null ||
      state.selectedStyle == null ||
      state.selectedColorOption == null ||
      state.uploadedImage == null
    ){
      return;
    }

    final request = GenerateRoomRequest(
      roomType: state.selectedRoomType!.id,
      style: state.selectedStyle!.id,
      color: state.selectedColorOption!.id,
      imagePath: state.uploadedImage!.path,
      provider: state.selectedAiModel.providerValue,
    );

    final aiService = AIGenerationService();

    final response = await aiService.generateRoom(request);

    state = state.copyWith(
      generatedRoomImage: response.generatedImage,
      matchedProducts: response.products,
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
