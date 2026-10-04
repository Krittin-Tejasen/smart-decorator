import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'dart:io';
import 'dart:typed_data';

import '../../core/models/ar_capture_state.dart';
import '../models/design_style.dart';
import '../models/color_option.dart';
import '../models/room_type.dart';
import '../models/product.dart';
import '../models/generate_room_request.dart';
import '../models/ai_model.dart';
import '../models/furniture_item.dart';
import '../models/generation_progress.dart';
import '../models/saved_design.dart';

import '../../core/services/ai_generation_service.dart';
import '../../core/services/design_repository.dart';
import '../utils/data_url.dart';


class AppState {

  final RoomType? selectedRoomType;

  final DesignStyle? selectedStyle;

  final ColorOption? selectedColorOption;

  final String? generatedRoomImage;
  final List<Product> matchedProducts;

  /// Furniture pieces the backend segmented out of [generatedRoomImage] on
  /// the last successful generation — one entry per detected item.
  final List<FurnitureItem> segmentedFurniture;

  final File? uploadedImage;

  final AiModel selectedAiModel;

  /// True while "Save Room Data" is uploading the current room.
  final bool isSavingDesign;

  /// True once the current generated room has been saved (reset by the next
  /// generation), so the same result can't be saved twice.
  final bool designSaved;

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
    this.segmentedFurniture = const [],
    this.uploadedImage,
    this.selectedAiModel = AiModel.serverDefault,
    this.isSavingDesign = false,
    this.designSaved = false,
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
    List<Product>? matchedProducts,
    List<FurnitureItem>? segmentedFurniture,
    AiModel? selectedAiModel,
    bool? isSavingDesign,
    bool? designSaved,
    ARCaptureState? arCaptureState,
    Map<String, dynamic>? scanSpatialData,
    bool? scanCompleted,
  }) {
    return AppState(
      selectedRoomType:
          selectedRoomType ?? this.selectedRoomType,

      selectedStyle:
          selectedStyle ?? this.selectedStyle,

      selectedColorOption:
          selectedColorOption ?? this.selectedColorOption,

      generatedRoomImage:
          generatedRoomImage ?? this.generatedRoomImage,

      matchedProducts:
          matchedProducts ?? this.matchedProducts,

      segmentedFurniture:
          segmentedFurniture ?? this.segmentedFurniture,

      uploadedImage:
          uploadedImage ?? this.uploadedImage,

      selectedAiModel:
          selectedAiModel ?? this.selectedAiModel,

      isSavingDesign:
          isSavingDesign ?? this.isSavingDesign,

      designSaved:
          designSaved ?? this.designSaved,

      arCaptureState:
          arCaptureState ?? this.arCaptureState,

      scanSpatialData:
          scanSpatialData ?? this.scanSpatialData,

      scanCompleted:
          scanCompleted ?? this.scanCompleted,
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

/// What a successful save reports back: the saved room and how full the
/// user's saved rooms now are.
class SaveRoomResult {
  final SavedDesign design;
  final int savedCount;
  final int limit;

  const SaveRoomResult({
    required this.design,
    required this.savedCount,
    required this.limit,
  });
}

class AppStateNotifier
  extends StateNotifier<AppState> {

  AppStateNotifier({
    DesignRepository Function()? repositoryFactory,
    Future<Uint8List> Function(File file)? sourceReader,
  }):
    _repositoryFactory = repositoryFactory,
    _readSource = sourceReader ?? ((file) => file.readAsBytes()),
    super(
      AppState(),
    );

  final DesignRepository Function()? _repositoryFactory;

  /// How the photo the room was made from is read for saving (a seam for tests).
  final Future<Uint8List> Function(File file) _readSource;

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
      segmentedFurniture: state.segmentedFurniture,
      uploadedImage: state.uploadedImage,
      selectedAiModel: state.selectedAiModel,
      isSavingDesign: state.isSavingDesign,
      designSaved: state.designSaved,
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
      segmentedFurniture:  state.segmentedFurniture,
      uploadedImage:       state.uploadedImage,
      selectedAiModel:     state.selectedAiModel,
      isSavingDesign:      state.isSavingDesign,
      designSaved:         state.designSaved,
      scanCompleted:       state.scanCompleted,
      // arCaptureState + scanSpatialData intentionally dropped
    );
  }

  void clearUploadedImage() {
    state = AppState(
      selectedRoomType:    state.selectedRoomType,
      selectedStyle:       state.selectedStyle,
      selectedColorOption: state.selectedColorOption,
      generatedRoomImage: state.generatedRoomImage,
      matchedProducts: state.matchedProducts,
      segmentedFurniture: state.segmentedFurniture,
      selectedAiModel: state.selectedAiModel,
      isSavingDesign: state.isSavingDesign,
      designSaved: state.designSaved,
      // arCaptureState intentionally cleared alongside image
    );
  }

  /// Generates the design as a backend job. [onProgress] is called with what
  /// the backend is really doing (stage + status text) while it runs;
  /// cancelling [cancelToken] stops the job. Throws [GenerationCancelled]
  /// when cancelled.
  Future<void> generateRoomDesign({
    void Function(GenerationProgress progress)? onProgress,
    GenerationCancelToken? cancelToken,
  }) async {
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

    final response = await aiService.generateRoomWithProgress(
      request,
      onProgress: onProgress,
      cancelToken: cancelToken,
    );

    // A fresh result starts unsaved (nothing is stored until the user taps
    // "Save Room Data"), so build the state directly: copyWith's ?? pattern
    // couldn't reset designSaved.
    state = AppState(
      selectedRoomType: state.selectedRoomType,
      selectedStyle: state.selectedStyle,
      selectedColorOption: state.selectedColorOption,
      generatedRoomImage: response.generatedImage,
      matchedProducts: response.products,
      segmentedFurniture: response.furnitureSegments?.items ?? const [],
      uploadedImage: state.uploadedImage,
      selectedAiModel: state.selectedAiModel,
      isSavingDesign: false,
      designSaved: false,
    );
  }

  /// Saves the current room -- the generated image and the photo it came
  /// from -- to the user's saved rooms. Throws [SaveLimitReachedException] when
  /// they already have the maximum, [NotSignedInException] when there is no
  /// user to save under, and a [StateError] if there is nothing to save yet.
  Future<SaveRoomResult> saveCurrentRoom() async {
    final repository = _repositoryFactory?.call();
    final generated = decodeDataUrl(state.generatedRoomImage);
    final source = state.uploadedImage;
    final roomType = state.selectedRoomType;
    final style = state.selectedStyle;
    final color = state.selectedColorOption;

    if (repository == null ||
        generated == null ||
        source == null ||
        roomType == null ||
        style == null ||
        color == null) {
      throw StateError('There is no generated room to save yet.');
    }
    if (state.isSavingDesign || state.designSaved) {
      throw StateError('This room is already saved.');
    }

    state = state.copyWith(isSavingDesign: true);

    try {
      final saved = await repository.save(
        DesignToSave(
          generatedImage: generated,
          sourceImage: await _readSource(source),
          sourceExtension: _extensionOf(source.path),
          roomType: roomType.id,
          style: style.id,
          color: color.id,
        ),
      );
      final count = (await repository.list()).length;

      state = state.copyWith(isSavingDesign: false, designSaved: true);
      return SaveRoomResult(
        design: saved,
        savedCount: count,
        limit: repository.maxSavedDesigns,
      );
    } catch (_) {
      state = state.copyWith(isSavingDesign: false);
      rethrow;
    }
  }

  static String _extensionOf(String path) {
    final dot = path.lastIndexOf('.');
    final extension = dot < 0 ? '' : path.substring(dot + 1).toLowerCase();
    return const {'jpg', 'jpeg', 'png', 'webp', 'heic'}.contains(extension)
        ? extension
        : 'jpg';
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
  return AppStateNotifier(
    repositoryFactory: () => ref.read(designRepositoryProvider),
  );
});
