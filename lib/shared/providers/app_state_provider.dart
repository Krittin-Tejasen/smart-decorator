import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'dart:io';

import '../models/design_theme.dart';
import '../models/room_type.dart';

class AppState {

  final RoomType? selectedRoomType;

  final DesignTheme? selectedTheme;

  final File? uploadedImage;

  AppState({
    this.selectedRoomType,
    this.selectedTheme,
    this.uploadedImage,
  });

  AppState copyWith({
    RoomType? selectedRoomType,
    DesignTheme? selectedTheme,
    File? uploadedImage,
  }) {
    return AppState(
      selectedRoomType:
          selectedRoomType ?? this.selectedRoomType,

      selectedTheme:
          selectedTheme ?? this.selectedTheme,

      uploadedImage: 
          uploadedImage ?? this.uploadedImage,
    );
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
}

final appStateProvider = StateNotifierProvider<AppStateNotifier, AppState>((ref) {
  return AppStateNotifier();
});