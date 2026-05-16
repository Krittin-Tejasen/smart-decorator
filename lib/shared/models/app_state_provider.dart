import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../models/design_theme.dart';
import '../models/room_type.dart';

class AppState {

  final RoomType? selectedRoomType;

  final DesignTheme? selectedTheme;

  AppState({
    this.selectedRoomType,
    this.selectedTheme,
  });

  AppState copyWith({
    RoomType? selectedRoomType,
    DesignTheme? selectedTheme,
  }) {
    return AppState(
      selectedRoomType:
          selectedRoomType ?? this.selectedRoomType,

      selectedTheme:
          selectedTheme ?? this.selectedTheme,
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
}

final appStateProvider = StateNotifierProvider<AppStateNotifier, AppState>((ref) {
  return AppStateNotifier();
});