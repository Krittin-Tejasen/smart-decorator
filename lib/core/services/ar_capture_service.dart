import 'dart:io';
import 'package:flutter/services.dart';
import '../models/ar_capture_state.dart';

/// Flutter bridge for native AR photo capture.
///
/// iOS path: the caller opens [ARPhotoCaptureView] which wraps a UiKitView.
/// The view exposes `capturePhoto` on its per-view MethodChannel
/// (`com.smartdeco.app/ar_camera_{viewId}`); the result is routed back to Dart
/// inside [ARPhotoCaptureView] itself.
///
/// Android path: calling [capturePhoto] on this service fires the global channel
/// (`com.smartdeco.app/ar_capture`). ARCoreCapturePlugin starts
/// ARCoreCaptureActivity (full-screen native UI), and the result is delivered
/// back here when the Activity finishes.
class ARCaptureService {
  static const MethodChannel _androidChannel =
      MethodChannel('com.smartdeco.app/ar_capture');

  /// Opens the native AR capture experience and waits for the user to take a photo.
  ///
  /// On iOS — **do not call this**; the view-level channel inside
  /// [ARPhotoCaptureView] handles capture directly.
  ///
  /// On Android — starts [ARCoreCaptureActivity] and returns the result.
  /// Returns `null` if the user cancels.
  static Future<ARCaptureState?> captureAndroid() async {
    assert(Platform.isAndroid, 'captureAndroid() is for Android only');
    try {
      final raw = await _androidChannel.invokeMethod<Map>('capturePhoto');
      if (raw == null) return null;
      return ARCaptureState.fromMap(raw);
    } on PlatformException catch (e) {
      if (e.code == 'CANCELLED') return null;
      rethrow;
    }
  }
}
