import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class HardwareService {
  static const platform = MethodChannel('com.smartdeco.app/sensors');

  static Future<bool> hasLidarSensor() async {
    // ถ้าไม่ใช่ iOS ให้ข้ามไปเลย เพราะ Android ไม่มีเซนเซอร์ LiDAR ของ Apple
    if (!Platform.isIOS) return false; 
    
    try {
      final bool hasLidar = await platform.invokeMethod('checkLidar');
      return hasLidar;
    } on PlatformException catch (e) {
      debugPrint("Failed to check LiDAR: '${e.message}'.");
      return false;
    }
  }
}