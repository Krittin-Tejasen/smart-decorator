import 'package:flutter/material.dart';
import '../../services/hardware_service.dart';
import 'views/lidar_view.dart';
import 'views/ar_vision_view.dart';
import 'views/manual_builder.dart';

class RoomScannerScreen extends StatefulWidget {
  const RoomScannerScreen({super.key});

  @override
  State<RoomScannerScreen> createState() => _RoomScannerScreenState();
}

class _RoomScannerScreenState extends State<RoomScannerScreen> {
  String _scanMode = 'loading'; // สถานะเริ่มต้น

  @override
  void initState() {
    super.initState();
    _determineScanMode();
  }

  Future<void> _determineScanMode() async {
    bool hasLidar = await HardwareService.hasLidarSensor();
    
    setState(() {
      if (hasLidar) {
        _scanMode = 'lidar';
      } else {
        _scanMode = 'ar_vision'; 
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    if (_scanMode == 'loading') {
      return const Scaffold(body: Center(child: CircularProgressIndicator()));
    }

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Room'),
        actions: [
          // ปุ่มสลับไปโหมดวาดมือเอง (Level 3) เผื่อผู้ใช้ไม่สะดวกใช้กล้อง
          IconButton(
            icon: const Icon(Icons.edit_note),
            onPressed: () => setState(() => _scanMode = 'manual'),
          )
        ],
      ),
      body: _buildScannerBody(),
    );
  }

  Widget _buildScannerBody() {
    switch (_scanMode) {
      case 'lidar':
        return const LidarView(); 
      case 'ar_vision':
        return const ARVisionView(); 
      case 'manual':
      default:
        return const ManualBuilderView(); 
    }
  }
}