import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class ARVisionView extends StatelessWidget {
  const ARVisionView({super.key});

  @override
  Widget build(BuildContext context) {
    return const Scaffold(
      // UiKitView จะทำการดึง View จากฝั่ง iOS มาเรนเดอร์ในพื้นที่นี้
      body: UiKitView(
        viewType: 'ar_camera_view', // ชื่อนี้ต้องตรงกับที่ตั้งไว้ใน Xcode
        creationParamsCodec: StandardMessageCodec(),
      ),
    );
  }
}