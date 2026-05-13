import 'package:flutter/material.dart';

class ARVisionView extends StatelessWidget {
  const ARVisionView({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        const Padding(
          padding: EdgeInsets.all(16.0),
          child: Text('เล็งกล้องไปที่มุมห้องบนพื้น แล้วกดปุ่มเพื่อวัดระยะ'),
        ),
        Expanded(
          child: Container(
            color: Colors.grey[300], 
            child: const Center(child: Text('AR Camera View (รอใส่ ar_flutter_plugin)')),
          ), 
        ),
      ],
    );
  }
}