import 'package:flutter/material.dart';

class ManualBuilderView extends StatefulWidget {
  const ManualBuilderView({super.key});

  @override
  State<ManualBuilderView> createState() => _ManualBuilderViewState();
}

class _ManualBuilderViewState extends State<ManualBuilderView> {
  double width = 4.0;
  double length = 4.0;

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: Center(
            child: Container(
              width: width * 40, // อัตราส่วนจำลอง (1 เมตร = 40 pixels)
              height: length * 40,
              decoration: BoxDecoration(
                color: Colors.blue.withOpacity(0.2),
                border: Border.all(color: Colors.blue, width: 2),
              ),
              child: Center(
                child: Text('${width.toStringAsFixed(1)}m x ${length.toStringAsFixed(1)}m', 
                  style: const TextStyle(fontWeight: FontWeight.bold)),
              ),
            ),
          ),
        ),
        Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            children: [
              const Text('ระบุขนาดห้อง', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              const SizedBox(height: 16),
              Row(
                children: [
                  const Text('กว้าง (W):'),
                  Expanded(
                    child: Slider(
                      value: width,
                      min: 2.0, max: 10.0, divisions: 80,
                      onChanged: (val) => setState(() => width = val),
                    ),
                  ),
                  Text('${width.toStringAsFixed(1)}m'),
                ],
              ),
              Row(
                children: [
                  const Text('ยาว (L):  '),
                  Expanded(
                    child: Slider(
                      value: length,
                      min: 2.0, max: 10.0, divisions: 80,
                      onChanged: (val) => setState(() => length = val),
                    ),
                  ),
                  Text('${length.toStringAsFixed(1)}m'),
                ],
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: () {
                  // TODO: จัดเก็บข้อมูลเป็น JSON เตรียมส่งขึ้น Cloud ให้ AI
                  print("Room Size Saved: $width x $length");
                },
                child: const Text('ยืนยันขนาดห้อง'),
              )
            ],
          ),
        )
      ],
    );
  }
}