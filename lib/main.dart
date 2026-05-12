import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

void main() {
  runApp(
    const ProviderScope(
      child: SmartDecoratorApp(),
    ),
  );
}

class SmartDecoratorApp extends StatelessWidget {
  const SmartDecoratorApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Smart Decorator',
      home: Scaffold(
        body: Center(
          child: Text('Smart Decorator'),
        ),
      ),
    );
  }
}