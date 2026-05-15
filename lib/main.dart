import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'core/theme/app_theme.dart';
import 'routes/app_router.dart';

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
    return MaterialApp.router(
      debugShowCheckedModeBanner: false,
      title: 'Smart Decorator',
      theme: AppTheme.lightTheme,
      routerConfig: appRouter,
    );
  }
}