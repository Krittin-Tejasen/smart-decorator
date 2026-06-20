import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import 'core/theme/app_theme.dart';
import 'routes/app_router.dart';

import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

Future<void> main() async {

  WidgetsFlutterBinding.ensureInitialized();

  // print("STEP 1");

  await dotenv.load(
    fileName: "assets/.env",
    
  );

  // print("STEP 2");

  await Supabase.initialize(
    url: dotenv.env['SUPABASE_URL']!,

    anonKey:
        dotenv.env['SUPABASE_ANON_KEY']!,
  );

  // print("STEP 3");

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