import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

import 'core/services/auth_service.dart';
import 'core/theme/app_theme.dart';
import 'routes/app_router.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();

  await dotenv.load(fileName: 'assets/.env');

  await Supabase.initialize(
    url: dotenv.env['SUPABASE_URL']!,
    anonKey: dotenv.env['SUPABASE_ANON_KEY']!,
  );

  // Everyone is an anonymous user (no login screen): this is what makes saved
  // rooms private to a person and countable against the 5-room limit. The
  // session is kept on the device, so this only creates a user the first time.
  // Failing here (offline, anonymous sign-ins off) must not stop the app:
  // generating still works, and saving asks again and explains if it can't.
  try {
    await AuthService().ensureSignedIn();
  } catch (error) {
    debugPrint('Anonymous sign-in failed: $error');
  }

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