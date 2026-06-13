import 'package:flutter/widgets.dart';
import 'package:go_router/go_router.dart';

import '../features/home/presentation/home_screen.dart';
import '../features/splash/presentation/splash_screen.dart';
import '../features/processing/presentation/processing_screen.dart';
import '../features/results/presentation/results_screen.dart';
import '../features/history/presentation/history_screen.dart';

import '../screens/scan/room_scanner_screen.dart';


final GoRouter appRouter = GoRouter(
  initialLocation: '/',
  routes: [
    GoRoute(
      path: '/',
      builder: (BuildContext context, GoRouterState state) => const SplashScreen(),
    ),
    GoRoute(
      path: '/home',
      builder: (BuildContext context, GoRouterState state) => const HomeScreen(),
    ),
    GoRoute(
      path: '/processing',
      builder: (BuildContext context, GoRouterState state) => const ProcessingScreen(),
    ),
    GoRoute(
      path: '/results',
      builder: (BuildContext context, GoRouterState state) => const ResultsScreen(),
    ),
    GoRoute(
      path: '/history',
      builder: (context, state) => const HistoryScreen(),
    ),
    GoRoute(
      path: '/scan_room',
      builder: (context, state) => const RoomScannerScreen(),
    ),
  ],
);