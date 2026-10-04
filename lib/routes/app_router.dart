import 'package:flutter/widgets.dart';
import 'package:go_router/go_router.dart';

import '../features/home/presentation/home_screen.dart';
import '../features/splash/presentation/splash_screen.dart';
import '../features/processing/presentation/processing_screen.dart';
import '../features/results/presentation/furniture_matches_screen.dart';
import '../features/results/presentation/results_screen.dart';
import '../shared/models/furniture_item.dart';
import '../shared/models/saved_design.dart';
import '../features/test/presentation/test_screen.dart';
import '../features/history/presentation/history_screen.dart';
import '../features/history/presentation/saved_design_screen.dart';

import '../screens/scan/room_scanner_screen.dart';
import '../screens/scan/scan_result_screen.dart';


/// Products matching one detected piece of furniture. The item travels as
/// `extra`; without one (after a hot restart, a reload, or the browser's back
/// button on web, which can't keep an object in its history) there is nothing
/// to show, so it goes back to the results. Public so tests can mount the real
/// route.
final GoRoute furnitureMatchesRoute = GoRoute(
  path: '/furniture-matches',
  redirect: (BuildContext context, GoRouterState state) =>
      state.extra is FurnitureItem ? null : '/results',
  builder: (BuildContext context, GoRouterState state) =>
      buildFurnitureMatchesPage(state.extra),
);

/// The matches page for [extra], or -- when a navigation path skipped the
/// redirect above and [extra] is missing -- a page that immediately returns to
/// the results instead of crashing on a bad cast (that showed the red error
/// screen when going back with the browser's button).
Widget buildFurnitureMatchesPage(Object? extra) {
  return extra is FurnitureItem
      ? FurnitureMatchesScreen(item: extra)
      : const _BackTo('/results');
}

/// A saved room opened from History. Same rule as above: the room travels as
/// `extra`, and without one we go back to the list instead of crashing.
final GoRoute savedDesignRoute = GoRoute(
  path: '/saved-design',
  redirect: (BuildContext context, GoRouterState state) =>
      state.extra is SavedDesign ? null : '/history',
  builder: (BuildContext context, GoRouterState state) =>
      buildSavedDesignPage(state.extra),
);

Widget buildSavedDesignPage(Object? extra) {
  return extra is SavedDesign
      ? SavedDesignScreen(design: extra)
      : const _BackTo('/history');
}

/// Immediately sends the user to [location]; the page a route builds when the
/// object it needs (`extra`) is missing.
class _BackTo extends StatefulWidget {
  final String location;

  const _BackTo(this.location);

  @override
  State<_BackTo> createState() => _BackToState();
}

class _BackToState extends State<_BackTo> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) context.go(widget.location);
    });
  }

  @override
  Widget build(BuildContext context) => const SizedBox.shrink();
}

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
    furnitureMatchesRoute,
    savedDesignRoute,
    GoRoute(
      path: '/history',
      builder: (BuildContext context, GoRouterState state) => const HistoryScreen(),
    ),

    GoRoute(
      path: '/test',
      builder: (BuildContext context, GoRouterState state) => const TestScreen(),
    ),
    GoRoute(
      path: '/scan_room',
      builder: (context, state) => const RoomScannerScreen(),
    ),
    GoRoute(
      path: '/scan_result',
      builder: (context, state) {
        final data = state.extra as Map<String, dynamic>? ?? {};
        return ScanResultScreen(data: data);
      },
    ),
  ],
);