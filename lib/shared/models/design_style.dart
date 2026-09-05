import 'color_option.dart';

/// A design/decor style (formerly "DesignTheme"). Each style owns its own
/// fixed set of [colorOptions] — color is chosen after style, not freely
/// mixed, so the AI prompt never ends up with a self-contradictory
/// combination (e.g. jewel-tone colors on a raw-industrial style).
class DesignStyle {
  final String id;
  final String title;
  final String subtitle;
  final List<ColorOption> colorOptions;

  DesignStyle({
    required this.id,
    required this.title,
    required this.subtitle,
    required this.colorOptions,
  });
}
