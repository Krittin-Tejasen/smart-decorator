import 'package:flutter/material.dart';

/// One of the fixed accent-color palettes available under a [DesignStyle].
/// [colors] holds 1-2 swatch colors used to render a duo-tone preview circle.
class ColorOption {
  final String id;
  final String title;
  final List<Color> colors;

  ColorOption({
    required this.id,
    required this.title,
    required this.colors,
  });
}
