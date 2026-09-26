import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../../../core/constants/app_colors.dart';
import '../../../shared/models/furniture_item.dart';

/// Bytes of a `data:image/...;base64,...` URL (or bare base64), or null if it
/// is empty or not valid base64.
Uint8List? decodeDataUrl(String? dataUrl) {
  if (dataUrl == null || dataUrl.isEmpty) return null;
  final payload = dataUrl.contains(',')
      ? dataUrl.substring(dataUrl.indexOf(',') + 1)
      : dataUrl;
  try {
    return base64Decode(payload);
  } on FormatException {
    return null;
  }
}

/// A detected furniture item on the app's sand-coloured tile.
///
/// When the backend produced a real mask the item is shown cut out of the room
/// (like a product photo); otherwise the rectangular crop is shown instead.
/// Fills whatever space it is given.
class FurnitureImage extends StatefulWidget {
  final FurnitureItem item;
  final double borderRadius;

  const FurnitureImage({
    super.key,
    required this.item,
    this.borderRadius = 10,
  });

  @override
  State<FurnitureImage> createState() => _FurnitureImageState();
}

class _FurnitureImageState extends State<FurnitureImage> {
  Uint8List? _cutout;
  Uint8List? _crop;

  @override
  void initState() {
    super.initState();
    _decode();
  }

  @override
  void didUpdateWidget(FurnitureImage oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!identical(oldWidget.item, widget.item)) _decode();
  }

  // Decoded once per item, not on every rebuild.
  void _decode() {
    _cutout = decodeDataUrl(widget.item.cutoutImage);
    _crop = decodeDataUrl(widget.item.cropImage);
  }

  Widget _fallbackIcon() => const Center(
        child: Icon(Icons.chair_rounded, size: 26, color: AppColors.muted),
      );

  @override
  Widget build(BuildContext context) {
    final Widget content;
    if (_cutout != null) {
      content = Padding(
        padding: const EdgeInsets.all(8),
        child: Image.memory(
          _cutout!,
          key: const ValueKey('furniture-cutout'),
          fit: BoxFit.contain,
          gaplessPlayback: true,
          errorBuilder: (context, error, stackTrace) => _fallbackIcon(),
        ),
      );
    } else if (_crop != null) {
      content = Image.memory(
        _crop!,
        key: const ValueKey('furniture-crop'),
        fit: BoxFit.cover,
        gaplessPlayback: true,
        errorBuilder: (context, error, stackTrace) => _fallbackIcon(),
      );
    } else {
      content = _fallbackIcon();
    }

    return ClipRRect(
      borderRadius: BorderRadius.circular(widget.borderRadius),
      child: ColoredBox(
        color: AppColors.sandTint,
        child: SizedBox.expand(child: content),
      ),
    );
  }
}
