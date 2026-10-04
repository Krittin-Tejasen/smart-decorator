import 'dart:convert';
import 'dart:typed_data';

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
