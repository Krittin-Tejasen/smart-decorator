import '../../core/services/product_match_service.dart' show titleCase;

// The ids the app saves with a room, mapped back to the titles shown on the
// Home screen. Keep in sync with the lists in home_screen.dart.
const _roomTypeTitles = <String, String>{
  'living_room': 'Living Room',
  'bedroom': 'Bedroom',
  'dining_room': 'Dining Room',
  'kitchen': 'Kitchen',
  'home_office': 'Home Office',
};

const _styleTitles = <String, String>{
  'japandi': 'Japandi',
  'industrial_loft': 'Industrial / Loft',
  'modern_luxury': 'Modern Luxury',
};

String roomTypeTitle(String id) => _roomTypeTitles[id] ?? titleCase(id);

String styleTitle(String id) => _styleTitles[id] ?? titleCase(id);

const _months = [
  'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
];

/// "27 Sep 2026, 14:05" (local time).
String formatSavedAt(DateTime time) {
  final local = time.toLocal();
  final hh = local.hour.toString().padLeft(2, '0');
  final mm = local.minute.toString().padLeft(2, '0');
  return '${local.day} ${_months[local.month - 1]} ${local.year}, $hh:$mm';
}
