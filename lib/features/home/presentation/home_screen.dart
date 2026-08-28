import 'dart:io' show File, Platform;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/widgets/app_icons.dart';
import '../../../shared/providers/app_state_provider.dart';
import '../../../shared/models/design_style.dart';
import '../../../shared/models/color_option.dart';
import '../../../shared/models/room_type.dart';
import '../../../shared/widgets/app_footer_nav.dart';
import '../../../shared/widgets/ai_model_picker.dart';

class HomeScreen extends ConsumerStatefulWidget {
  const HomeScreen({super.key});

  @override
  ConsumerState<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends ConsumerState<HomeScreen> {
  final ScrollController _roomTypeScrollController = ScrollController();

  @override
  void dispose() {
    _roomTypeScrollController.dispose();
    super.dispose();
  }

  Future<void> pickImage(WidgetRef ref) async {
    final picker = ImagePicker();

    final pickedFile = await picker.pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      ref
          .read(appStateProvider.notifier)
          .setUploadedImage(File(pickedFile.path));
    }
  }

  @override
  Widget build(BuildContext context) {
    // Room Types the app supports
    final roomTypes = [
      RoomType(
        id: 'living_room',
        title: 'Living Room',
        icon: Icons.weekend_rounded,
      ),

      RoomType(id: 'bedroom', title: 'Bedroom', icon: Icons.bed_rounded),

      RoomType(
        id: 'dining_room',
        title: 'Dining Room',
        icon: Icons.table_restaurant_rounded,
      ),

      RoomType(id: 'kitchen', title: 'Kitchen', icon: Icons.kitchen_rounded),

      RoomType(
        id: 'home_office',
        title: 'Home Office',
        icon: Icons.work_rounded,
      ),
    ];

    // Design Styles, each with its own fixed set of accent-color palettes
    final styles = [
      DesignStyle(
        id: 'japandi',
        title: 'Japandi',
        subtitle: 'Warm Minimalist',
        colorOptions: [
          ColorOption(
            id: 'warm_oat_cream',
            title: 'Warm Oat & Cream',
            colors: const [Color(0xFFE8DCC8), Color(0xFFF7F2E9)],
          ),
          ColorOption(
            id: 'muted_sage_green',
            title: 'Muted Sage Green',
            colors: const [Color(0xFFA7B29A), Color(0xFF7E8B76)],
          ),
          ColorOption(
            id: 'soft_terracotta',
            title: 'Soft Terracotta',
            colors: const [Color(0xFFE0977C), Color(0xFFEFC3A8)],
          ),
        ],
      ),

      DesignStyle(
        id: 'industrial_loft',
        title: 'Industrial / Loft',
        subtitle: 'Raw & Edgy',
        colorOptions: [
          ColorOption(
            id: 'raw_concrete_matte_black',
            title: 'Raw Concrete & Matte Black',
            colors: const [Color(0xFF9C9C97), Color(0xFF2A2A2A)],
          ),
          ColorOption(
            id: 'rusty_brick_leather',
            title: 'Rusty Brick & Leather',
            colors: const [Color(0xFFA35A3A), Color(0xFF5A3825)],
          ),
          ColorOption(
            id: 'dark_navy_blue',
            title: 'Dark Navy Blue',
            colors: const [Color(0xFF1E2A4A), Color(0xFF10182C)],
          ),
        ],
      ),

      DesignStyle(
        id: 'modern_luxury',
        title: 'Modern Luxury',
        subtitle: 'Sleek & Upscale',
        colorOptions: [
          ColorOption(
            id: 'ivory_champagne_gold',
            title: 'Ivory & Champagne Gold',
            colors: const [Color(0xFFF6F1E4), Color(0xFFD8B47E)],
          ),
          ColorOption(
            id: 'emerald_green_brass',
            title: 'Emerald Green & Brass',
            colors: const [Color(0xFF0E6E4E), Color(0xFFB68A4E)],
          ),
          ColorOption(
            id: 'midnight_blue_silver',
            title: 'Midnight Blue & Silver',
            colors: const [Color(0xFF0C1E3E), Color(0xFFC3C6CC)],
          ),
        ],
      ),
    ];

    final appState = ref.watch(appStateProvider);

    final selectedStyle = appState.selectedStyle;

    final canGenerate = appState.canGenerateDesign;

    return Scaffold(
      backgroundColor: AppColors.background,

      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),

          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,

              children: [
                const SizedBox(height: 16),

                ///////////////  App Title + AI Model Picker
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Text.rich(
                        TextSpan(
                          children: [
                            TextSpan(
                              text: 'Smart ',
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            TextSpan(
                              text: 'Decorator',
                              style: TextStyle(fontWeight: FontWeight.normal),
                            ),
                          ],
                        ),
                        style: TextStyle(
                          fontSize: 34,
                          height: 1.2,
                          color: AppColors.ink,
                        ),
                      ),
                    ),
                    const SizedBox(width: 10),
                    Padding(
                      padding: const EdgeInsets.only(top: 8),
                      child: AiModelPicker(
                        selected: appState.selectedAiModel,
                        onSelected: (model) {
                          ref
                              .read(appStateProvider.notifier)
                              .selectAiModel(model);
                        },
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 6),

                ///////////////  App Subtitle
                const Text(
                  'Scan to Visualize & Shop Your Dream Room',
                  style: TextStyle(
                    fontSize: 14,
                    color: AppColors.muted,
                    height: 1.4,
                  ),
                ),

                const SizedBox(height: 28),

                ///////////////  Room Type Options
                const _SectionLabel('Choose a Room Type'),

                const SizedBox(height: 12),

                SizedBox(
                  height: 112,
                  child: Scrollbar(
                    controller: _roomTypeScrollController,
                    thumbVisibility: true,
                    trackVisibility: false,
                    thickness: 4,
                    radius: const Radius.circular(4),
                    child: ListView.separated(
                      controller: _roomTypeScrollController,
                      padding: const EdgeInsets.only(bottom: 14),
                      scrollDirection: Axis.horizontal,
                      itemCount: roomTypes.length,
                      separatorBuilder: (context, _) =>
                          const SizedBox(width: 10),
                      itemBuilder: (context, index) {
                        final room = roomTypes[index];
                        final selected =
                            appState.selectedRoomType?.id == room.id;

                        return GestureDetector(
                          onTap: () {
                            ref
                                .read(appStateProvider.notifier)
                                .selectRoomType(room);
                          },

                          child: AnimatedContainer(
                            width: 84,
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            duration: const Duration(milliseconds: 180),

                            decoration: BoxDecoration(
                              color: AppColors.surface,
                              borderRadius: BorderRadius.circular(16),

                              boxShadow: selected
                                  ? [
                                      BoxShadow(
                                        color: AppColors.ink.withValues(
                                          alpha: 0.08,
                                        ),
                                        blurRadius: 14,
                                        offset: const Offset(0, 6),
                                      ),
                                    ]
                                  : [],

                              border: Border.all(
                                color: selected
                                    ? AppColors.sage
                                    : Colors.transparent,
                                width: 1.5,
                              ),
                            ),

                            child: Column(
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Container(
                                  width: 40,
                                  height: 40,
                                  decoration: BoxDecoration(
                                    color: selected
                                        ? AppColors.sageTint
                                        : AppColors.sandTint,
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Icon(
                                    room.icon,
                                    size: 20,
                                    color: selected
                                        ? AppColors.sageDeep
                                        : AppColors.brassDeep,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  room.title,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: selected
                                        ? AppColors.ink
                                        : AppColors.muted,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ),

                const SizedBox(height: 26),

                ///////////////  Design Style Options
                const _SectionLabel('Choose a Style'),

                const SizedBox(height: 12),

                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: styles.map((style) {
                    final selected = selectedStyle?.id == style.id;

                    return Expanded(
                      child: Padding(
                        padding: EdgeInsets.only(
                          right: style == styles.last ? 0 : 10,
                        ),
                        child: GestureDetector(
                          onTap: () {
                            ref
                                .read(appStateProvider.notifier)
                                .selectStyle(style);
                          },

                          child: AnimatedContainer(
                            padding: const EdgeInsets.symmetric(
                              vertical: 14,
                              horizontal: 6,
                            ),
                            duration: const Duration(milliseconds: 180),

                            decoration: BoxDecoration(
                              color: AppColors.surface,
                              borderRadius: BorderRadius.circular(16),

                              boxShadow: selected
                                  ? [
                                      BoxShadow(
                                        color: AppColors.ink.withValues(
                                          alpha: 0.08,
                                        ),
                                        blurRadius: 14,
                                        offset: const Offset(0, 6),
                                      ),
                                    ]
                                  : [],

                              border: Border.all(
                                color: selected
                                    ? AppColors.sage
                                    : Colors.transparent,
                                width: 1.5,
                              ),
                            ),

                            child: Column(
                              children: [
                                Container(
                                  width: 40,
                                  height: 40,
                                  decoration: BoxDecoration(
                                    color: selected
                                        ? AppColors.sageTint
                                        : AppColors.sandTint,
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Icon(
                                    _styleIcon(style.id),
                                    size: 20,
                                    color: selected
                                        ? AppColors.sageDeep
                                        : AppColors.brassDeep,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  style.title,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: selected
                                        ? AppColors.ink
                                        : AppColors.muted,
                                  ),
                                ),
                                const SizedBox(height: 2),
                                Text(
                                  style.subtitle,
                                  textAlign: TextAlign.center,
                                  style: const TextStyle(
                                    fontSize: 9.5,
                                    color: AppColors.muted,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    );
                  }).toList(),
                ),

                const SizedBox(height: 26),

                ///////////////  Color Options (depend on the selected style)
                const _SectionLabel('Choose a Color'),

                const SizedBox(height: 12),

                if (selectedStyle == null)
                  Text(
                    'Pick a style above to see its color options',
                    style: TextStyle(fontSize: 12.5, color: AppColors.muted),
                  )
                else
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: selectedStyle.colorOptions.map((colorOption) {
                      final selected =
                          appState.selectedColorOption?.id == colorOption.id;

                      return Padding(
                        padding: const EdgeInsets.only(right: 16),
                        child: GestureDetector(
                          onTap: () {
                            ref
                                .read(appStateProvider.notifier)
                                .selectColorOption(colorOption);
                          },

                          child: Column(
                            children: [
                              AnimatedContainer(
                                duration: const Duration(milliseconds: 180),
                                width: 46,
                                height: 46,
                                decoration: BoxDecoration(
                                  shape: BoxShape.circle,
                                  gradient: LinearGradient(
                                    begin: Alignment.topLeft,
                                    end: Alignment.bottomRight,
                                    colors: colorOption.colors,
                                  ),
                                  border: Border.all(
                                    color: selected
                                        ? AppColors.sage
                                        : Colors.transparent,
                                    width: 2.5,
                                  ),
                                ),
                                child: selected
                                    ? Icon(
                                        Icons.check_rounded,
                                        color: _iconColorFor(
                                          colorOption.colors.first,
                                        ),
                                        size: 20,
                                      )
                                    : null,
                              ),
                              const SizedBox(height: 6),
                              SizedBox(
                                width: 72,
                                child: Text(
                                  colorOption.title,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 10.5,
                                    fontWeight: FontWeight.w500,
                                    color: selected
                                        ? AppColors.ink
                                        : AppColors.muted,
                                  ),
                                ),
                              ),
                            ],
                          ),
                        ),
                      );
                    }).toList(),
                  ),

                const SizedBox(height: 26),

                ///////////////  Upload Room Photo Section
                const _SectionLabel('Upload Current Room Photo'),

                const SizedBox(height: 12),

                if (appState.uploadedImage == null)
                  ///////////////  Upload Button
                  GestureDetector(
                    onTap: () {
                      pickImage(ref);
                    },

                    child: Container(
                      width: double.infinity,

                      padding: const EdgeInsets.symmetric(
                        horizontal: 18,
                        vertical: 18,
                      ),

                      decoration: BoxDecoration(
                        color: AppColors.surface,
                        borderRadius: BorderRadius.circular(16),
                        border: Border.all(
                          color: AppColors.muted.withValues(alpha: 0.35),
                          width: 1.4,
                          style: BorderStyle.solid,
                        ),
                      ),

                      child: Row(
                        children: [
                          Container(
                            width: 36,
                            height: 36,
                            decoration: BoxDecoration(
                              color: AppColors.sageTint,
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: const Icon(
                              Icons.camera_alt_rounded,
                              size: 18,
                              color: AppColors.sageDeep,
                            ),
                          ),
                          const SizedBox(width: 12),
                          const Text(
                            'Upload Photo',
                            style: TextStyle(
                              fontSize: 14,
                              fontWeight: FontWeight.w600,
                              color: AppColors.ink,
                            ),
                          ),
                        ],
                      ),
                    ),
                  )
                else
                  ClipRRect(
                    borderRadius: BorderRadius.circular(18),
                    child: Stack(
                      children: [
                        Image.file(
                          appState.uploadedImage!,
                          height: 150,
                          width: double.infinity,
                          fit: BoxFit.cover,
                        ),
                        Positioned(
                          top: 10,
                          right: 10,
                          child: Material(
                            color: AppColors.ink.withValues(alpha: 0.55),
                            shape: const CircleBorder(),
                            child: IconButton(
                              tooltip: 'Remove photo',
                              iconSize: 18,
                              icon: const Icon(
                                Icons.close_rounded,
                                color: Colors.white,
                              ),
                              onPressed: () {
                                ref
                                    .read(appStateProvider.notifier)
                                    .clearUploadedImage();
                              },
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),

                const SizedBox(height: 18),

                ///////////////  LiDAR Scan Button Card
                Material(
                  color: AppColors.sageDeep,
                  borderRadius: BorderRadius.circular(20),
                  clipBehavior: Clip.antiAlias,

                  child: InkWell(
                    onTap: () async {
                      if (Platform.isAndroid) {
                        final proceed = await showDialog<bool>(
                          context: context,
                          builder: (ctx) => AlertDialog(
                            title: const Text('Note: Lower Accuracy'),
                            content: const Text(
                              'Android devices do not have a LiDAR sensor.\n\n'
                              'Room scan accuracy will be lower than on an iPhone with LiDAR. '
                              'Tap each room corner as precisely as possible for the best results.',
                            ),
                            actions: [
                              TextButton(
                                onPressed: () => Navigator.pop(ctx, false),
                                child: const Text('Cancel'),
                              ),
                              ElevatedButton(
                                onPressed: () => Navigator.pop(ctx, true),
                                child: const Text('Got it, continue'),
                              ),
                            ],
                          ),
                        );
                        if (proceed != true || !context.mounted) return;
                      }
                      if (context.mounted) context.push('/scan_room');
                    },

                    child: Padding(
                      padding: const EdgeInsets.all(18),
                      child: Row(
                        crossAxisAlignment: CrossAxisAlignment.center,
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                const Text(
                                  'Scan Room\nwith LiDAR',
                                  style: TextStyle(
                                    fontSize: 17,
                                    fontWeight: FontWeight.bold,
                                    color: Colors.white,
                                  ),
                                ),

                                const SizedBox(height: 6),

                                Text(
                                  'Precise 3D Room Scan with suitable iOS device',
                                  style: TextStyle(
                                    fontSize: 12,
                                    color: Colors.white.withValues(alpha: 0.75),
                                  ),
                                ),
                              ],
                            ),
                          ),

                          const SizedBox(width: 12),

                          LidarScanGraphic(
                            size: 64,
                            bracketColor: AppColors.brass,
                          ),
                        ],
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 26),

                ///////////////  Generate Design Button
                SizedBox(
                  width: double.infinity,
                  height: 56,

                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: canGenerate
                          ? AppColors.brass
                          : AppColors.brass.withValues(alpha: 0.45),
                      foregroundColor: AppColors.ink,
                      elevation: 0,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(16),
                      ),
                    ),

                    onPressed: canGenerate
                        ? () {
                            debugPrint('Generate Design Tapped');
                            context.go('/processing');
                            // context.go('/test');
                          }
                        : null,

                    child: const Text(
                      'Generate Design',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),

                if (!canGenerate) ...[
                  const SizedBox(height: 10),

                  const Center(
                    child: Text(
                      'Please select room type, style, color, and uplaod your room image',
                      textAlign: TextAlign.center,
                      style: TextStyle(fontSize: 12, color: AppColors.muted),
                    ),
                  ),
                ],
                const SizedBox(height: 36),
              ],
            ),
          ),
        ),
      ),

      bottomNavigationBar: const AppFooterNav(current: FooterTab.home),
    );
  }
}

IconData _styleIcon(String styleId) {
  switch (styleId) {
    case 'industrial_loft':
      return Icons.factory_rounded;
    case 'modern_luxury':
      return Icons.diamond_rounded;
    case 'japandi':
    default:
      return Icons.spa_rounded;
  }
}

/// Picks a legible checkmark color against a swatch's first color, since
/// the color options range from very light (ivory) to very dark (navy).
Color _iconColorFor(Color background) {
  return background.computeLuminance() > 0.55 ? AppColors.ink : Colors.white;
}

class _SectionLabel extends StatelessWidget {
  final String text;
  const _SectionLabel(this.text);

  @override
  Widget build(BuildContext context) {
    return Text(
      text.toUpperCase(),
      style: const TextStyle(
        fontSize: 12,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.6,
        color: AppColors.ink,
      ),
    );
  }
}
