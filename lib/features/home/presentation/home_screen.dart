import 'dart:io' show File, Platform;
import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:go_router/go_router.dart';
import 'package:image_picker/image_picker.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/widgets/app_icons.dart';
import '../../../shared/providers/app_state_provider.dart';
import '../../../shared/models/design_theme.dart';
import '../../../shared/models/room_type.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

  Future<void> pickImage(
    WidgetRef ref,
  ) async {
    final picker = ImagePicker();

    final pickedFile =
      await picker.pickImage(
        source: ImageSource.gallery,
      );
    if (pickedFile != null) {
      ref
        .read(appStateProvider.notifier)
        .setUploadedImage(
          File(pickedFile.path),
        );
    }
  }

  @override
  Widget build(BuildContext context, WidgetRef ref) {

    // Sample Room Types
    final roomTypes = [
      RoomType(
        id: 'living_room',
        title: 'Living Room',
        icon: Icons.weekend_rounded,
      ),

      RoomType(
        id: 'bedroom',
        title: 'Bedroom',
        icon: Icons.bed_rounded,
      ),

      RoomType(
        id: 'dining_room',
        title: 'Dining Room',
        icon: Icons.table_restaurant_rounded,
      ),
    ];

    // Sample Theme Colors
    final themes = [
      DesignTheme(
        id: 'minimal',
        title: 'Minimal',
        color: AppColors.sand,
      ),

      DesignTheme(
        id: 'modern',
        title: 'Modern',
        color: AppColors.sage,
      ),

      DesignTheme(
        id: 'luxury',
        title: 'Luxury',
        color: AppColors.brassDeep,
      ),
    ];

    final appState = ref.watch(appStateProvider);

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

                ///////////////  Eyebrow
                Row(
                  children: [
                    const Icon(Icons.home_rounded, size: 16, color: AppColors.sage),
                    const SizedBox(width: 6),
                    Text(
                      'SMART DECORATOR',
                      style: TextStyle(
                        fontSize: 11,
                        fontWeight: FontWeight.w600,
                        letterSpacing: 1.1,
                        color: AppColors.sage,
                      ),
                    ),
                  ],
                ),

                const SizedBox(height: 6),

                ///////////////  App Title
                const Text(
                  'Scan, visualise,\nshop your room',
                  style: TextStyle(
                    fontSize: 28,
                    height: 1.25,
                    fontWeight: FontWeight.bold,
                    color: AppColors.ink,
                  ),
                ),

                const SizedBox(height: 6),

                ///////////////  App Subtitle
                const Text(
                  'Real dimensions in, a finished room out.',
                  style: TextStyle(
                    fontSize: 14,
                    color: AppColors.muted,
                    height: 1.4,
                  ),
                ),

                const SizedBox(height: 28),

                ///////////////  Room Type Options
                const _SectionLabel('Room type'),

                const SizedBox(height: 12),

                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,

                  children: roomTypes.map((room) {
                    final selected = appState.selectedRoomType?.id == room.id;

                    return Expanded(
                      child: Padding(
                        padding: EdgeInsets.only(
                          right: room == roomTypes.last ? 0 : 10,
                        ),
                        child: GestureDetector(
                          onTap: () {
                            ref
                              .read(appStateProvider.notifier)
                              .selectRoomType(room);
                          },

                          child: AnimatedContainer(
                            padding: const EdgeInsets.symmetric(vertical: 14),
                            duration: const Duration(milliseconds: 180),

                            decoration: BoxDecoration(
                              color: AppColors.surface,
                              borderRadius: BorderRadius.circular(16),

                              boxShadow: selected
                                  ? [
                                      BoxShadow(
                                        color: AppColors.ink.withValues(alpha: 0.08),
                                        blurRadius: 14,
                                        offset: const Offset(0, 6),
                                      ),
                                    ]
                                  : [],

                              border: Border.all(
                                color: selected ? AppColors.sage : Colors.transparent,
                                width: 1.5,
                              ),
                            ),

                            child: Column(
                              children: [
                                Container(
                                  width: 40,
                                  height: 40,
                                  decoration: BoxDecoration(
                                    color: selected ? AppColors.sageTint : AppColors.sandTint,
                                    borderRadius: BorderRadius.circular(12),
                                  ),
                                  child: Icon(
                                    room.icon,
                                    size: 20,
                                    color: selected ? AppColors.sageDeep : AppColors.brassDeep,
                                  ),
                                ),
                                const SizedBox(height: 8),
                                Text(
                                  room.title,
                                  textAlign: TextAlign.center,
                                  style: TextStyle(
                                    fontSize: 11,
                                    fontWeight: FontWeight.w600,
                                    color: selected ? AppColors.ink : AppColors.muted,
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

                ///////////////  Theme Color Options
                const _SectionLabel('Theme'),

                const SizedBox(height: 12),

                Row(
                  children: themes.map((theme) {
                    final selected = appState.selectedTheme?.id == theme.id;

                    return Padding(
                      padding: const EdgeInsets.only(right: 16),
                      child: GestureDetector(
                        onTap: () {
                          ref
                            .read(appStateProvider.notifier)
                            .selectTheme(theme);
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
                                  colors: [
                                    theme.color,
                                    Color.lerp(theme.color, Colors.black, 0.18)!,
                                  ],
                                ),
                                border: Border.all(
                                  color: selected ? AppColors.sage : Colors.transparent,
                                  width: 2.5,
                                ),
                              ),
                              child: selected
                                  ? const Icon(Icons.check_rounded, color: Colors.white, size: 20)
                                  : null,
                            ),
                            const SizedBox(height: 6),
                            Text(
                              theme.title,
                              style: TextStyle(
                                fontSize: 11,
                                fontWeight: FontWeight.w500,
                                color: selected ? AppColors.ink : AppColors.muted,
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
                const _SectionLabel('Current room photo'),

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
                            'Upload photo',
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

                    child: Stack(
                      children: [
                        Positioned(
                          right: -18,
                          top: -18,
                          child: Container(
                            width: 90,
                            height: 90,
                            decoration: BoxDecoration(
                              shape: BoxShape.circle,
                              color: Colors.white.withValues(alpha: 0.06),
                            ),
                          ),
                        ),

                        Padding(
                          padding: const EdgeInsets.fromLTRB(18, 18, 18, 4),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,

                            children: [
                              Row(
                                children: [
                                  const ScanBracketIcon(size: 18, color: AppColors.brass),
                                  const SizedBox(width: 8),
                                  const Text(
                                    'Scan room with LiDAR',
                                    style: TextStyle(
                                      fontSize: 15,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.white,
                                    ),
                                  ),
                                ],
                              ),

                              const SizedBox(height: 4),

                              Text(
                                'Tap corners, get exact room size',
                                style: TextStyle(
                                  fontSize: 12,
                                  color: Colors.white.withValues(alpha: 0.75),
                                ),
                              ),

                              Align(
                                alignment: Alignment.centerRight,
                                child: LidarHandIllustration(
                                  width: 200,
                                  height: 78,
                                  bracketColor: AppColors.brass,
                                ),
                              ),
                            ],
                          ),
                        ),
                      ],
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
                      backgroundColor:
                        canGenerate
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
                      'Generate design',
                      style: TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),

                if(!canGenerate) ...[
                  const SizedBox(height: 10),

                  const Center(
                    child: Text(
                      'Select a room type, theme, and photo to continue',
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        fontSize: 12,
                        color: AppColors.muted,
                      ),
                    )
                  )
                ],
                const SizedBox(height: 36),
              ],
            ),
          ),
        ),
      ),

      bottomNavigationBar: Container(
        decoration: BoxDecoration(
          color: AppColors.surface,
          border: Border(
            top: BorderSide(color: AppColors.ink.withValues(alpha: 0.06)),
          ),
        ),
        child: BottomNavigationBar(
          currentIndex: 0,
          backgroundColor: Colors.transparent,
          elevation: 0,
          onTap: (index) {
            if (index == 1) {
              context.go('/history');
            }
          },

          selectedItemColor: AppColors.sageDeep,
          unselectedItemColor: AppColors.muted,

          items: const [
            BottomNavigationBarItem(
              icon: Icon(
                Icons.home_rounded,
              ),
              label: 'Home',
            ),

            BottomNavigationBarItem(
              icon: Icon(
                Icons.history_rounded,
              ),
              label: 'History',
            ),
          ],
        ),
      ),
    );
  }
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
