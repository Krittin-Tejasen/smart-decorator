import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/hardware_service.dart';

import '../../../shared/models/app_state_provider.dart';
import '../../../shared/models/design_theme.dart';
import '../../../shared/models/room_type.dart';

class HomeScreen extends ConsumerWidget {
  const HomeScreen({super.key});

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
        color: AppColors.lightCard,
      ),

      DesignTheme(
        id: 'modern',
        title: 'Modern',
        color: AppColors.beige,
      ),

      DesignTheme(
        id: 'luxury',
        title: 'Luxury',
        color: AppColors.primary,
      ),
    ];

    final appState = ref.watch(appStateProvider);

    return Scaffold(

      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),

          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,

              children: [

                const SizedBox(height: 20),

                ///////////////  App Title
                Text.rich(
                  TextSpan(
                    children: [
                      TextSpan(
                        text: 'Smart ',
                        style: TextStyle(
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      TextSpan(
                        text: 'Decorator',
                        style: TextStyle(
                          fontWeight: FontWeight.normal,
                        ),
                      ),
                    ],
                  ),
                  style: TextStyle(
                    fontSize: 42,
                    height: 1.2,
                    color: AppColors.darkText,
                  ),
                ),

                const SizedBox(height: 8),

                ///////////////  App Subtitle
                const Text(
                  'Scan to Visualize & Shop Your Dream Room',
                  style: TextStyle(
                    fontSize: 18,
                    color: Colors.black54,
                    height: 1.4,
                  ),
                ),

                const SizedBox(height: 30),

                ///////////////  Room Type Options
                const Text(
                  'Choose a Room Type',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),

                const SizedBox(height: 16),

                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,

                  children: roomTypes.map((room) {
                    return GestureDetector(
                      onTap: () {
                        ref
                          .read(appStateProvider.notifier)
                          .selectRoomType(room);
                      },

                      child: AnimatedContainer(
                        width: 100,
                        padding: const EdgeInsets.all(12),
                        duration: Duration(milliseconds: 180),

                        decoration: BoxDecoration(
                          color:
                            appState.selectedRoomType?.id == room.id
                              ? Colors.white
                              : AppColors.lightCard,
                          // color: AppColors.lightCard,

                          borderRadius: BorderRadius.circular(20),

                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withValues(alpha: 0.04),
                              blurRadius: 12,
                              offset: Offset(0, 4),
                            )
                          ],

                          border: Border.all(
                            color:
                              appState.selectedRoomType?.id == room.id
                                  ? AppColors.primary
                                  : Colors.transparent,

                            width: 3,
                          ),
                        ),

                        child: Column(
                          children: [

                            Icon(
                              room.icon,
                              size: 40,
                            ),

                            const SizedBox(height: 10),

                            Text(
                              room.title,
                              textAlign: TextAlign.center,
                            ),
                          ],
                        ),
                      ),
                    );
                  }).toList(),
                ),

                const SizedBox(height: 30),

                ///////////////  Theme Color Options
                const Text(
                  'Choose Theme',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),

                const SizedBox(height: 16),

                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,

                  children: themes.map((theme) {

                    return GestureDetector(
                      onTap: () {
                        ref
                          .read(appStateProvider.notifier)
                          .selectTheme(theme);
                      },

                      child: AnimatedContainer(
                        width: 100,
                        height: 100,
                        padding: const EdgeInsets.all(12),
                        duration: Duration(milliseconds: 180),

                        decoration: BoxDecoration(
                          color: theme.color,
                          borderRadius: BorderRadius.circular(20),
                          
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withValues(alpha: 0.04),
                              blurRadius: 12,
                              offset: Offset(0, 4),
                            )
                          ],

                          border: Border.all(
                            color: appState.selectedTheme?.id == theme.id
                                ? AppColors.primary
                                : Colors.transparent,
                            width: 3,
                          ),
                        ),

                        child: Column(
                          children: [
                            Icon(
                              Icons.palette_rounded,
                              size: 40,
                              color: Colors.white,
                            ),

                            const SizedBox(height: 10),

                            Text(
                              theme.title,
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                color: Colors.white,
                              ),
                            ),
                          ],
                        )
                      )
                    );
                  }).toList(),
                ),


                const SizedBox(height: 30),

                ///////////////  Upload Room Photo Section
                const Text(
                  'Upload Current Room Photo',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),

                const SizedBox(height: 16),

                ///////////////  Upload Button
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.symmetric(
                    horizontal: 20,
                    vertical: 18,
                  ),

                  decoration: BoxDecoration(
                    color: AppColors.lightCard,
                    borderRadius: BorderRadius.circular(18),
                  ),

                  child: const Row(
                    children: [

                      Icon(
                        Icons.camera_alt_rounded,
                        color: AppColors.primary,
                      ),

                      SizedBox(width: 12),

                      Text(
                        'Upload Photo',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ],
                  ),
                ),

                const SizedBox(height: 20),

                ///////////////  LiDAR Scan Button Card
                Material(
                  color: const Color(0xFFDDE4E1),
                  borderRadius: BorderRadius.circular(20),

                  child: InkWell(
                    borderRadius: BorderRadius.circular(20),

                    onTap: () {
                      debugPrint('LiDAR Scan Tapped');
                      HardwareService.hasLidarSensor().then((hasLidar) {
                        if (hasLidar) {
                          debugPrint('Device has LiDAR - Proceed to LiDAR Scan');
                        } else {
                          debugPrint('No LiDAR detected - Consider AR Vision Scan');
                        }
                      });
                    },

                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(20),

                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,

                        children: [

                          const Row(
                            children: [

                              Icon(
                                Icons.wifi_tethering_rounded,
                                size: 30,
                              ),

                              SizedBox(width: 12),

                              Text(
                                'Scan Room\nwith LiDAR',
                                style: TextStyle(
                                  fontSize: 24,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ],
                          ),

                          const SizedBox(height: 12),

                          Text(
                            'Precise 3D Room Scan with suitable iOS device',
                            style: TextStyle(
                              color: Colors.grey.shade700,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                ),

                const SizedBox(height: 30),

                ///////////////  Generate Design Button
                SizedBox(
                  width: double.infinity,
                  height: 60,

                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: AppColors.primary,
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(18),
                      ),
                    ),

                    onPressed: () {
                      debugPrint('Generate Design Tapped');
                    },

                    child: const Text(
                      'Generate Design',
                      style: TextStyle(
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 40),
              ],
            ),
          ),
        ),
      ),

      bottomNavigationBar: BottomNavigationBar(
        currentIndex: 0,

        selectedItemColor: AppColors.primary,

        items: const [

          BottomNavigationBarItem(
            icon: Icon(Icons.home_rounded),
            label: 'Home',
          ),

          BottomNavigationBarItem(
            icon: Icon(Icons.history_rounded),
            label: 'History',
          ),
        ],
      ),
    );
  }
}