import 'package:flutter/material.dart';

import '../../../core/constants/app_colors.dart';
import '../../../core/services/hardware_service.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {

    final roomTypes = [
      'Living Room',
      'Bedroom',
      'Dining Room',
    ];

    final themeColors = [
      AppColors.lightCard,
      AppColors.beige,
      AppColors.primary,
    ];

    return Scaffold(

      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 20),

          child: SingleChildScrollView(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,

              children: [

                const SizedBox(height: 20),

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
                    fontSize: 40,
                    color: AppColors.darkText,
                  ),
                ),

                const SizedBox(height: 8),

                const Text(
                  'Scan to Visualize & Shop Your Dream Room',
                  style: TextStyle(
                    fontSize: 18,
                    color: Colors.black54,
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

                    return Container(
                      width: 100,
                      padding: const EdgeInsets.all(12),

                      decoration: BoxDecoration(
                        color: AppColors.lightCard,
                        borderRadius: BorderRadius.circular(20),
                      ),

                      child: Column(
                        children: [

                          const Icon(
                            Icons.chair_alt_rounded,
                            size: 40,
                          ),

                          const SizedBox(height: 10),

                          Text(
                            room,
                            textAlign: TextAlign.center,
                          ),
                        ],
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

                  children: themeColors.map((color) {

                    return Container(
                      width: 100,
                      height: 100,

                      decoration: BoxDecoration(
                        color: color,
                        borderRadius: BorderRadius.circular(20),
                      ),

                      child: const Center(
                        child: Text(
                          'Theme',
                          style: TextStyle(
                            color: Colors.white,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ),
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