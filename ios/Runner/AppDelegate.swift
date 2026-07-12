import Flutter
import UIKit
import ARKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    
    // 1. ขอตัวส่งสารจาก Flutter โดยตรงผ่าน Registrar (ไม่ต้องผ่าน window)
    if let pluginRegistrar = self.registrar(forPlugin: "SmartDecoratorPlugin") {
        
        let messenger = pluginRegistrar.messenger()
        
        // 2. สร้างสะพาน MethodChannel สำหรับเช็ค Lidar
        let sensorChannel = FlutterMethodChannel(name: "com.smartdeco.app/sensors", binaryMessenger: messenger)
        
        sensorChannel.setMethodCallHandler({
          (call: FlutterMethodCall, result: @escaping FlutterResult) -> Void in
          
          if call.method == "checkLidar" {
            if #available(iOS 13.4, *) {
                // เช็คว่าเครื่องรองรับ Lidar ไหม
                let hasLidar = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
                result(hasLidar)
            } else {
                result(false)
            }
          } else {
            result(FlutterMethodNotImplemented)
          }
        })
        
        // 3. ลงทะเบียนหน้าจอ AR Camera สำหรับฝั่ง Dart
        let arFactory = ARCameraFactory(messenger: messenger)
        pluginRegistrar.register(arFactory, withId: "ar_camera_view")
    }
    
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
  }
}