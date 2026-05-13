import Flutter
import UIKit
import ARKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    let controller : FlutterViewController = window?.rootViewController as! FlutterViewController
    let sensorChannel = FlutterMethodChannel(name: "com.smartdeco.app/sensors", binaryMessenger: controller.binaryMessenger)
    
    sensorChannel.setMethodCallHandler({
      (call: FlutterMethodCall, result: @escaping FlutterResult) -> Void in
      
      if call.method == "checkLidar" {
        if #available(iOS 13.4, *) {
            let hasLidar = ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
            result(hasLidar)
        } else {
            result(false)
        }
      } else {
        result(FlutterMethodNotImplemented)
      }
    })
    
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
  }
}
