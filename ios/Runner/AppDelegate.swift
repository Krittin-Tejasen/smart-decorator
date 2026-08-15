import Flutter
import UIKit
import ARKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {

  // Holds the pending Flutter result while a RoomPlan scan is in progress
  private var pendingRoomPlanResult: FlutterResult?

  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {

    if let pluginRegistrar = self.registrar(forPlugin: "SmartDecoratorPlugin") {
        let messenger = pluginRegistrar.messenger()

        // ── Sensor channel (LiDAR check) ──────────────────────────────────
        let sensorChannel = FlutterMethodChannel(
            name: "com.smartdeco.app/sensors",
            binaryMessenger: messenger
        )
        sensorChannel.setMethodCallHandler { (call, result) in
            if call.method == "checkLidar" {
                if #available(iOS 13.4, *) {
                    result(ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh))
                } else {
                    result(false)
                }
            } else {
                result(FlutterMethodNotImplemented)
            }
        }

        // ── Room-tools channel ─────────────────────────────────────────────
        // Provides: measureFurniture, startRoomPlanScan,
        //           checkCollision, renderDepthMap
        let toolsChannel = FlutterMethodChannel(
            name: "com.smartdeco.app/room_tools",
            binaryMessenger: messenger
        )
        toolsChannel.setMethodCallHandler { [weak self] (call, result) in
            self?.handleRoomTools(call: call, result: result)
        }

        // ── AR Camera platform view ────────────────────────────────────────
        pluginRegistrar.register(ARCameraFactory(messenger: messenger),
                                 withId: "ar_camera_view")
    }

    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }

  func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
    GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
  }

  // MARK: - room_tools handler

  private func handleRoomTools(call: FlutterMethodCall, result: @escaping FlutterResult) {
    switch call.method {

    // ── measureFurniture ──────────────────────────────────────────────────
    // Args: { "xMin": Float, "yMin": Float, "xMax": Float, "yMax": Float,
    //         "snapshotJson": String }
    // Returns: { "widthMeters": Float, "heightMeters": Float, "depthMeters": Float }
    case "measureFurniture":
        guard
            let args = call.arguments as? [String: Any],
            let xMin = args["xMin"] as? Double,
            let yMin = args["yMin"] as? Double,
            let xMax = args["xMax"] as? Double,
            let yMax = args["yMax"] as? Double,
            let snapshotJson = args["snapshotJson"] as? String,
            let jsonData = snapshotJson.data(using: .utf8),
            let state = try? RoomCaptureState.from(jsonData: jsonData)
        else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "measureFurniture requires xMin/yMin/xMax/yMax and snapshotJson",
                                details: nil))
            return
        }
        let bbox = BBox2D(xMin: Float(xMin), yMin: Float(yMin),
                          xMax: Float(xMax), yMax: Float(yMax))
        if let size = SizeCalculator.measure(bbox: bbox, in: state) {
            result([
                "widthMeters":  size.widthMeters,
                "heightMeters": size.heightMeters,
                "depthMeters":  size.depthMeters,
            ])
        } else {
            result(FlutterError(code: "NO_FLOOR",
                                message: "No floor plane detected in snapshot — scan more floor area first",
                                details: nil))
        }

    // ── checkCollision ────────────────────────────────────────────────────
    // Args: { "newBox": DetectedBox3D map,
    //         "existingBoxes": [DetectedBox3D map] }
    // Returns: Bool
    case "checkCollision":
        guard
            let args = call.arguments as? [String: Any],
            let newBoxMap = args["newBox"] as? [String: Any],
            let existingMaps = args["existingBoxes"] as? [[String: Any]],
            let newBox  = decodeBox(newBoxMap),
            let existing = try? existingMaps.map({ map -> DetectedBox3D in
                guard let b = decodeBox(map) else {
                    throw NSError(domain: "decode", code: 0)
                }
                return b
            })
        else {
            result(FlutterError(code: "BAD_ARGS", message: "Invalid box data", details: nil))
            return
        }
        result(CollisionChecker.collides(newBox, with: existing))

    // ── renderDepthMap ────────────────────────────────────────────────────
    // Args: { "roomPlanJson": String, "snapshotJson": String,
    //         "width": Int (optional, default 512),
    //         "height": Int (optional, default 512) }
    // Returns: String — "data:image/png;base64,…"
    case "renderDepthMap":
        guard
            let args = call.arguments as? [String: Any],
            let roomPlanJson  = args["roomPlanJson"]  as? String,
            let snapshotJson  = args["snapshotJson"]  as? String,
            let roomData      = roomPlanJson.data(using: .utf8),
            let snapData      = snapshotJson.data(using: .utf8),
            let roomPlan      = try? JSONDecoder().decode(RoomPlanResult.self,    from: roomData),
            let captureState  = try? JSONDecoder().decode(RoomCaptureState.self,  from: snapData)
        else {
            result(FlutterError(code: "BAD_ARGS",
                                message: "renderDepthMap requires roomPlanJson and snapshotJson",
                                details: nil))
            return
        }
        let w = args["width"]  as? Int ?? 512
        let h = args["height"] as? Int ?? 512
        let allBoxes = roomPlan.objects + roomPlan.walls
        guard let img = DepthMapRenderer.render(
            boxes: allBoxes,
            captureState: captureState,
            outputSize: CGSize(width: w, height: h)
        ), let pngData = img.pngData() else {
            result(FlutterError(code: "RENDER_FAILED", message: "Depth map rendering failed", details: nil))
            return
        }
        result("data:image/png;base64," + pngData.base64EncodedString())

    // ── startRoomPlanScan ─────────────────────────────────────────────────
    // No args. Presents the RoomPlan UI modally.
    // Returns: RoomPlanResult map when the user taps "Done Scanning".
    case "startRoomPlanScan":
        #if canImport(RoomPlan)
        if #available(iOS 16.0, *) {
            guard RoomPlanScannerSession.isSupported else {
                result(FlutterError(code: "UNSUPPORTED",
                                    message: "This device does not support RoomPlan (LiDAR required)",
                                    details: nil))
                return
            }
            guard pendingRoomPlanResult == nil else {
                result(FlutterError(code: "SCAN_IN_PROGRESS",
                                    message: "A RoomPlan scan is already running",
                                    details: nil))
                return
            }
            pendingRoomPlanResult = result
            DispatchQueue.main.async { [weak self] in self?.presentRoomPlanVC() }
        } else {
            result(FlutterError(code: "UNSUPPORTED", message: "RoomPlan requires iOS 16+", details: nil))
        }
        #else
        result(FlutterError(code: "UNSUPPORTED", message: "RoomPlan framework not available", details: nil))
        #endif

    default:
        result(FlutterMethodNotImplemented)
    }
  }

  // MARK: - Helpers

  private func decodeBox(_ map: [String: Any]) -> DetectedBox3D? {
    guard
        let label = map["label"] as? String,
        let cx = map["cx"] as? Double, let cy = map["cy"] as? Double,
        let cz = map["cz"] as? Double, let hw = map["hw"] as? Double,
        let hh = map["hh"] as? Double, let hd = map["hd"] as? Double,
        let yaw = map["yaw"] as? Double
    else { return nil }
    return DetectedBox3D(
        label: label,
        cx: Float(cx), cy: Float(cy), cz: Float(cz),
        hw: Float(hw), hh: Float(hh), hd: Float(hd),
        yaw: Float(yaw)
    )
  }

  #if canImport(RoomPlan)
  @available(iOS 16.0, *)
  private func presentRoomPlanVC() {
    let vc = RoomPlanViewController()
    vc.modalPresentationStyle = .fullScreen
    vc.onComplete = { [weak self] scanResult in
        guard let self else { return }
        switch scanResult {
        case .success(let planResult):
            if let map = try? planResult.toFlutterMap() {
                self.pendingRoomPlanResult?(map)
            } else {
                self.pendingRoomPlanResult?(
                    FlutterError(code: "SERIALISE_FAILED",
                                 message: "Could not serialise RoomPlan result",
                                 details: nil)
                )
            }
        case .failure(let error):
            self.pendingRoomPlanResult?(
                FlutterError(code: "SCAN_FAILED",
                             message: error.localizedDescription,
                             details: nil)
            )
        }
        self.pendingRoomPlanResult = nil
    }

    // Present from the key window's root view controller
    let scene = UIApplication.shared.connectedScenes
        .compactMap { $0 as? UIWindowScene }
        .first
    let rootVC = scene?.windows.first(where: \.isKeyWindow)?.rootViewController
    rootVC?.present(vc, animated: true)
  }
  #endif
}