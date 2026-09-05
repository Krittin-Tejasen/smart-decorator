import Flutter
import UIKit
import ARKit

@main
@objc class AppDelegate: FlutterAppDelegate, FlutterImplicitEngineDelegate {

    private var pendingRoomPlanResult: FlutterResult?

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {

        if let reg = self.registrar(forPlugin: "SmartDecoratorPlugin") {
            let messenger = reg.messenger()

            // ── Sensors channel ───────────────────────────────────────────
            let sensorCh = FlutterMethodChannel(name: "com.smartdeco.app/sensors",
                                                binaryMessenger: messenger)
            sensorCh.setMethodCallHandler { call, result in
                switch call.method {
                case "checkLidar":
                    if #available(iOS 13.4, *) {
                        result(ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh))
                    } else { result(false) }
                case "getCapabilities":
                    result(ARCapabilityService.toFlutterMap())
                default:
                    result(FlutterMethodNotImplemented)
                }
            }

            // ── Room-tools channel ────────────────────────────────────────
            let toolsCh = FlutterMethodChannel(name: "com.smartdeco.app/room_tools",
                                               binaryMessenger: messenger)
            toolsCh.setMethodCallHandler { [weak self] call, result in
                self?.handleRoomTools(call: call, result: result)
            }

            // ── AR Camera platform view ───────────────────────────────────
            reg.register(ARCameraFactory(messenger: messenger), withId: "ar_camera_view")
        }

        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }

    func didInitializeImplicitFlutterEngine(_ engineBridge: FlutterImplicitEngineBridge) {
        GeneratedPluginRegistrant.register(with: engineBridge.pluginRegistry)
    }

    // MARK: - room_tools dispatch

    private func handleRoomTools(call: FlutterMethodCall, result: @escaping FlutterResult) {
        switch call.method {

        // measureFurniture ────────────────────────────────────────────────
        // Args: { xMin, yMin, xMax, yMax: Double, snapshotJson: String }
        // Returns: { widthMeters, heightMeters, depthMeters, method: String }
        case "measureFurniture":
            guard let a = call.arguments as? [String: Any],
                  let xMin = a["xMin"] as? Double, let yMin = a["yMin"] as? Double,
                  let xMax = a["xMax"] as? Double, let yMax = a["yMax"] as? Double,
                  let json = a["snapshotJson"] as? String,
                  let data = json.data(using: .utf8),
                  let state = try? RoomCaptureState.from(jsonData: data)
            else {
                result(FlutterError(code: "BAD_ARGS",
                                    message: "measureFurniture: missing or invalid args", details: nil))
                return
            }
            let bbox = BBox2D(xMin: Float(xMin), yMin: Float(yMin),
                              xMax: Float(xMax), yMax: Float(yMax))
            if let sz = SizeCalculator.measure(bbox: bbox, in: state) {
                result(["widthMeters": sz.widthMeters, "heightMeters": sz.heightMeters,
                        "depthMeters": sz.depthMeters, "method": sz.method])
            } else {
                result(FlutterError(code: "NO_REFERENCE",
                                    message: "No floor plane or depth data in snapshot", details: nil))
            }

        // checkCollision ──────────────────────────────────────────────────
        // Args: { newBox: Map, existingBoxes: [Map], snapshotJson: String? }
        // Returns: Bool
        case "checkCollision":
            guard let a = call.arguments as? [String: Any],
                  let newMap  = a["newBox"] as? [String: Any],
                  let exMaps  = a["existingBoxes"] as? [[String: Any]],
                  let newBox  = decodeBox(newMap)
            else {
                result(FlutterError(code: "BAD_ARGS", message: "Invalid box data", details: nil)); return
            }
            let existing = exMaps.compactMap { decodeBox($0) }

            // Also check against LiDAR mesh if snapshot provided
            if let json  = (a["snapshotJson"] as? String),
               let data  = json.data(using: .utf8),
               let state = try? RoomCaptureState.from(jsonData: data),
               let mesh  = state.meshAnchors, !mesh.isEmpty {
                result(CollisionChecker.collides(newBox, with: existing) ||
                       CollisionChecker.collidesWithMesh(newBox, meshAnchors: mesh))
            } else {
                result(CollisionChecker.collides(newBox, with: existing))
            }

        // renderDepthMap ──────────────────────────────────────────────────
        // Args: { snapshotJson: String, roomPlanJson: String?, width: Int?, height: Int? }
        // Returns: "data:image/png;base64,…"
        case "renderDepthMap":
            guard let a    = call.arguments as? [String: Any],
                  let sJson = a["snapshotJson"] as? String,
                  let sData = sJson.data(using: .utf8),
                  let state = try? RoomCaptureState.from(jsonData: sData)
            else {
                result(FlutterError(code: "BAD_ARGS", message: "renderDepthMap: missing snapshotJson", details: nil)); return
            }
            let w = a["width"]  as? Int ?? 512
            let h = a["height"] as? Int ?? 512
            let size = CGSize(width: w, height: h)

            var boxes = [DetectedBox3D]()
            if let rpJson = a["roomPlanJson"] as? String,
               let rpData = rpJson.data(using: .utf8),
               let rp     = try? JSONDecoder().decode(RoomPlanResult.self, from: rpData) {
                boxes = rp.objects + rp.walls
            }

            guard let img  = DepthMapRenderer.render(boxes: boxes, captureState: state, outputSize: size),
                  let png  = img.pngData()
            else {
                result(FlutterError(code: "RENDER_FAILED", message: "Depth map render failed", details: nil)); return
            }
            result("data:image/png;base64," + png.base64EncodedString())

        // startRoomPlanScan ───────────────────────────────────────────────
        // No args. Presents native RoomPlan UI. Returns RoomPlanResult map.
        case "startRoomPlanScan":
            #if canImport(RoomPlan)
            if #available(iOS 16.0, *) {
                guard RoomPlanScannerSession.isSupported else {
                    result(FlutterError(code: "UNSUPPORTED",
                                        message: "RoomPlan requires LiDAR", details: nil)); return
                }
                guard pendingRoomPlanResult == nil else {
                    result(FlutterError(code: "BUSY",
                                        message: "Scan already in progress", details: nil)); return
                }
                pendingRoomPlanResult = result
                DispatchQueue.main.async { [weak self] in self?.presentRoomPlanVC() }
            } else {
                result(FlutterError(code: "UNSUPPORTED", message: "iOS 16+ required", details: nil))
            }
            #else
            result(FlutterError(code: "UNSUPPORTED", message: "RoomPlan unavailable", details: nil))
            #endif

        // getCapabilities ─────────────────────────────────────────────────
        case "getCapabilities":
            result(ARCapabilityService.toFlutterMap())

        default:
            result(FlutterMethodNotImplemented)
        }
    }

    // MARK: - Helpers

    private func decodeBox(_ m: [String: Any]) -> DetectedBox3D? {
        guard let label = m["label"] as? String,
              let cx = m["cx"] as? Double, let cy = m["cy"] as? Double,
              let cz = m["cz"] as? Double, let hw = m["hw"] as? Double,
              let hh = m["hh"] as? Double, let hd = m["hd"] as? Double,
              let yaw = m["yaw"] as? Double else { return nil }
        return DetectedBox3D(label: label,
                             cx: Float(cx), cy: Float(cy), cz: Float(cz),
                             hw: Float(hw), hh: Float(hh), hd: Float(hd),
                             yaw: Float(yaw))
    }

    #if canImport(RoomPlan)
    @available(iOS 16.0, *)
    private func presentRoomPlanVC() {
        let vc = RoomPlanViewController()
        vc.modalPresentationStyle = .fullScreen
        vc.onComplete = { [weak self] r in
            guard let self else { return }
            switch r {
            case .success(let plan):
                self.pendingRoomPlanResult?(try? plan.toFlutterMap())
            case .failure(let err):
                self.pendingRoomPlanResult?(
                    FlutterError(code: "SCAN_FAILED",
                                 message: err.localizedDescription, details: nil))
            }
            self.pendingRoomPlanResult = nil
        }
        let scene  = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
        let rootVC = scene?.windows.first(where: \.isKeyWindow)?.rootViewController
        rootVC?.present(vc, animated: true)
    }
    #endif
}
