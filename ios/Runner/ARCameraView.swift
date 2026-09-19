import Flutter
import UIKit
import ARKit

// MARK: - Factory

class ARCameraFactory: NSObject, FlutterPlatformViewFactory {
    private var messenger: FlutterBinaryMessenger

    init(messenger: FlutterBinaryMessenger) {
        self.messenger = messenger
        super.init()
    }

    func create(withFrame frame: CGRect,
                viewIdentifier viewId: Int64,
                arguments args: Any?) -> FlutterPlatformView {
        ARCameraNativeView(frame: frame, viewIdentifier: viewId,
                           arguments: args, binaryMessenger: messenger)
    }
}

// MARK: - Platform view

class ARCameraNativeView: NSObject, FlutterPlatformView, ARSCNViewDelegate {
    private let arView = ARSCNView()
    private var points:     [simd_float3] = []
    private var sceneNodes: [SCNNode]     = []

    init(frame: CGRect, viewIdentifier viewId: Int64,
         arguments args: Any?, binaryMessenger messenger: FlutterBinaryMessenger) {
        super.init()
        arView.frame = frame
        arView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        arView.delegate = self
        arView.debugOptions = [.showFeaturePoints]

        startARSession()

        let ch = FlutterMethodChannel(name: "com.smartdeco.app/ar_camera_\(viewId)",
                                      binaryMessenger: messenger)
        ch.setMethodCallHandler { [weak self] call, result in
            guard let self else { return }
            switch call.method {
            case "addPoint":       self.handleAddPoint(result: result)
            case "clearPoints":    self.clearAll(); result(nil)
            case "captureSnapshot":self.handleCaptureSnapshot(result: result)
            case "samplePoint":    self.handleSamplePoint(result: result)
            case "capturePhoto":   self.handleCapturePhoto(result: result)
            case "getCapabilities":result(ARCapabilityService.toFlutterMap())
            default:               result(FlutterMethodNotImplemented)
            }
        }
    }

    func view() -> UIView { arView }

    // MARK: - AR session configuration

    private func startARSession() {
        let config = ARWorldTrackingConfiguration()
        config.planeDetection = [.horizontal, .vertical]

        if ARCapabilityService.hasLiDAR {
            if #available(iOS 13.4, *) {
                config.sceneReconstruction = .meshWithClassification
            }
            if #available(iOS 14.0, *), ARCapabilityService.supportsSceneDepth {
                config.frameSemantics = [.sceneDepth, .smoothedSceneDepth]
            }
            arView.debugOptions = []                 // clean view on LiDAR devices
        }

        arView.session.run(config, options: [.resetTracking, .removeExistingAnchors])
    }

    // MARK: - Add corner point

    private func handleAddPoint(result: FlutterResult) {
        guard let frame = arView.session.currentFrame else {
            result(["error": "no_frame"]); return
        }

        let centre = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
        let position: simd_float3

        // LiDAR path: sample depth at screen centre for sub-centimetre accuracy
        if ARCapabilityService.hasLiDAR,
           #available(iOS 14.0, *),
           let depth = frame.smoothedSceneDepth ?? frame.sceneDepth {
            position = depthSamplePoint(frame: frame, depth: depth, screenPt: centre)
        } else if let query = arView.raycastQuery(from: centre, allowing: .estimatedPlane,
                                                  alignment: .any),
                  let hit = arView.session.raycast(query).first {
            let c = hit.worldTransform.columns.3
            position = simd_make_float3(c.x, c.y, c.z)
        } else {
            let c = frame.camera.transform.columns.3
            position = simd_make_float3(c.x, c.y, c.z)
        }

        addSphereMarker(at: position)
        if let prev = points.last { addLineMarker(from: prev, to: position) }
        points.append(position)

        let dist: Float = points.count >= 2
            ? simd_length(points.last! - points[points.count - 2]) : 0
        let allPts = points.map { ["x": Double($0.x), "y": Double($0.y), "z": Double($0.z)] }
        result(["points": points.count, "distance": Double(dist), "allPoints": allPts])
    }

    // MARK: - Sample single 3-D point (no marker, used for ceiling height)

    private func handleSamplePoint(result: FlutterResult) {
        guard let frame = arView.session.currentFrame else {
            result(FlutterError(code: "NO_FRAME", message: "No current AR frame", details: nil))
            return
        }
        let centre = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
        let position: simd_float3

        if ARCapabilityService.hasLiDAR,
           #available(iOS 14.0, *),
           let depth = frame.smoothedSceneDepth ?? frame.sceneDepth {
            position = depthSamplePoint(frame: frame, depth: depth, screenPt: centre)
        } else if let query = arView.raycastQuery(from: centre, allowing: .estimatedPlane,
                                                  alignment: .any),
                  let hit = arView.session.raycast(query).first {
            let c = hit.worldTransform.columns.3
            position = simd_make_float3(c.x, c.y, c.z)
        } else {
            result(FlutterError(code: "NO_HIT", message: "Could not sample depth", details: nil))
            return
        }

        result(["x": Double(position.x), "y": Double(position.y), "z": Double(position.z)])
    }

    // MARK: - Capture snapshot

    private func handleCaptureSnapshot(result: FlutterResult) {
        guard let frame = arView.session.currentFrame else {
            result(FlutterError(code: "NO_FRAME", message: "No current AR frame", details: nil))
            return
        }
        do {
            let state = RoomCaptureState(frame: frame)
            result(try state.toFlutterMap())
        } catch {
            result(FlutterError(code: "SERIALISE_FAILED",
                                message: error.localizedDescription, details: nil))
        }
    }

    // MARK: - AR photo capture (image + extrinsics + intrinsics)

    private func handleCapturePhoto(result: @escaping FlutterResult) {
        guard let frame = arView.session.currentFrame else {
            result(FlutterError(code: "NO_FRAME", message: "No current AR frame", details: nil))
            return
        }

        let pixelBuffer = frame.capturedImage
        let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
        let context = CIContext(options: nil)
        guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
            result(FlutterError(code: "CONVERSION_FAILED", message: "Cannot convert pixel buffer to CGImage", details: nil))
            return
        }
        let uiImage = UIImage(cgImage: cgImage)
        guard let jpegData = uiImage.jpegData(compressionQuality: 0.85) else {
            result(FlutterError(code: "JPEG_FAILED", message: "Cannot encode JPEG", details: nil))
            return
        }

        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let ts = Int(Date().timeIntervalSince1970 * 1000)
        let fileURL = docs.appendingPathComponent("ar_photo_\(ts).jpg")
        do {
            try jpegData.write(to: fileURL)
        } catch {
            result(FlutterError(code: "WRITE_FAILED", message: error.localizedDescription, details: nil))
            return
        }

        // Camera-to-world transform (column-major 4×4)
        let T = frame.camera.transform
        let ext: [Double] = [
            Double(T.columns.0.x), Double(T.columns.0.y), Double(T.columns.0.z), Double(T.columns.0.w),
            Double(T.columns.1.x), Double(T.columns.1.y), Double(T.columns.1.z), Double(T.columns.1.w),
            Double(T.columns.2.x), Double(T.columns.2.y), Double(T.columns.2.z), Double(T.columns.2.w),
            Double(T.columns.3.x), Double(T.columns.3.y), Double(T.columns.3.z), Double(T.columns.3.w),
        ]

        // Camera intrinsics (row-major 3×3): [fx,0,cx, 0,fy,cy, 0,0,1]
        // ARKit stores K column-major: cols.0=(fx,0,0), cols.1=(0,fy,0), cols.2=(cx,cy,1)
        let K = frame.camera.intrinsics
        let intr: [Double] = [
            Double(K.columns.0.x), Double(K.columns.1.x), Double(K.columns.2.x),
            Double(K.columns.0.y), Double(K.columns.1.y), Double(K.columns.2.y),
            Double(K.columns.0.z), Double(K.columns.1.z), Double(K.columns.2.z),
        ]

        let imgW = CVPixelBufferGetWidth(pixelBuffer)
        let imgH = CVPixelBufferGetHeight(pixelBuffer)

        let iso = ISO8601DateFormatter()
        iso.formatOptions = [.withInternetDateTime, .withFractionalSeconds]

        result([
            "imagePath":  fileURL.path,
            "extrinsics": ext,
            "intrinsics": intr,
            "imageWidth":  imgW,
            "imageHeight": imgH,
            "platform":   "ios",
            "capturedAt": iso.string(from: Date()),
        ] as [String: Any])
    }

    // MARK: - LiDAR depth point sampling

    @available(iOS 14.0, *)
    private func depthSamplePoint(frame: ARFrame, depth: ARDepthData,
                                  screenPt: CGPoint) -> simd_float3 {
        let K = frame.camera.intrinsics
        let camT = frame.camera.transform

        let dpBuf = depth.depthMap
        let dW = CVPixelBufferGetWidth(dpBuf)
        let dH = CVPixelBufferGetHeight(dpBuf)

        // Map screen point into depth-buffer space
        let viewW = arView.bounds.width, viewH = arView.bounds.height
        let du = Float(screenPt.x / viewW) * Float(dW)
        let dv = Float(screenPt.y / viewH) * Float(dH)
        let col = min(max(Int(du), 0), dW - 1)
        let row = min(max(Int(dv), 0), dH - 1)

        CVPixelBufferLockBaseAddress(dpBuf, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(dpBuf, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(dpBuf) else {
            let c = camT.columns.3; return simd_make_float3(c.x, c.y, c.z)
        }
        let depthM = base.assumingMemoryBound(to: Float32.self)[row * dW + col]

        guard depthM.isFinite, depthM > 0.1 else {
            let c = camT.columns.3; return simd_make_float3(c.x, c.y, c.z)
        }

        // Unproject: camera intrinsics (landscape) vs screen (portrait) need normalised UV
        let viewSize  = CGSize(width: viewW, height: viewH)
        let imageSize = frame.camera.imageResolution
        let normPt    = frame.camera.projectPoint(
            simd_float3(0, 0, -1),               // ignored — we only need the unproject direction
            orientation: .portrait,
            viewportSize: viewSize
        )
        // Use direct pinhole unproject with landscape intrinsics
        // ARKit: camera looks along -Z; depth = distance along that axis
        let xCam = (Float(screenPt.x / viewW) * Float(imageSize.width)  - K.columns.2.x) / K.columns.0.x
        let yCam = (Float(screenPt.y / viewH) * Float(imageSize.height) - K.columns.2.y) / K.columns.1.y
        let _ = normPt // suppress unused warning

        let camSpacePt = SIMD4<Float>(xCam * depthM, yCam * depthM, -depthM, 1)
        let worldPt    = camT * camSpacePt
        return simd_make_float3(worldPt.x, worldPt.y, worldPt.z)
    }

    // MARK: - Scene markers

    private func clearAll() {
        points.removeAll()
        sceneNodes.forEach { $0.removeFromParentNode() }
        sceneNodes.removeAll()
    }

    private func addSphereMarker(at pos: simd_float3) {
        let s = SCNSphere(radius: 0.02)
        s.firstMaterial?.diffuse.contents   = UIColor.yellow
        s.firstMaterial?.emission.contents  = UIColor.yellow.withAlphaComponent(0.4)
        let n = SCNNode(geometry: s)
        n.position = SCNVector3(pos.x, pos.y, pos.z)
        arView.scene.rootNode.addChildNode(n)
        sceneNodes.append(n)
    }

    private func addLineMarker(from start: simd_float3, to end: simd_float3) {
        let diff   = end - start
        let length = max(simd_length(diff), 0.001)
        let cyl    = SCNCylinder(radius: 0.006, height: CGFloat(length))
        cyl.firstMaterial?.diffuse.contents  = UIColor.cyan
        cyl.firstMaterial?.emission.contents = UIColor.cyan.withAlphaComponent(0.5)
        let n   = SCNNode(geometry: cyl)
        let mid = (start + end) * 0.5
        n.position = SCNVector3(mid.x, mid.y, mid.z)

        let yAxis = simd_float3(0, 1, 0)
        let dir   = simd_normalize(diff)
        let cross = simd_cross(yAxis, dir)
        let cLen  = simd_length(cross)
        if cLen < 0.001 {
            if simd_dot(yAxis, dir) < 0 { n.eulerAngles = SCNVector3(Float.pi, 0, 0) }
        } else {
            n.simdRotation = simd_float4(cross / cLen,
                                         acos(min(max(simd_dot(yAxis, dir), -1), 1)))
        }
        arView.scene.rootNode.addChildNode(n)
        sceneNodes.append(n)
    }
}
