import Foundation
import UIKit
import simd

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Shared 3-D box model
// ─────────────────────────────────────────────────────────────────────────────

struct DetectedBox3D: Codable {
    let label: String
    // World-space centre
    let cx: Float; let cy: Float; let cz: Float
    // Half-extents
    let hw: Float; let hh: Float; let hd: Float
    // Yaw rotation around Y-axis (radians)
    let yaw: Float

    // Conservative AABB (axis-aligned, ignores yaw — slightly over-estimates)
    var aabbMin: SIMD3<Float> { SIMD3(cx - hw, cy - hh, cz - hd) }
    var aabbMax: SIMD3<Float> { SIMD3(cx + hw, cy + hh, cz + hd) }
}

struct RoomPlanResult: Codable {
    let capturedAtISO: String
    let objects: [DetectedBox3D]   // furniture + appliances
    let walls:   [DetectedBox3D]

    func toFlutterMap() throws -> [String: Any] {
        let data = try JSONEncoder().encode(self)
        return try (JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Collision avoidance
// ─────────────────────────────────────────────────────────────────────────────

enum CollisionChecker {
    /// Returns `true` when `newBox` (AABB) overlaps any box in `existing`.
    /// Call this before placing AI-generated furniture to avoid spatial conflicts.
    static func collides(_ newBox: DetectedBox3D, with existing: [DetectedBox3D]) -> Bool {
        let nMin = newBox.aabbMin
        let nMax = newBox.aabbMax
        for box in existing {
            let eMin = box.aabbMin
            let eMax = box.aabbMax
            if nMin.x <= eMax.x && nMax.x >= eMin.x &&
               nMin.y <= eMax.y && nMax.y >= eMin.y &&
               nMin.z <= eMax.z && nMax.z >= eMin.z {
                return true
            }
        }
        return false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Depth-map renderer
// ─────────────────────────────────────────────────────────────────────────────
// Renders every DetectedBox3D as its eight corners projected onto the camera
// image plane, drawing a filled screen-space AABB per box.
// Output: greyscale UIImage — white = near, black = far/empty.
// Send this alongside the room photo to ControlNet for geometry-aware generation.

enum DepthMapRenderer {

    static func render(
        boxes: [DetectedBox3D],
        captureState: RoomCaptureState,
        outputSize: CGSize = CGSize(width: 512, height: 512)
    ) -> UIImage? {
        let W = Int(outputSize.width)
        let H = Int(outputSize.height)
        var depthBuf = [Float](repeating: .infinity, count: W * H)

        let K    = captureState.intrinsics
        let camT = captureState.extrinsics.matrix
        // World → camera transform
        let camInv = camT.inverse

        // Scale from capture-resolution pixels to output-resolution pixels
        let scaleX = Float(W) / Float(K.imageWidth)
        let scaleY = Float(H) / Float(K.imageHeight)

        for box in boxes {
            let corners = worldCorners(of: box)

            var projPts  = [(px: Float, py: Float)]()
            var minDepth = Float.infinity

            for wc in corners {
                // Transform world point into camera space
                let cc = camInv * SIMD4<Float>(wc.x, wc.y, wc.z, 1)
                // ARKit camera looks along -Z: depth is positive in front of camera
                guard cc.z < 0 else { continue }
                let depth = -cc.z

                let px = (K.fx * cc.x / depth + K.cx) * scaleX
                let py = (K.fy * cc.y / depth + K.cy) * scaleY
                projPts.append((px, py))
                if depth < minDepth { minDepth = depth }
            }
            guard projPts.count >= 2, minDepth < .infinity else { continue }

            // Fill the screen-space bounding rect of this box's projected corners
            let minX = max(0, Int(projPts.map(\.px).min()!.rounded(.down)))
            let maxX = min(W - 1, Int(projPts.map(\.px).max()!.rounded(.up)))
            let minY = max(0, Int(projPts.map(\.py).min()!.rounded(.down)))
            let maxY = min(H - 1, Int(projPts.map(\.py).max()!.rounded(.up)))

            guard minX <= maxX, minY <= maxY else { continue }

            for row in minY...maxY {
                for col in minX...maxX {
                    let idx = row * W + col
                    if minDepth < depthBuf[idx] { depthBuf[idx] = minDepth }
                }
            }
        }

        // Normalise to 0–255: closer → brighter (ControlNet convention)
        let finite = depthBuf.filter { $0 < .infinity }
        let maxD   = finite.max() ?? 1
        let minD   = finite.min() ?? 0
        let range  = max(maxD - minD, 1e-4)

        var pixels = [UInt8](repeating: 0, count: W * H)
        for i in 0 ..< pixels.count {
            guard depthBuf[i] < .infinity else { continue }
            let norm   = 1.0 - (depthBuf[i] - minD) / range   // 1 = closest
            pixels[i]  = UInt8(min(max(norm * 255, 0), 255))
        }

        return makeGrey(pixels: pixels, w: W, h: H)
    }

    // MARK: - Helpers

    /// Returns the 8 world-space corners of a yaw-rotated box.
    private static func worldCorners(of box: DetectedBox3D) -> [SIMD3<Float>] {
        let c = Foundation.cos(box.yaw)
        let s = Foundation.sin(box.yaw)
        return [(-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
                (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1)].map { (sx, sy, sz) in
            let lx = Float(sx) * box.hw
            let ly = Float(sy) * box.hh
            let lz = Float(sz) * box.hd
            return SIMD3<Float>(
                box.cx + c * lx + s * lz,
                box.cy + ly,
                box.cz - s * lx + c * lz
            )
        }
    }

    private static func makeGrey(pixels: [UInt8], w: Int, h: Int) -> UIImage? {
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ), let ptr = ctx.data else { return nil }

        ptr.copyMemory(from: pixels, byteCount: pixels.count)
        return ctx.makeImage().map { UIImage(cgImage: $0) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - RoomPlan session (iOS 16 + LiDAR required)
// ─────────────────────────────────────────────────────────────────────────────

#if canImport(RoomPlan)
import RoomPlan

@available(iOS 16.0, *)
final class RoomPlanScannerSession: NSObject, RoomCaptureSessionDelegate {

    private let captureSession = RoomCaptureSession()
    var onComplete: ((Result<RoomPlanResult, Error>) -> Void)?

    static var isSupported: Bool { RoomCaptureSession.isSupported }

    func start(in view: RoomCaptureView) {
        captureSession.delegate = self
        let config = RoomCaptureSession.Configuration()
        view.captureSession = captureSession
        captureSession.run(configuration: config)
    }

    func stop() { captureSession.stop() }

    // MARK: Delegate

    func captureSession(_ session: RoomCaptureSession,
                        didEndWith structuredRoomData: CapturedRoom,
                        error: Error?) {
        if let error { onComplete?(.failure(error)); return }
        onComplete?(.success(buildResult(from: structuredRoomData)))
    }

    // MARK: Build result

    private func buildResult(from room: CapturedRoom) -> RoomPlanResult {
        let objects = room.objects.map { obj -> DetectedBox3D in
            let t = obj.transform; let d = obj.dimensions
            return DetectedBox3D(
                label: label(for: obj.category),
                cx: t.columns.3.x, cy: t.columns.3.y, cz: t.columns.3.z,
                hw: d.x * 0.5,     hh: d.y * 0.5,     hd: d.z * 0.5,
                yaw: atan2(t.columns.0.z, t.columns.0.x)
            )
        }
        let walls = room.walls.map { w -> DetectedBox3D in
            let t = w.transform; let d = w.dimensions
            return DetectedBox3D(
                label: "wall",
                cx: t.columns.3.x, cy: t.columns.3.y, cz: t.columns.3.z,
                hw: d.x * 0.5,     hh: d.y * 0.5,     hd: d.z * 0.5,
                yaw: atan2(t.columns.0.z, t.columns.0.x)
            )
        }
        return RoomPlanResult(
            capturedAtISO: ISO8601DateFormatter().string(from: Date()),
            objects: objects, walls: walls
        )
    }

    private func label(for category: CapturedRoom.Object.Category) -> String {
        switch category {
        case .sofa:         return "sofa"
        case .chair:        return "chair"
        case .table:        return "table"
        case .bed:          return "bed"
        case .storage:      return "storage"
        case .television:   return "television"
        case .refrigerator: return "refrigerator"
        case .washerDryer:  return "washerDryer"
        case .oven:         return "oven"
        case .sink:         return "sink"
        case .toilet:       return "toilet"
        case .bathtub:      return "bathtub"
        case .stairs:       return "stairs"
        case .fireplace:    return "fireplace"
        default:            return "furniture"
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - RoomPlan full-screen view controller
// ─────────────────────────────────────────────────────────────────────────────

@available(iOS 16.0, *)
final class RoomPlanViewController: UIViewController {

    private let scannerSession = RoomPlanScannerSession()
    private let captureView    = RoomCaptureView()
    var onComplete: ((Result<RoomPlanResult, Error>) -> Void)?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        captureView.frame = view.bounds
        captureView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(captureView)

        // Done button
        let doneBtn = UIButton(type: .system)
        doneBtn.setTitle("Done Scanning", for: .normal)
        doneBtn.titleLabel?.font = .boldSystemFont(ofSize: 17)
        doneBtn.backgroundColor = UIColor.systemBlue
        doneBtn.setTitleColor(.white, for: .normal)
        doneBtn.layer.cornerRadius = 14
        doneBtn.addTarget(self, action: #selector(doneTapped), for: .touchUpInside)
        doneBtn.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(doneBtn)
        NSLayoutConstraint.activate([
            doneBtn.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            doneBtn.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            doneBtn.widthAnchor.constraint(equalToConstant: 200),
            doneBtn.heightAnchor.constraint(equalToConstant: 52),
        ])

        scannerSession.onComplete = { [weak self] result in
            DispatchQueue.main.async {
                self?.dismiss(animated: true) { self?.onComplete?(result) }
            }
        }
        scannerSession.start(in: captureView)
    }

    @objc private func doneTapped() {
        scannerSession.stop()   // triggers delegate → onComplete
    }
}
#endif
