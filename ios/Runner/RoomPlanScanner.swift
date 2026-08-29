import Foundation
import UIKit
import simd

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Shared 3-D box model
// ─────────────────────────────────────────────────────────────────────────────

struct DetectedBox3D: Codable {
    let label: String
    let cx, cy, cz: Float      // world-space centre
    let hw, hh, hd: Float      // half-extents
    let yaw: Float             // Y-axis rotation (radians)

    var aabbMin: SIMD3<Float> { SIMD3(cx-hw, cy-hh, cz-hd) }
    var aabbMax: SIMD3<Float> { SIMD3(cx+hw, cy+hh, cz+hd) }
}

struct RoomPlanResult: Codable {
    let capturedAtISO: String
    let objects: [DetectedBox3D]
    let walls:   [DetectedBox3D]

    func toFlutterMap() throws -> [String: Any] {
        let data = try JSONEncoder().encode(self)
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - Collision avoidance (AABB)
// ─────────────────────────────────────────────────────────────────────────────

enum CollisionChecker {
    /// True if `newBox` overlaps any box in `existing` (conservative AABB test).
    static func collides(_ newBox: DetectedBox3D, with existing: [DetectedBox3D]) -> Bool {
        let nMin = newBox.aabbMin; let nMax = newBox.aabbMax
        for b in existing {
            if nMin.x <= b.aabbMax.x && nMax.x >= b.aabbMin.x &&
               nMin.y <= b.aabbMax.y && nMax.y >= b.aabbMin.y &&
               nMin.z <= b.aabbMax.z && nMax.z >= b.aabbMin.z { return true }
        }
        return false
    }

    /// Also checks against LiDAR mesh triangles (simplified bounding-box test per anchor).
    static func collidesWithMesh(_ newBox: DetectedBox3D,
                                  meshAnchors: [CapturedMeshAnchor]) -> Bool {
        let nMin = newBox.aabbMin; let nMax = newBox.aabbMax
        for anchor in meshAnchors {
            let t = CapturedExtrinsics(transform: .init()).matrix   // identity default
            let at = simd_float4x4(columns: (
                SIMD4(anchor.transformM[0],  anchor.transformM[1],
                      anchor.transformM[2],  anchor.transformM[3]),
                SIMD4(anchor.transformM[4],  anchor.transformM[5],
                      anchor.transformM[6],  anchor.transformM[7]),
                SIMD4(anchor.transformM[8],  anchor.transformM[9],
                      anchor.transformM[10], anchor.transformM[11]),
                SIMD4(anchor.transformM[12], anchor.transformM[13],
                      anchor.transformM[14], anchor.transformM[15])
            ))
            _ = t   // suppress unused warning
            var meshMin = SIMD3<Float>(repeating:  .infinity)
            var meshMax = SIMD3<Float>(repeating: -.infinity)
            let verts = anchor.vertices
            guard verts.count >= 3 else { continue }
            for i in stride(from: 0, to: verts.count - 2, by: 3) {
                let lp = SIMD4<Float>(verts[i], verts[i+1], verts[i+2], 1)
                let wp = at * lp
                let p3 = SIMD3<Float>(wp.x, wp.y, wp.z)
                meshMin = simd_min(meshMin, p3)
                meshMax = simd_max(meshMax, p3)
            }
            if nMin.x <= meshMax.x && nMax.x >= meshMin.x &&
               nMin.y <= meshMax.y && nMax.y >= meshMin.y &&
               nMin.z <= meshMax.z && nMax.z >= meshMin.z { return true }
        }
        return false
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - ControlNet depth-map renderer
// ─────────────────────────────────────────────────────────────────────────────

enum DepthMapRenderer {

    // MARK: LiDAR path — use the real captured depth PNG (most accurate)

    /// Pass-through: the depth PNG captured from ARFrame is already ControlNet-ready.
    /// Optionally resize to `outputSize`.
    static func renderFromLiDAR(
        captureState: RoomCaptureState,
        outputSize: CGSize = CGSize(width: 512, height: 512)
    ) -> UIImage? {
        guard let b64 = captureState.depthMapPng,
              let data  = Data(base64Encoded: b64),
              let src   = UIImage(data: data) else { return nil }

        // If size matches, return as-is; otherwise resize
        if abs(src.size.width  - outputSize.width)  < 1 &&
           abs(src.size.height - outputSize.height) < 1 { return src }

        UIGraphicsBeginImageContextWithOptions(outputSize, false, 1)
        src.draw(in: CGRect(origin: .zero, size: outputSize))
        let out = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return out
    }

    // MARK: Standard path — project RoomPlan/mesh boxes onto camera plane

    static func renderFromBoxes(
        boxes: [DetectedBox3D],
        captureState: RoomCaptureState,
        outputSize: CGSize = CGSize(width: 512, height: 512)
    ) -> UIImage? {
        let W = Int(outputSize.width), H = Int(outputSize.height)
        var depthBuf = [Float](repeating: .infinity, count: W * H)

        let K      = captureState.intrinsics
        let camInv = captureState.extrinsics.matrix.inverse
        let sX     = Float(W) / Float(K.imageWidth)
        let sY     = Float(H) / Float(K.imageHeight)

        for box in boxes {
            var minD = Float.infinity
            var pxArr = [(Float, Float)]()
            for wc in worldCorners(of: box) {
                let cc = camInv * SIMD4<Float>(wc.x, wc.y, wc.z, 1)
                guard cc.z < 0 else { continue }
                let d  = -cc.z
                let px = (K.fx * cc.x / d + K.cx) * sX
                let py = (K.fy * cc.y / d + K.cy) * sY
                pxArr.append((px, py))
                if d < minD { minD = d }
            }
            guard pxArr.count >= 2, minD < .infinity else { continue }
            let x0 = max(0, Int(pxArr.map(\.0).min()!.rounded(.down)))
            let x1 = min(W-1, Int(pxArr.map(\.0).max()!.rounded(.up)))
            let y0 = max(0, Int(pxArr.map(\.1).min()!.rounded(.down)))
            let y1 = min(H-1, Int(pxArr.map(\.1).max()!.rounded(.up)))
            guard x0 <= x1, y0 <= y1 else { continue }
            for r in y0...y1 { for c in x0...x1 {
                let i = r*W+c; if minD < depthBuf[i] { depthBuf[i] = minD }
            }}
        }

        let finite = depthBuf.filter { $0 < .infinity }
        let maxD   = finite.max() ?? 1; let minD = finite.min() ?? 0
        let rng    = max(maxD - minD, 1e-4)
        var pixels = [UInt8](repeating: 0, count: W * H)
        for i in 0..<pixels.count {
            guard depthBuf[i] < .infinity else { continue }
            pixels[i] = UInt8(min(max((1 - (depthBuf[i]-minD)/rng) * 255, 0), 255))
        }
        return makeGrey(pixels: pixels, w: W, h: H)
    }

    // MARK: Smart dispatch — uses LiDAR depth if available, boxes otherwise

    static func render(
        boxes: [DetectedBox3D],
        captureState: RoomCaptureState,
        outputSize: CGSize = CGSize(width: 512, height: 512)
    ) -> UIImage? {
        if captureState.captureMode == "lidar",
           captureState.depthMapPng != nil {
            return renderFromLiDAR(captureState: captureState, outputSize: outputSize)
        }
        return renderFromBoxes(boxes: boxes, captureState: captureState, outputSize: outputSize)
    }

    // MARK: - Private helpers

    private static func worldCorners(of b: DetectedBox3D) -> [SIMD3<Float>] {
        let c = cos(b.yaw), s = sin(b.yaw)
        let signs: [(Float,Float,Float)] = [
            (-1,-1,-1),(1,-1,-1),(1,1,-1),(-1,1,-1),
            (-1,-1, 1),(1,-1, 1),(1,1, 1),(-1,1, 1),
        ]
        return signs.map { sx, sy, sz in
            let lx = sx*b.hw, lz = sz*b.hd
            return SIMD3<Float>(b.cx + c*lx + s*lz,  b.cy + sy*b.hh,  b.cz - s*lx + c*lz)
        }
    }

    private static func makeGrey(pixels: [UInt8], w: Int, h: Int) -> UIImage? {
        guard let ctx = CGContext(data: nil, width: w, height: h,
                                  bitsPerComponent: 8, bytesPerRow: w,
                                  space: CGColorSpaceCreateDeviceGray(),
                                  bitmapInfo: CGImageAlphaInfo.none.rawValue),
              let ptr = ctx.data else { return nil }
        pixels.withUnsafeBytes { ptr.copyMemory(from: $0.baseAddress!, byteCount: pixels.count) }
        return ctx.makeImage().map { UIImage(cgImage: $0) }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MARK: - RoomPlan session (iOS 16+ / LiDAR only)
// ─────────────────────────────────────────────────────────────────────────────

#if canImport(RoomPlan)
import RoomPlan

@available(iOS 16.0, *)
final class RoomPlanScannerSession: NSObject, RoomCaptureSessionDelegate {

    private weak var captureView: RoomCaptureView?
    var onComplete: ((Result<RoomPlanResult, Error>) -> Void)?
    static var isSupported: Bool { RoomCaptureSession.isSupported }

    func start(in view: RoomCaptureView) {
        captureView = view
        view.captureSession.delegate = self
        view.captureSession.run(configuration: .init())
    }

    func stop() { captureView?.captureSession.stop() }

    func captureSession(_ session: RoomCaptureSession,
                        didEndWith room: CapturedRoom, error: Error?) {
        if let e = error { onComplete?(.failure(e)); return }
        onComplete?(.success(buildResult(from: room)))
    }

    private func buildResult(from room: CapturedRoom) -> RoomPlanResult {
        func box(_ t: simd_float4x4, _ d: simd_float3, _ l: String) -> DetectedBox3D {
            DetectedBox3D(label: l,
                          cx: t.columns.3.x, cy: t.columns.3.y, cz: t.columns.3.z,
                          hw: d.x*0.5, hh: d.y*0.5, hd: d.z*0.5,
                          yaw: atan2(t.columns.0.z, t.columns.0.x))
        }
        let objects = room.objects.map { box($0.transform, $0.dimensions, label(for: $0.category)) }
        let walls   = room.walls.map   { box($0.transform, $0.dimensions, "wall") }
        return RoomPlanResult(capturedAtISO: ISO8601DateFormatter().string(from: Date()),
                              objects: objects, walls: walls)
    }

    private func label(for cat: CapturedRoom.Object.Category) -> String {
        switch cat {
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

@available(iOS 16.0, *)
final class RoomPlanViewController: UIViewController {

    private let scannerSession = RoomPlanScannerSession()
    private let captureView    = RoomCaptureView()
    var onComplete: ((Result<RoomPlanResult, Error>) -> Void)?

    override func viewDidLoad() {
        super.viewDidLoad()
        view.backgroundColor = .black
        captureView.frame    = view.bounds
        captureView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        view.addSubview(captureView)

        let btn = UIButton(type: .system)
        btn.setTitle("Done Scanning", for: .normal)
        btn.titleLabel?.font = .boldSystemFont(ofSize: 17)
        btn.backgroundColor  = .systemBlue
        btn.setTitleColor(.white, for: .normal)
        btn.layer.cornerRadius = 14
        btn.addTarget(self, action: #selector(doneTapped), for: .touchUpInside)
        btn.translatesAutoresizingMaskIntoConstraints = false
        view.addSubview(btn)
        NSLayoutConstraint.activate([
            btn.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            btn.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            btn.widthAnchor.constraint(equalToConstant: 200),
            btn.heightAnchor.constraint(equalToConstant: 52),
        ])

        scannerSession.onComplete = { [weak self] r in
            DispatchQueue.main.async { self?.dismiss(animated: true) { self?.onComplete?(r) } }
        }
        scannerSession.start(in: captureView)
    }

    @objc private func doneTapped() { scannerSession.stop() }
}
#endif
