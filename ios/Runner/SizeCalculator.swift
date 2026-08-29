import UIKit
import simd

// MARK: - Public types

struct BBox2D {
    let xMin, yMin, xMax, yMax: Float
}

struct FurnitureSize: Codable {
    let widthMeters, heightMeters, depthMeters: Float
    let method: String   // "lidar_depth" | "plane_raycast"
}

// MARK: - Calculator

enum SizeCalculator {

    /// Primary entry point — dispatches to the correct path based on what the snapshot contains.
    static func measure(bbox: BBox2D, in state: RoomCaptureState) -> FurnitureSize? {
        if state.captureMode == "lidar",
           let pngB64   = state.depthMapPng,
           let minM     = state.depthMinMeters,
           let maxM     = state.depthMaxMeters {
            return measureWithDepth(bbox: bbox,
                                    depthPngB64: pngB64,
                                    depthMin: minM, depthMax: maxM,
                                    K: state.intrinsics,
                                    camT: state.extrinsics.matrix)
        }
        return measureWithPlane(bbox: bbox, in: state)
    }

    // MARK: - LiDAR path: depth-map unprojection

    /// Samples the saved depth map at the bbox corners, unprojects to 3-D, measures distances.
    static func measureWithDepth(
        bbox: BBox2D,
        depthPngB64: String,
        depthMin: Float,   // metres when normalised pixel = 1.0 (closest)
        depthMax: Float,   // metres when normalised pixel = 0.0 (farthest)
        K: CapturedIntrinsics,
        camT: simd_float4x4
    ) -> FurnitureSize? {
        guard
            let pngData  = Data(base64Encoded: depthPngB64),
            let img      = UIImage(data: pngData),
            let cgImg    = img.cgImage
        else { return nil }

        let dW = cgImg.width, dH = cgImg.height
        guard dW > 0, dH > 0 else { return nil }

        // Render grey image into a UInt8 buffer
        var buf = [UInt8](repeating: 0, count: dW * dH)
        guard let ctx = CGContext(
            data: &buf, width: dW, height: dH,
            bitsPerComponent: 8, bytesPerRow: dW,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ) else { return nil }
        ctx.draw(cgImg, in: CGRect(x: 0, y: 0, width: dW, height: dH))

        // Corners: (pixel in capture-image space, scaled to depth-buffer size)
        let scaleX = Float(dW) / Float(K.imageWidth)
        let scaleY = Float(dH) / Float(K.imageHeight)

        let corners: [SIMD2<Float>] = [
            SIMD2(bbox.xMin, bbox.yMin), SIMD2(bbox.xMax, bbox.yMin),
            SIMD2(bbox.xMin, bbox.yMax), SIMD2(bbox.xMax, bbox.yMax),
        ]

        var pts3D = [SIMD3<Float>]()
        for c in corners {
            let du = Int(c.x * scaleX); let dv = Int(c.y * scaleY)
            let col = min(max(du, 0), dW - 1)
            let row = min(max(dv, 0), dH - 1)
            let pixVal = Float(buf[row * dW + col]) / 255.0  // 1=closest=minM, 0=farthest=maxM
            let depthM = depthMin + (1.0 - pixVal) * (depthMax - depthMin)
            guard depthM.isFinite, depthM > 0.05 else { continue }
            if let p = unproject(pixel: c, depth: depthM, K: K, camT: camT) {
                pts3D.append(p)
            }
        }
        guard pts3D.count == 4 else { return nil }

        let width  = (simd_distance(pts3D[0], pts3D[1]) + simd_distance(pts3D[2], pts3D[3])) * 0.5
        let depth  = (simd_distance(pts3D[0], pts3D[2]) + simd_distance(pts3D[1], pts3D[3])) * 0.5
        let height = estimateHeight(bbox: bbox, pts3D: pts3D, K: K, camT: camT)

        return FurnitureSize(widthMeters: max(width, 0),
                             heightMeters: max(height, 0),
                             depthMeters: max(depth, 0),
                             method: "lidar_depth")
    }

    // MARK: - Standard path: ray-plane intersection

    static func measureWithPlane(bbox: BBox2D, in state: RoomCaptureState) -> FurnitureSize? {
        guard let floor = state.floorPlane else { return nil }
        let K = state.intrinsics; let camT = state.extrinsics.matrix

        let corners = [
            SIMD2<Float>(bbox.xMin, bbox.yMax), SIMD2<Float>(bbox.xMax, bbox.yMax),
            SIMD2<Float>(bbox.xMin, bbox.yMin), SIMD2<Float>(bbox.xMax, bbox.yMin),
        ]
        let pts = corners.compactMap { floorPoint(pixel: $0, K: K, camT: camT, floor: floor) }
        guard pts.count == 4 else { return nil }

        let width  = (simd_distance(pts[0], pts[1]) + simd_distance(pts[2], pts[3])) * 0.5
        let depth  = (simd_distance(pts[0], pts[2]) + simd_distance(pts[1], pts[3])) * 0.5
        let height = estimateHeight(bbox: bbox, pts3D: pts, K: K, camT: camT)

        return FurnitureSize(widthMeters: max(width, 0),
                             heightMeters: max(height, 0),
                             depthMeters: max(depth, 0),
                             method: "plane_raycast")
    }

    // MARK: - Shared math

    /// Unprojects a pixel + metric depth into world space.
    /// ARKit pinhole model: Xc = D*(u-cx)/fx, Yc = D*(v-cy)/fy, Zc = -D
    static func unproject(pixel: SIMD2<Float>, depth: Float,
                           K: CapturedIntrinsics, camT: simd_float4x4) -> SIMD3<Float>? {
        guard depth > 0 else { return nil }
        let xc = (pixel.x - K.cx) / K.fx * depth
        let yc = (pixel.y - K.cy) / K.fy * depth
        let wPt = camT * SIMD4<Float>(xc, yc, -depth, 1)
        return SIMD3(wPt.x, wPt.y, wPt.z)
    }

    static func worldRayDir(pixel: SIMD2<Float>,
                             K: CapturedIntrinsics,
                             camT: simd_float4x4) -> SIMD3<Float> {
        let dirCam = SIMD3<Float>((pixel.x - K.cx) / K.fx,
                                   (pixel.y - K.cy) / K.fy, -1.0)
        let R = simd_float3x3(
            SIMD3(camT.columns.0.x, camT.columns.0.y, camT.columns.0.z),
            SIMD3(camT.columns.1.x, camT.columns.1.y, camT.columns.1.z),
            SIMD3(camT.columns.2.x, camT.columns.2.y, camT.columns.2.z)
        )
        return simd_normalize(R * dirCam)
    }

    private static func floorPoint(pixel: SIMD2<Float>, K: CapturedIntrinsics,
                                    camT: simd_float4x4, floor: CapturedPlane) -> SIMD3<Float>? {
        let origin = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        let dir    = worldRayDir(pixel: pixel, K: K, camT: camT)
        let denom  = simd_dot(floor.normal, dir)
        guard abs(denom) > 1e-6 else { return nil }
        let t = (floor.distanceFromOrigin - simd_dot(floor.normal, origin)) / denom
        guard t > 0 else { return nil }
        return origin + dir * t
    }

    private static func estimateHeight(bbox: BBox2D, pts3D: [SIMD3<Float>],
                                        K: CapturedIntrinsics, camT: simd_float4x4) -> Float {
        let floorCentre = pts3D.reduce(SIMD3<Float>.zero, +) / Float(pts3D.count)
        let camPos      = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        let dist        = simd_distance(camPos, floorCentre)
        let midX        = (bbox.xMin + bbox.xMax) * 0.5
        let rTop        = worldRayDir(pixel: SIMD2(midX, bbox.yMin), K: K, camT: camT)
        let rBot        = worldRayDir(pixel: SIMD2(midX, bbox.yMax), K: K, camT: camT)
        let cosA        = simd_dot(simd_normalize(rTop), simd_normalize(rBot))
        let angle       = acos(min(max(cosA, -1), 1))
        return dist * tan(angle * 0.5) * 2.0
    }
}
