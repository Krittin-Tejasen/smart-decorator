import Foundation
import simd

// MARK: - Public types

/// 2-D axis-aligned bounding box in image-pixel coordinates (top-left origin).
struct BBox2D {
    let xMin: Float
    let yMin: Float
    let xMax: Float
    let yMax: Float
}

/// Real-world physical dimensions of a detected furniture item.
struct FurnitureSize: Codable {
    let widthMeters: Float
    let heightMeters: Float
    let depthMeters: Float
}

// MARK: - Calculator

enum SizeCalculator {

    /// Measures the physical size of a piece of furniture from its 2-D bounding box.
    ///
    /// **How it works:**
    /// 1. The four corners of the bbox are unprojected into world-space rays using the
    ///    saved camera intrinsics and extrinsics.
    /// 2. Each ray is intersected with the detected floor plane to obtain 3-D floor points.
    /// 3. Width  = average of the top-edge and bottom-edge horizontal spans on the floor.
    ///    Depth  = average of the left-edge and right-edge depth spans on the floor.
    ///    Height = angular height of the bbox × distance from camera to floor centre.
    ///
    /// - Parameters:
    ///   - bbox:  2-D bbox from the AI segmentation result, in **image-pixel** coordinates.
    ///   - state: ARKit snapshot taken at capture time.
    /// - Returns: Physical size, or `nil` when no floor plane was detected.
    static func measure(bbox: BBox2D, in state: RoomCaptureState) -> FurnitureSize? {
        guard let floor = state.floorPlane else { return nil }

        let K    = state.intrinsics
        let camT = state.extrinsics.matrix

        // Bottom edge → floor footprint width
        // Top edge    → far-end footprint width
        let botL = SIMD2<Float>(bbox.xMin, bbox.yMax)
        let botR = SIMD2<Float>(bbox.xMax, bbox.yMax)
        let topL = SIMD2<Float>(bbox.xMin, bbox.yMin)
        let topR = SIMD2<Float>(bbox.xMax, bbox.yMin)

        guard
            let p_botL = floorPoint(pixel: botL, K: K, camT: camT, floor: floor),
            let p_botR = floorPoint(pixel: botR, K: K, camT: camT, floor: floor),
            let p_topL = floorPoint(pixel: topL, K: K, camT: camT, floor: floor),
            let p_topR = floorPoint(pixel: topR, K: K, camT: camT, floor: floor)
        else { return nil }

        let widthM = (simd_distance(p_botL, p_botR) + simd_distance(p_topL, p_topR)) * 0.5
        let depthM = (simd_distance(p_botL, p_topL) + simd_distance(p_botR, p_topR)) * 0.5

        // Height: use vertical angular span of bbox and camera→floor-centre distance
        let floorCentre = (p_botL + p_botR + p_topL + p_topR) * 0.25
        let camPos      = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        let dist        = simd_distance(camPos, floorCentre)

        let midX    = (bbox.xMin + bbox.xMax) * 0.5
        let rayTop  = worldRayDir(pixel: SIMD2(midX, bbox.yMin), K: K, camT: camT)
        let rayBot  = worldRayDir(pixel: SIMD2(midX, bbox.yMax), K: K, camT: camT)
        let cosAng  = simd_dot(simd_normalize(rayTop), simd_normalize(rayBot))
        let angle   = acos(min(max(cosAng, -1), 1))
        let heightM = dist * tan(angle * 0.5) * 2.0

        return FurnitureSize(
            widthMeters:  max(widthM,  0),
            heightMeters: max(heightM, 0),
            depthMeters:  max(depthM,  0)
        )
    }

    // MARK: - Private helpers

    /// Converts a pixel coordinate to a normalised world-space ray direction.
    ///
    /// ARKit's camera coordinate system: +X right, +Y up, **-Z forward**.
    /// The unprojection formula is:
    ///   dir_cam = ((u - cx)/fx,  (v - cy)/fy,  -1)   // z = -1 because forward = -Z
    ///   dir_world = R_cam * dir_cam               // R_cam = upper-3×3 of camT
    private static func worldRayDir(
        pixel: SIMD2<Float>,
        K: CapturedIntrinsics,
        camT: simd_float4x4
    ) -> SIMD3<Float> {
        let dirCam = SIMD3<Float>(
            (pixel.x - K.cx) / K.fx,
            (pixel.y - K.cy) / K.fy,
            -1.0                       // ARKit: camera looks along -Z
        )
        let R = simd_float3x3(
            SIMD3(camT.columns.0.x, camT.columns.0.y, camT.columns.0.z),
            SIMD3(camT.columns.1.x, camT.columns.1.y, camT.columns.1.z),
            SIMD3(camT.columns.2.x, camT.columns.2.y, camT.columns.2.z)
        )
        return simd_normalize(R * dirCam)
    }

    /// Finds where a pixel ray hits the floor plane.
    ///
    /// Plane equation: n · p = d
    /// Ray equation:   p = origin + t * dir
    /// Solution:       t = (d - n·origin) / (n·dir)
    private static func floorPoint(
        pixel: SIMD2<Float>,
        K: CapturedIntrinsics,
        camT: simd_float4x4,
        floor: CapturedPlane
    ) -> SIMD3<Float>? {
        let origin = SIMD3<Float>(camT.columns.3.x, camT.columns.3.y, camT.columns.3.z)
        let dir    = worldRayDir(pixel: pixel, K: K, camT: camT)
        let n      = floor.normal
        let d      = floor.distanceFromOrigin

        let denom = simd_dot(n, dir)
        guard abs(denom) > 1e-6 else { return nil }   // ray parallel to floor

        let t = (d - simd_dot(n, origin)) / denom
        guard t > 0 else { return nil }               // floor is behind camera

        return origin + dir * t
    }
}
