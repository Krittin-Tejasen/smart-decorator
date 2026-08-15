import ARKit
import simd

// MARK: - Plane snapshot

struct CapturedPlane: Codable {
    let identifier: String
    let alignment: String         // "horizontal" | "vertical"
    let normalX: Float            // world-space normal
    let normalY: Float
    let normalZ: Float
    let distanceFromOrigin: Float // d in plane equation n·p = d
    let centerX: Float            // world-space anchor centre
    let centerY: Float
    let centerZ: Float
    let extentW: Float            // estimated plane width (metres)
    let extentH: Float            // estimated plane height (metres)

    init(anchor: ARPlaneAnchor) {
        identifier = anchor.identifier.uuidString
        switch anchor.alignment {
        case .horizontal: alignment = "horizontal"
        case .vertical:   alignment = "vertical"
        @unknown default: alignment = "unknown"
        }
        // ARPlaneAnchor's local Y-axis is the plane normal in world space
        let col1 = anchor.transform.columns.1
        normalX = col1.x; normalY = col1.y; normalZ = col1.z

        let col3 = anchor.transform.columns.3
        centerX = col3.x; centerY = col3.y; centerZ = col3.z

        distanceFromOrigin = simd_dot(
            SIMD3(normalX, normalY, normalZ),
            SIMD3(centerX, centerY, centerZ)
        )
        extentW = anchor.planeExtent.width
        extentH = anchor.planeExtent.height
    }

    var normal: SIMD3<Float> { SIMD3(normalX, normalY, normalZ) }
    var center: SIMD3<Float> { SIMD3(centerX, centerY, centerZ) }
}

// MARK: - Camera intrinsics snapshot

struct CapturedIntrinsics: Codable {
    let fx: Float          // focal length x (pixels)
    let fy: Float          // focal length y (pixels)
    let cx: Float          // principal point x (pixels)
    let cy: Float          // principal point y (pixels)
    let imageWidth: Int
    let imageHeight: Int

    // ARCamera.intrinsics is column-major simd_float3x3:
    //   col 0 = (fx, 0, 0), col 1 = (0, fy, 0), col 2 = (cx, cy, 1)
    init(camera: ARCamera) {
        let K = camera.intrinsics
        fx = K.columns.0.x
        fy = K.columns.1.y
        cx = K.columns.2.x
        cy = K.columns.2.y
        let res = camera.imageResolution
        imageWidth  = Int(res.width)
        imageHeight = Int(res.height)
    }
}

// MARK: - Camera extrinsics snapshot (4×4 column-major)

struct CapturedExtrinsics: Codable {
    /// 16 floats stored column-major: columns 0-3 concatenated
    let m: [Float]

    init(transform: simd_float4x4) {
        m = [
            transform.columns.0.x, transform.columns.0.y,
            transform.columns.0.z, transform.columns.0.w,
            transform.columns.1.x, transform.columns.1.y,
            transform.columns.1.z, transform.columns.1.w,
            transform.columns.2.x, transform.columns.2.y,
            transform.columns.2.z, transform.columns.2.w,
            transform.columns.3.x, transform.columns.3.y,
            transform.columns.3.z, transform.columns.3.w,
        ]
    }

    var matrix: simd_float4x4 {
        simd_float4x4(columns: (
            SIMD4(m[0],  m[1],  m[2],  m[3]),
            SIMD4(m[4],  m[5],  m[6],  m[7]),
            SIMD4(m[8],  m[9],  m[10], m[11]),
            SIMD4(m[12], m[13], m[14], m[15])
        ))
    }
}

// MARK: - Top-level snapshot

struct RoomCaptureState: Codable {
    let capturedAtISO: String
    let extrinsics: CapturedExtrinsics
    let intrinsics: CapturedIntrinsics
    let planes: [CapturedPlane]

    /// Lowest horizontal plane — best candidate for the floor
    var floorPlane: CapturedPlane? {
        planes
            .filter { $0.alignment == "horizontal" }
            .min(by: { $0.centerY < $1.centerY })
    }

    init(frame: ARFrame) {
        capturedAtISO = ISO8601DateFormatter().string(from: Date())
        extrinsics    = CapturedExtrinsics(transform: frame.camera.transform)
        intrinsics    = CapturedIntrinsics(camera: frame.camera)
        planes        = frame.anchors
            .compactMap { $0 as? ARPlaneAnchor }
            .map        { CapturedPlane(anchor: $0) }
    }

    // MARK: Serialisation

    func toJSON() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(self)
    }

    static func from(jsonData: Data) throws -> RoomCaptureState {
        try JSONDecoder().decode(RoomCaptureState.self, from: jsonData)
    }

    /// Returns a `[String: Any]` dict that can be handed directly to a Flutter MethodChannel result.
    func toFlutterMap() throws -> [String: Any] {
        let data = try toJSON()
        return try (JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }
}
