import ARKit
import UIKit
import simd

// MARK: - Plane (standard AR fallback)

struct CapturedPlane: Codable {
    let identifier: String
    let alignment: String          // "horizontal" | "vertical"
    let normalX, normalY, normalZ: Float
    let distanceFromOrigin: Float
    let centerX, centerY, centerZ: Float
    let extentW, extentH: Float

    init(anchor: ARPlaneAnchor) {
        identifier = anchor.identifier.uuidString
        switch anchor.alignment {
        case .horizontal: alignment = "horizontal"
        case .vertical:   alignment = "vertical"
        @unknown default: alignment = "unknown"
        }
        let n = anchor.transform.columns.1
        normalX = n.x; normalY = n.y; normalZ = n.z
        let c = anchor.transform.columns.3
        centerX = c.x; centerY = c.y; centerZ = c.z
        distanceFromOrigin = simd_dot(SIMD3(normalX,normalY,normalZ), SIMD3(centerX,centerY,centerZ))
        if #available(iOS 16.0, *) {
            extentW = anchor.planeExtent.width
            extentH = anchor.planeExtent.height
        } else {
            extentW = anchor.extent.x
            extentH = anchor.extent.z
        }
    }
    var normal: SIMD3<Float> { SIMD3(normalX, normalY, normalZ) }
    var center: SIMD3<Float> { SIMD3(centerX, centerY, centerZ) }
}

// MARK: - Camera intrinsics

struct CapturedIntrinsics: Codable {
    let fx, fy, cx, cy: Float
    let imageWidth, imageHeight: Int

    // ARCamera.intrinsics is column-major:
    //   col0=(fx,0,0)  col1=(0,fy,0)  col2=(cx,cy,1)
    init(camera: ARCamera) {
        let K = camera.intrinsics
        fx = K.columns.0.x; fy = K.columns.1.y
        cx = K.columns.2.x; cy = K.columns.2.y
        let r = camera.imageResolution
        imageWidth = Int(r.width); imageHeight = Int(r.height)
    }
}

// MARK: - Camera extrinsics (4×4 column-major)

struct CapturedExtrinsics: Codable {
    let m: [Float]   // 16 floats, columns 0-3

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

// MARK: - LiDAR mesh anchor

struct CapturedMeshAnchor: Codable {
    let identifier: String
    let transformM: [Float]         // 16 floats, column-major world transform
    let vertices: [Float]           // flat x0,y0,z0, x1,y1,z1 …  (local space)
    let faceIndices: [Int]          // flat i0,j0,k0, i1,j1,k1 …  (triangles)
    let classifications: [Int]      // per-face: 0=none 1=wall 2=floor 3=ceiling
                                    //           4=table 5=seat 6=window 7=door

    @available(iOS 13.4, *)
    init(anchor: ARMeshAnchor, maxVertices: Int = 2000) {
        identifier = anchor.identifier.uuidString
        let t = anchor.transform
        transformM = [
            t.columns.0.x, t.columns.0.y, t.columns.0.z, t.columns.0.w,
            t.columns.1.x, t.columns.1.y, t.columns.1.z, t.columns.1.w,
            t.columns.2.x, t.columns.2.y, t.columns.2.z, t.columns.2.w,
            t.columns.3.x, t.columns.3.y, t.columns.3.z, t.columns.3.w,
        ]

        // Vertices
        let vSrc = anchor.geometry.vertices
        let vPtr = vSrc.buffer.contents().advanced(by: vSrc.offset)
        let vCount = min(vSrc.count, maxVertices)
        var verts = [Float]()
        verts.reserveCapacity(vCount * 3)
        for i in 0 ..< vCount {
            let p = vPtr.advanced(by: i * vSrc.stride).assumingMemoryBound(to: Float.self)
            verts.append(p[0]); verts.append(p[1]); verts.append(p[2])
        }
        vertices = verts

        // Faces — only keep triangles whose vertex indices are in range
        let fSrc = anchor.geometry.faces
        let fPtr = fSrc.buffer.contents()
        let fCount = fSrc.count
        var idxs = [Int]()
        idxs.reserveCapacity(fCount * 3)
        for i in 0 ..< fCount {
            var valid = true
            var tri = [Int](repeating: 0, count: 3)
            for j in 0 ..< 3 {
                let byteOff = i * fSrc.indexCountPerPrimitive * fSrc.bytesPerIndex
                              + j * fSrc.bytesPerIndex
                let idx: Int
                if fSrc.bytesPerIndex == 2 {
                    idx = Int(fPtr.advanced(by: byteOff).load(as: UInt16.self))
                } else {
                    idx = Int(fPtr.advanced(by: byteOff).load(as: UInt32.self))
                }
                if idx >= vCount { valid = false; break }
                tri[j] = idx
            }
            if valid { idxs.append(contentsOf: tri) }
        }
        faceIndices = idxs

        // Per-face semantic classification
        var cls = [Int]()
        cls.reserveCapacity(fCount)
        if let clsSrc = anchor.geometry.classification {
            let clsPtr = clsSrc.buffer.contents().advanced(by: clsSrc.offset)
            for i in 0 ..< fCount {
                let raw = clsPtr.advanced(by: i * clsSrc.stride).load(as: UInt8.self)
                cls.append(Int(raw))
            }
        }
        classifications = cls
    }
}

// MARK: - Top-level snapshot

struct RoomCaptureState: Codable {
    // Common
    let capturedAtISO: String
    let captureMode:   String       // "lidar" | "standard"
    let extrinsics:    CapturedExtrinsics
    let intrinsics:    CapturedIntrinsics

    // Standard AR fallback
    let planes: [CapturedPlane]

    // LiDAR — nil on non-LiDAR devices
    let depthMapPng:      String?   // base64 greyscale PNG, closer=brighter
    let depthMinMeters:   Float?    // actual depth at pixel value 255
    let depthMaxMeters:   Float?    // actual depth at pixel value 0
    let confidenceMapPng: String?   // base64 greyscale PNG: 0=low,128=med,255=high
    let meshAnchors:      [CapturedMeshAnchor]?

    var floorPlane: CapturedPlane? {
        planes.filter { $0.alignment == "horizontal" }
              .min(by: { $0.centerY < $1.centerY })
    }

    // MARK: - Init from ARFrame

    init(frame: ARFrame) {
        capturedAtISO = ISO8601DateFormatter().string(from: Date())
        captureMode   = ARCapabilityService.captureMode
        extrinsics    = CapturedExtrinsics(transform: frame.camera.transform)
        intrinsics    = CapturedIntrinsics(camera: frame.camera)
        planes        = frame.anchors.compactMap { $0 as? ARPlaneAnchor }
                                     .map        { CapturedPlane(anchor: $0) }

        if #available(iOS 14.0, *), ARCapabilityService.supportsSceneDepth,
           let depth = frame.smoothedSceneDepth ?? frame.sceneDepth {
            let result = Self.extractDepthPNG(from: depth.depthMap)
            depthMapPng    = result?.png
            depthMinMeters = result?.minM
            depthMaxMeters = result?.maxM
            confidenceMapPng = depth.confidenceMap.flatMap { Self.extractConfidencePNG(from: $0) }
        } else {
            depthMapPng    = nil
            depthMinMeters = nil
            depthMaxMeters = nil
            confidenceMapPng = nil
        }

        if #available(iOS 13.4, *), ARCapabilityService.hasLiDAR {
            meshAnchors = frame.anchors.compactMap { $0 as? ARMeshAnchor }
                                       .map        { CapturedMeshAnchor(anchor: $0) }
        } else {
            meshAnchors = nil
        }
    }

    // MARK: - Serialisation

    func toJSON() throws -> Data {
        let enc = JSONEncoder()
        enc.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try enc.encode(self)
    }

    static func from(jsonData: Data) throws -> RoomCaptureState {
        try JSONDecoder().decode(RoomCaptureState.self, from: jsonData)
    }

    func toFlutterMap() throws -> [String: Any] {
        let data = try toJSON()
        return (try JSONSerialization.jsonObject(with: data) as? [String: Any]) ?? [:]
    }

    // MARK: - Depth / confidence PNG helpers

    private static func extractDepthPNG(
        from pixelBuffer: CVPixelBuffer
    ) -> (png: String, minM: Float, maxM: Float)? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let W = CVPixelBufferGetWidth(pixelBuffer)
        let H = CVPixelBufferGetHeight(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let floats = base.assumingMemoryBound(to: Float32.self)

        var minD: Float = .infinity, maxD: Float = 0
        for i in 0 ..< W * H {
            let v = floats[i]
            guard v.isFinite, v > 0 else { continue }
            if v < minD { minD = v }
            if v > maxD { maxD = v }
        }
        guard minD < .infinity else { return nil }
        let range = max(maxD - minD, 1e-4)

        var pixels = [UInt8](repeating: 0, count: W * H)
        for i in 0 ..< W * H {
            let v = floats[i]
            guard v.isFinite, v > 0 else { continue }
            let norm = 1.0 - (v - minD) / range   // closer = brighter
            pixels[i] = UInt8(min(max(norm * 255, 0), 255))
        }
        guard let img = makeGreyImage(pixels: pixels, w: W, h: H),
              let pngData = img.pngData() else { return nil }
        return (pngData.base64EncodedString(), minD, maxD)
    }

    private static func extractConfidencePNG(from pixelBuffer: CVPixelBuffer) -> String? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        let W = CVPixelBufferGetWidth(pixelBuffer)
        let H = CVPixelBufferGetHeight(pixelBuffer)
        guard let base = CVPixelBufferGetBaseAddress(pixelBuffer) else { return nil }
        let bytes = base.assumingMemoryBound(to: UInt8.self)

        // Map 0/1/2 → 0/128/255
        let lut: [UInt8] = [0, 128, 255]
        var pixels = [UInt8](repeating: 0, count: W * H)
        for i in 0 ..< W * H {
            let v = min(Int(bytes[i]), 2)
            pixels[i] = lut[v]
        }
        guard let img = makeGreyImage(pixels: pixels, w: W, h: H),
              let pngData = img.pngData() else { return nil }
        return pngData.base64EncodedString()
    }

    private static func makeGreyImage(pixels: [UInt8], w: Int, h: Int) -> UIImage? {
        guard let ctx = CGContext(
            data: nil, width: w, height: h,
            bitsPerComponent: 8, bytesPerRow: w,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        ), let ptr = ctx.data else { return nil }
        pixels.withUnsafeBytes { ptr.copyMemory(from: $0.baseAddress!, byteCount: pixels.count) }
        return ctx.makeImage().map { UIImage(cgImage: $0) }
    }
}
