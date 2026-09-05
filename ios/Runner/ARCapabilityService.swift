import ARKit

#if canImport(RoomPlan)
import RoomPlan
#endif

// Single source of truth for device AR capability detection.
enum ARCapabilityService {

    /// True on iPhone 12 Pro / 13 Pro / 14 Pro / 15 Pro / iPad Pro 2020+
    static var hasLiDAR: Bool {
        if #available(iOS 13.4, *) {
            return ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh)
        }
        return false
    }

    /// Requires LiDAR AND iOS 14+
    static var supportsSceneDepth: Bool {
        if #available(iOS 14.0, *) {
            return ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)
        }
        return false
    }

    /// RoomPlan requires LiDAR AND iOS 16+
    static var isRoomPlanSupported: Bool {
        #if canImport(RoomPlan)
        if #available(iOS 16.0, *) {
            return RoomCaptureSession.isSupported
        }
        #endif
        return false
    }

    /// "lidar" | "standard"
    static var captureMode: String { hasLiDAR ? "lidar" : "standard" }

    static func toFlutterMap() -> [String: Any] {
        [
            "hasLiDAR":            hasLiDAR,
            "supportsSceneDepth":  supportsSceneDepth,
            "isRoomPlanSupported": isRoomPlanSupported,
            "captureMode":         captureMode,
        ]
    }
}
