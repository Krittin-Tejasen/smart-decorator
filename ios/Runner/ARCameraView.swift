import Flutter
import UIKit
import ARKit

class ARCameraFactory: NSObject, FlutterPlatformViewFactory {
    private var messenger: FlutterBinaryMessenger

    init(messenger: FlutterBinaryMessenger) {
        self.messenger = messenger
        super.init()
    }

    func create(
        withFrame frame: CGRect,
        viewIdentifier viewId: Int64,
        arguments args: Any?
    ) -> FlutterPlatformView {
        return ARCameraNativeView(
            frame: frame,
            viewIdentifier: viewId,
            arguments: args,
            binaryMessenger: messenger
        )
    }
}

class ARCameraNativeView: NSObject, FlutterPlatformView, ARSCNViewDelegate {
    private let arView: ARSCNView
    private var points: [simd_float3] = []
    private var sceneNodes: [SCNNode] = []

    init(
        frame: CGRect,
        viewIdentifier viewId: Int64,
        arguments args: Any?,
        binaryMessenger messenger: FlutterBinaryMessenger
    ) {
        arView = ARSCNView(frame: frame)
        super.init()

        arView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        arView.delegate = self
        arView.debugOptions = [ARSCNDebugOptions.showFeaturePoints]

        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = [.horizontal, .vertical]
        arView.session.run(configuration)

        let channel = FlutterMethodChannel(
            name: "com.smartdeco.app/ar_camera_\(viewId)",
            binaryMessenger: messenger
        )
        channel.setMethodCallHandler { [weak self] (call, result) in
            guard let self = self else { return }
            switch call.method {
            case "addPoint":
                self.handleAddPoint(result: result)
            case "clearPoints":
                self.clearAll()
                result(nil)
            default:
                result(FlutterMethodNotImplemented)
            }
        }
    }

    func view() -> UIView { return arView }

    private func clearAll() {
        points.removeAll()
        sceneNodes.forEach { $0.removeFromParentNode() }
        sceneNodes.removeAll()
    }

    private func handleAddPoint(result: FlutterResult) {
        guard let frame = arView.session.currentFrame else {
            result(["error": "no_frame"])
            return
        }

        let screenCenter = CGPoint(x: arView.bounds.midX, y: arView.bounds.midY)
        let query = arView.raycastQuery(from: screenCenter, allowing: .estimatedPlane, alignment: .any)

        let position: simd_float3
        if let query = query, let hit = arView.session.raycast(query).first {
            let col = hit.worldTransform.columns.3
            position = simd_make_float3(col.x, col.y, col.z)
        } else {
            let col = frame.camera.transform.columns.3
            position = simd_make_float3(col.x, col.y, col.z)
        }

        // Yellow sphere marker at corner
        let sphere = SCNSphere(radius: 0.02)
        sphere.firstMaterial?.diffuse.contents = UIColor.yellow
        sphere.firstMaterial?.emission.contents = UIColor.yellow.withAlphaComponent(0.4)
        let sphereNode = SCNNode(geometry: sphere)
        sphereNode.position = SCNVector3(position.x, position.y, position.z)
        arView.scene.rootNode.addChildNode(sphereNode)
        sceneNodes.append(sphereNode)

        // Cyan line from previous corner to this one
        if let prev = points.last {
            let lineNode = makeLine(from: prev, to: position)
            arView.scene.rootNode.addChildNode(lineNode)
            sceneNodes.append(lineNode)
        }

        points.append(position)

        var distance: Float = 0
        if points.count >= 2 {
            let d = points.last! - points[points.count - 2]
            distance = simd_length(d)
        }

        let allPoints = points.map { ["x": Double($0.x), "y": Double($0.y), "z": Double($0.z)] }
        result(["points": points.count, "distance": Double(distance), "allPoints": allPoints])
    }

    private func makeLine(from start: simd_float3, to end: simd_float3) -> SCNNode {
        let diff = end - start
        let length = max(simd_length(diff), 0.001)

        let cylinder = SCNCylinder(radius: 0.006, height: CGFloat(length))
        cylinder.firstMaterial?.diffuse.contents = UIColor.cyan
        cylinder.firstMaterial?.emission.contents = UIColor.cyan.withAlphaComponent(0.5)

        let node = SCNNode(geometry: cylinder)
        let mid = (start + end) * 0.5
        node.position = SCNVector3(mid.x, mid.y, mid.z)

        // Rotate SCNCylinder (Y-axis) to align with diff direction
        let yAxis = simd_float3(0, 1, 0)
        let dir = simd_normalize(diff)
        let cross = simd_cross(yAxis, dir)
        let crossLen = simd_length(cross)

        if crossLen < 0.001 {
            if simd_dot(yAxis, dir) < 0 {
                node.eulerAngles = SCNVector3(Float.pi, 0, 0)
            }
        } else {
            let angle = acos(min(max(simd_dot(yAxis, dir), -1.0), 1.0))
            let axis = cross / crossLen
            node.simdRotation = simd_float4(axis.x, axis.y, axis.z, angle)
        }

        return node
    }
}
