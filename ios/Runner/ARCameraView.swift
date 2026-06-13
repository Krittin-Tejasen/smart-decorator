//
//  ARCameraView.swift
//  Runner
//
//  Created by W on 13/6/2569 BE.
//

import Flutter
import UIKit
import ARKit

// 1. โรงงานสำหรับสร้าง View ให้ Flutter
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
            binaryMessenger: messenger)
    }
}

// 2. ตัวหน้าจอ ARKit ที่จะถูกส่งไปแสดงใน Flutter
class ARCameraNativeView: NSObject, FlutterPlatformView, ARSCNViewDelegate {
    private var _view: UIView
    private var arView: ARSCNView

    init(
        frame: CGRect,
        viewIdentifier viewId: Int64,
        arguments args: Any?,
        binaryMessenger messenger: FlutterBinaryMessenger
    ) {
        _view = UIView()
        arView = ARSCNView()
        super.init()
        
        arView.frame = frame
        arView.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        
        // กำหนดให้คลาสนี้เป็นคนจัดการ Event ต่างๆ ของกล้อง
        arView.delegate = self
        
        // โชว์จุด Feature Points (Point Cloud) เพื่อให้ผู้ใช้รู้ว่ากำลังสแกนพื้นที่
        arView.debugOptions = [ARSCNDebugOptions.showFeaturePoints]
        
        _view.addSubview(arView)
        
        // ตั้งค่าการสแกนระดับโลก (World Tracking)
        let configuration = ARWorldTrackingConfiguration()
        // สั่งให้สแกนหาทั้งพื้น (horizontal) และกำแพง (vertical)
        configuration.planeDetection = [.horizontal, .vertical]
        
        // เริ่มรันกล้อง
        arView.session.run(configuration)
    }

    func view() -> UIView {
        return _view
    }
}
