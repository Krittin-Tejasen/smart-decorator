package com.example.smart_decorator

import android.app.Activity
import android.content.Intent
import io.flutter.embedding.engine.plugins.FlutterPlugin
import io.flutter.embedding.engine.plugins.activity.ActivityAware
import io.flutter.embedding.engine.plugins.activity.ActivityPluginBinding
import io.flutter.plugin.common.MethodCall
import io.flutter.plugin.common.MethodChannel
import io.flutter.plugin.common.PluginRegistry

/** Handles `com.smartdeco.app/ar_capture` — starts [ARCoreCaptureActivity] and returns result. */
class ARCoreCapturePlugin :
    FlutterPlugin,
    ActivityAware,
    MethodChannel.MethodCallHandler,
    PluginRegistry.ActivityResultListener {

    companion object {
        private const val CHANNEL = "com.smartdeco.app/ar_capture"
        private const val REQUEST_CODE = 8891
    }

    private lateinit var channel: MethodChannel
    private var activity: Activity? = null
    private var activityBinding: ActivityPluginBinding? = null
    private var pendingResult: MethodChannel.Result? = null

    // ── FlutterPlugin ─────────────────────────────────────────────────────

    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel = MethodChannel(binding.binaryMessenger, CHANNEL)
        channel.setMethodCallHandler(this)
    }

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {
        channel.setMethodCallHandler(null)
    }

    // ── ActivityAware ─────────────────────────────────────────────────────

    override fun onAttachedToActivity(binding: ActivityPluginBinding) {
        activity = binding.activity
        binding.addActivityResultListener(this)
        activityBinding = binding
    }

    override fun onDetachedFromActivityForConfigChanges() = onDetachedFromActivity()

    override fun onReattachedToActivityForConfigChanges(binding: ActivityPluginBinding) =
        onAttachedToActivity(binding)

    override fun onDetachedFromActivity() {
        activityBinding?.removeActivityResultListener(this)
        activityBinding = null
        activity = null
    }

    // ── MethodCallHandler ─────────────────────────────────────────────────

    override fun onMethodCall(call: MethodCall, result: MethodChannel.Result) {
        when (call.method) {
            "capturePhoto" -> {
                val act = activity
                if (act == null) {
                    result.error("NO_ACTIVITY", "No active Android Activity", null)
                    return
                }
                pendingResult = result
                act.startActivityForResult(
                    Intent(act, ARCoreCaptureActivity::class.java),
                    REQUEST_CODE
                )
            }
            else -> result.notImplemented()
        }
    }

    // ── ActivityResultListener ────────────────────────────────────────────

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?): Boolean {
        if (requestCode != REQUEST_CODE) return false

        val pr = pendingResult ?: return true
        pendingResult = null

        if (resultCode != Activity.RESULT_OK || data == null) {
            pr.error("CANCELLED", "User cancelled AR capture", null)
            return true
        }

        val imagePath = data.getStringExtra(ARCoreCaptureActivity.EXTRA_IMAGE_PATH)
        if (imagePath == null) {
            pr.error("NO_DATA", "No image path in result", null)
            return true
        }

        val extrRaw = data.getFloatArrayExtra(ARCoreCaptureActivity.EXTRA_EXTRINSICS) ?: FloatArray(16)
        val intrRaw = data.getFloatArrayExtra(ARCoreCaptureActivity.EXTRA_INTRINSICS) ?: FloatArray(9)
        val imgW = data.getIntExtra(ARCoreCaptureActivity.EXTRA_IMG_WIDTH, 0)
        val imgH = data.getIntExtra(ARCoreCaptureActivity.EXTRA_IMG_HEIGHT, 0)
        val capturedAt = data.getStringExtra(ARCoreCaptureActivity.EXTRA_CAPTURED_AT) ?: ""

        pr.success(
            mapOf(
                "imagePath"   to imagePath,
                "extrinsics"  to extrRaw.map { it.toDouble() },
                "intrinsics"  to intrRaw.map { it.toDouble() },
                "imageWidth"  to imgW,
                "imageHeight" to imgH,
                "platform"    to "android",
                "capturedAt"  to capturedAt,
            )
        )
        return true
    }
}
