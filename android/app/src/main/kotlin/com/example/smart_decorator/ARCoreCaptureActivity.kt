package com.example.smart_decorator

import android.app.Activity
import android.content.Intent
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.opengl.GLES11Ext
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.Gravity
import android.widget.FrameLayout
import android.widget.ImageButton
import com.google.ar.core.*
import com.google.ar.core.exceptions.*
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileOutputStream
import java.time.Instant
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10

/**
 * Full-screen Activity that runs an ARCore session and lets the user capture
 * a single photo with spatial metadata (extrinsics + intrinsics).
 *
 * Returns result via [setResult] / [RESULT_OK] with extras:
 *   [EXTRA_IMAGE_PATH], [EXTRA_EXTRINSICS] (float[16] column-major),
 *   [EXTRA_INTRINSICS] (float[9] row-major), [EXTRA_IMG_WIDTH], [EXTRA_IMG_HEIGHT],
 *   [EXTRA_CAPTURED_AT] (ISO-8601 string).
 */
class ARCoreCaptureActivity : Activity(), GLSurfaceView.Renderer {

    companion object {
        const val EXTRA_IMAGE_PATH  = "imagePath"
        const val EXTRA_EXTRINSICS  = "extrinsics"
        const val EXTRA_INTRINSICS  = "intrinsics"
        const val EXTRA_IMG_WIDTH   = "imageWidth"
        const val EXTRA_IMG_HEIGHT  = "imageHeight"
        const val EXTRA_CAPTURED_AT = "capturedAt"
    }

    private lateinit var surfaceView: GLSurfaceView
    private var session: Session? = null
    private var cameraTextureId = -1

    @Volatile private var pendingCapture = false

    // ── Lifecycle ─────────────────────────────────────────────────────────

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Build UI programmatically — no XML layout dependency
        val root = FrameLayout(this)

        surfaceView = GLSurfaceView(this).apply {
            setEGLContextClientVersion(2)
            setRenderer(this@ARCoreCaptureActivity)
            renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY
        }
        root.addView(surfaceView, FrameLayout.LayoutParams(-1, -1))

        // Shutter button
        val shutterBtn = ImageButton(this).apply {
            setImageResource(android.R.drawable.ic_menu_camera)
            scaleX = 2f; scaleY = 2f
            setBackgroundResource(android.R.drawable.btn_default_small)
            setOnClickListener { pendingCapture = true }
        }
        val lp = FrameLayout.LayoutParams(160, 160).apply {
            gravity = Gravity.BOTTOM or Gravity.CENTER_HORIZONTAL
            bottomMargin = 140
        }
        root.addView(shutterBtn, lp)

        // Close button
        val closeBtn = ImageButton(this).apply {
            setImageResource(android.R.drawable.ic_menu_close_clear_cancel)
            setBackgroundResource(android.R.drawable.btn_default_small)
            setOnClickListener { setResult(RESULT_CANCELED); finish() }
        }
        val closeLp = FrameLayout.LayoutParams(120, 120).apply {
            gravity = Gravity.TOP or Gravity.START
            topMargin = 80; leftMargin = 32
        }
        root.addView(closeBtn, closeLp)

        setContentView(root)
    }

    override fun onResume() {
        super.onResume()
        if (session == null) {
            try {
                if (ArCoreApk.getInstance().requestInstall(this, true) ==
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    return
                }
                session = Session(this).also { s ->
                    s.configure(Config(s).apply {
                        updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
                    })
                }
            } catch (e: Exception) {
                setResult(RESULT_CANCELED)
                finish()
                return
            }
        }
        try { session!!.resume() } catch (e: CameraNotAvailableException) {
            setResult(RESULT_CANCELED); finish(); return
        }
        surfaceView.onResume()
    }

    override fun onPause() {
        super.onPause()
        surfaceView.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        super.onDestroy()
        session?.close()
        session = null
    }

    // ── GLSurfaceView.Renderer ────────────────────────────────────────────

    override fun onSurfaceCreated(gl: GL10, config: EGLConfig) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        val ids = IntArray(1)
        GLES20.glGenTextures(1, ids, 0)
        cameraTextureId = ids[0]
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, cameraTextureId)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_LINEAR)
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES,
            GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_LINEAR)
        session?.setCameraTextureName(cameraTextureId)
    }

    override fun onSurfaceChanged(gl: GL10, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        // Surface rotation 0 = landscape; ARCore requires correct rotation for tracking
        session?.setDisplayGeometry(windowManager.defaultDisplay.rotation, width, height)
    }

    override fun onDrawFrame(gl: GL10) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val sess = session ?: return
        try {
            val frame = sess.update()
            if (pendingCapture) {
                pendingCapture = false
                captureFrame(frame)
            }
        } catch (_: CameraNotAvailableException) { }
    }

    // ── Capture ───────────────────────────────────────────────────────────

    private fun captureFrame(frame: Frame) {
        try {
            val image = frame.acquireCameraImage()
            val jpegBytes = imageToJpeg(image) ?: run {
                image.close()
                runOnUiThread { setResult(RESULT_CANCELED); finish() }
                return
            }
            image.close()

            // Save JPEG
            val dir = getExternalFilesDir(android.os.Environment.DIRECTORY_PICTURES) ?: filesDir
            val file = File(dir, "ar_photo_${System.currentTimeMillis()}.jpg")
            FileOutputStream(file).use { it.write(jpegBytes) }

            val camera = frame.camera

            // Extrinsics: column-major 4×4 (ARCore Pose → matrix)
            val extrinsics = FloatArray(16)
            camera.pose.toMatrix(extrinsics, 0)

            // Intrinsics: row-major 3×3 [fx,0,cx, 0,fy,cy, 0,0,1]
            val ki = camera.imageIntrinsics
            val fl = ki.focalLength       // float[2]: {fx, fy}
            val pp = ki.principalPoint    // float[2]: {cx, cy}
            val dims = ki.imageDimensions // int[2]:   {w, h}
            val intrinsics = floatArrayOf(
                fl[0], 0f, pp[0],
                0f, fl[1], pp[1],
                0f, 0f, 1f
            )

            val capturedAt = Instant.now().toString()

            val data = Intent().apply {
                putExtra(EXTRA_IMAGE_PATH, file.absolutePath)
                putExtra(EXTRA_EXTRINSICS, extrinsics)
                putExtra(EXTRA_INTRINSICS, intrinsics)
                putExtra(EXTRA_IMG_WIDTH, dims[0])
                putExtra(EXTRA_IMG_HEIGHT, dims[1])
                putExtra(EXTRA_CAPTURED_AT, capturedAt)
            }
            runOnUiThread { setResult(RESULT_OK, data); finish() }
        } catch (e: Exception) {
            runOnUiThread { setResult(RESULT_CANCELED); finish() }
        }
    }

    // ── YUV_420_888 → JPEG ───────────────────────────────────────────────

    private fun imageToJpeg(image: android.media.Image): ByteArray? {
        if (image.format != ImageFormat.YUV_420_888) return null

        val w = image.width
        val h = image.height
        val planes = image.planes

        val yPlane = planes[0]; val uPlane = planes[1]; val vPlane = planes[2]
        val yStride = yPlane.rowStride
        val uvStride = uPlane.rowStride
        val uvPixelStride = uPlane.pixelStride

        val yBuf = yPlane.buffer
        val uBuf = uPlane.buffer
        val vBuf = vPlane.buffer

        // Build NV21: Y rows then VU interleaved
        val nv21 = ByteArray(w * h + (w / 2) * (h / 2) * 2)
        var pos = 0

        // Copy Y with row stride
        for (row in 0 until h) {
            yBuf.position(row * yStride)
            yBuf.get(nv21, pos, w)
            pos += w
        }

        // Copy V,U interleaved (NV21 = VU)
        for (row in 0 until h / 2) {
            for (col in 0 until w / 2) {
                val vIdx = row * uvStride + col * uvPixelStride
                val uIdx = row * uvStride + col * uvPixelStride
                vBuf.position(vIdx); nv21[pos++] = vBuf.get()
                uBuf.position(uIdx); nv21[pos++] = uBuf.get()
            }
        }

        val yuvImg = YuvImage(nv21, ImageFormat.NV21, w, h, null)
        val out = ByteArrayOutputStream()
        yuvImg.compressToJpeg(Rect(0, 0, w, h), 85, out)
        return out.toByteArray()
    }
}
