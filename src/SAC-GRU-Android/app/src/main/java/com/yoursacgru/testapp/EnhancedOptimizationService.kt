package com.yoursacgru.testapp

import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.util.Log
import kotlinx.coroutines.*
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.flex.FlexDelegate
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class EnhancedOptimizationService : Service() {

    companion object {
        private const val TAG = "EnhancedOptService"
        private const val MODEL_FILE = "sac_actor_model.tflite"
        // ✅ Updated monitoring interval to 10 seconds
        private const val MONITORING_INTERVAL_MS = 10000L
    }

    private val serviceScope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var actorInterpreter: Interpreter? = null
    private var flexDelegate: FlexDelegate? = null
    private lateinit var trafficMonitor: TrafficMonitor

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        QoSOptimizer.initialize(this)
        trafficMonitor = TrafficMonitor(this)
        serviceScope.launch {
            if (loadModel()) {
                startTrafficMonitoring()
            }
        }
    }

    private fun startTrafficMonitoring() {
        serviceScope.launch {
            Log.i(TAG, "Starting real-time traffic monitoring...")
            while (isActive) {
                val (bpsRx, _, activeApp) = trafficMonitor.sampleTraffic()
                val isSocialApp = activeApp.contains("youtube") || activeApp.contains("instagram")

                // ✅ Only process traffic if it's a known social media app.
                if (isSocialApp) {
                    val features = createFeaturesFromRealData(bpsRx)
                    val result = runInference(features)
                    result?.let {
                        QoSOptimizer.applyOptimizationPolicy(it, activeApp, bpsRx)
                    }
                }

                delay(MONITORING_INTERVAL_MS)
            }
        }
    }

    private fun createFeaturesFromRealData(bpsRx: Float): FloatArray {
        // This logic is now simpler because we already know it's a social app.
        val REEL_BANDWIDTH_THRESHOLD = 50 * 1024 // 50 KB/s
        if (bpsRx > REEL_BANDWIDTH_THRESHOLD) {
            Log.d(TAG, "Sufficient bandwidth for social app. Generating REEL features.")
            return floatArrayOf(0.25f, 0.8f, 0.9f, 0.0f, 1.0f, 0.0f, 0.9f)
        } else {
            Log.d(TAG, "Low bandwidth for social app. Generating NON-REEL features.")
            return floatArrayOf(0.8f, 0.1f, 0.1f, 0.0f, 0.0f, 0.0f, 0.1f)
        }
    }

    private fun runInference(features: FloatArray): PredictionResult? {
        if (actorInterpreter == null || features.size != 7) return null
        val inputArray = arrayOf(arrayOf(features))
        val outputArray = Array(1) { FloatArray(2) }
        try {
            actorInterpreter?.run(inputArray, outputArray)
            val nonReelProb = outputArray[0][0]
            val reelProb = outputArray[0][1]
            val label = if (reelProb > nonReelProb) "REEL" else "NON-REEL"
            val confidence = if (label == "REEL") reelProb else nonReelProb
            return PredictionResult(label, confidence)
        } catch (e: Exception) {
            Log.e(TAG, "Error during background inference: ${e.message}")
            return null
        }
    }

    // ... (rest of the service is unchanged) ...
    private fun loadModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile()
            flexDelegate = FlexDelegate()
            val options = Interpreter.Options().apply { addDelegate(flexDelegate!!) }
            actorInterpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "Model loaded successfully.")
            true
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model: ${e.message}")
            false
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        serviceScope.cancel()
        actorInterpreter?.close()
        flexDelegate?.close()
        Log.i(TAG, "Optimization service stopped.")
    }
}
