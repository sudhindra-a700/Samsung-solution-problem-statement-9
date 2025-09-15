package com.yoursacgru.testapp

import android.content.Context
import android.os.PowerManager
import android.util.Log
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkManager
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import java.util.concurrent.TimeUnit

object QoSOptimizer {
    private const val TAG = "QoSOptimizer"
    private const val WAKE_LOCK_TAG = "VULCAN::PerformanceWakeLock"
    private const val HISTORY_SIZE = 10 // Store the last 10 events for visualization

    // This StateFlow now holds a list of the last 10 status updates.
    private val _statusHistory = MutableStateFlow<List<LiveStatus>>(listOf(LiveStatus("Monitoring", null)))
    val statusHistory = _statusHistory.asStateFlow()

    private var powerManager: PowerManager? = null
    private var workManager: WorkManager? = null
    private var wakeLock: PowerManager.WakeLock? = null

    fun initialize(context: Context) {
        powerManager = context.getSystemService(Context.POWER_SERVICE) as PowerManager
        workManager = WorkManager.getInstance(context)
    }

    fun applyOptimizationPolicy(prediction: PredictionResult, appName: String = "", bandwidthBps: Float) {
        val actionText = when (prediction.label) {
            "REEL" -> handleReelTraffic(prediction.confidence, appName)
            "NON-REEL" -> handleNonReelTraffic(appName)
            else -> applyDefaultPolicy()
        }

        // Create a new status event with all the necessary data.
        val newStatus = LiveStatus(actionText, prediction, bandwidthBps)

        // Add the new status to the front of the list and keep only the last `HISTORY_SIZE` items.
        _statusHistory.value = (listOf(newStatus) + _statusHistory.value).take(HISTORY_SIZE)

        Log.i(TAG, "Final Action Taken: $actionText")
    }

    fun resetState() {
        resetCpuPerformance()
        _statusHistory.value = listOf(LiveStatus("Monitoring", null))
        Log.i(TAG, "QoS Optimizer state reset to Monitoring.")
    }

    private fun handleReelTraffic(confidence: Float, appName: String): String {
        return if (confidence > 0.85f) {
            boostCpuPerformance()
            "Performance Boosted for $appName"
        } else {
            resetCpuPerformance()
            "Balanced Policy for $appName"
        }
    }

    private fun handleNonReelTraffic(appName: String): String {
        resetCpuPerformance()
        deferNonCriticalBackgroundTasks()
        return "Power Saving Mode for $appName"
    }

    private fun applyDefaultPolicy(): String {
        resetCpuPerformance()
        return "Applying Balanced Policy"
    }

    // ... (rest of the optimization functions are unchanged)
    private fun boostCpuPerformance() {
        if (wakeLock?.isHeld == false || wakeLock == null) {
            wakeLock = powerManager?.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, WAKE_LOCK_TAG)?.apply { acquire(10 * 60 * 1000L) }
            Log.i(TAG, "REAL ACTION: Acquired CPU WakeLock.")
        }
    }

    private fun deferNonCriticalBackgroundTasks() {
        val deferredSyncWork = OneTimeWorkRequestBuilder<DeferredSyncWorker>().build()
        workManager?.enqueue(deferredSyncWork)
        Log.i(TAG, "REAL ACTION: Deferred non-critical tasks.")
    }

    private fun resetCpuPerformance() {
        if (wakeLock?.isHeld == true) {
            wakeLock?.release()
            wakeLock = null
            Log.i(TAG, "REAL ACTION: Released CPU WakeLock.")
        }
    }
}
