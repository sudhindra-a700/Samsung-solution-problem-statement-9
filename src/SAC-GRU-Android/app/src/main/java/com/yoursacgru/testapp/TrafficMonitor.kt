package com.yoursacgru.testapp

import android.app.usage.UsageStatsManager
import android.content.Context
import android.net.TrafficStats
import android.util.Log

class TrafficMonitor(context: Context) {

    private val usageStatsManager = context.getSystemService(Context.USAGE_STATS_SERVICE) as UsageStatsManager
    private var lastBytesRx: Long = 0
    private var lastBytesTx: Long = 0
    private var lastTimestamp: Long = 0

    init {
        lastBytesRx = TrafficStats.getTotalRxBytes()
        lastBytesTx = TrafficStats.getTotalTxBytes()
        lastTimestamp = System.currentTimeMillis()
    }

    fun sampleTraffic(): Triple<Float, Float, String> {
        val currentTimestamp = System.currentTimeMillis()
        val elapsedSeconds = (currentTimestamp - lastTimestamp) / 1000f

        if (elapsedSeconds <= 0.1f) {
            return Triple(0f, 0f, "unknown")
        }

        val currentBytesRx = TrafficStats.getTotalRxBytes()
        val currentBytesTx = TrafficStats.getTotalTxBytes()

        val bytesRx = currentBytesRx - lastBytesRx
        val bytesTx = currentBytesTx - lastBytesTx

        val bpsRx = bytesRx / elapsedSeconds
        val bpsTx = bytesTx / elapsedSeconds

        lastBytesRx = currentBytesRx
        lastBytesTx = currentBytesTx
        lastTimestamp = currentTimestamp

        val foregroundApp = getForegroundApp()

        Log.d("TrafficMonitor", "BPS Rx: ${"%.2f".format(bpsRx / 1024f)} KB/s, BPS Tx: ${"%.2f".format(bpsTx / 1024f)} KB/s, Active App: $foregroundApp")
        return Triple(bpsRx, bpsTx, foregroundApp)
    }

    private fun getForegroundApp(): String {
        return try {
            val time = System.currentTimeMillis()
            val usageStatsList = usageStatsManager.queryUsageStats(
                UsageStatsManager.INTERVAL_DAILY,
                time - 10000,
                time
            )

            if (usageStatsList != null && usageStatsList.isNotEmpty()) {
                return usageStatsList.maxByOrNull { it.lastTimeUsed }?.packageName ?: "unknown"
            }
            "unknown"
        } catch (e: Exception) {
            Log.e("TrafficMonitor", "Could not get foreground app using UsageStatsManager", e)
            "unknown"
        }
    }
}
