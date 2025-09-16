package com.yoursacgru.testapp

data class PredictionResult(val label: String, val confidence: Float)

enum class OptimizationProfile {
    DATA_SAVER,
    BALANCED,
    HIGH_PERFORMANCE
}

data class LiveStatus(
    val action: String,
    val prediction: PredictionResult?,
    val bandwidthBps: Float = 0f,
    val uploadBps: Float = 0f,
    val timestamp: Long = System.currentTimeMillis()
)
