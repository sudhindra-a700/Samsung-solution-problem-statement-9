package com.yoursacgru.testapp
data class PredictionResult(val label: String, val confidence: Float)
data class LiveStatus(
    val action: String,
    val prediction: PredictionResult?,
    val bandwidthBps: Float = 0f, // Add bandwidth to the status
    val timestamp: Long = System.currentTimeMillis()
)
