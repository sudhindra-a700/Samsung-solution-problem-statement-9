package com.yoursacgru.testapp

import android.content.Context
import android.util.Log
import androidx.work.CoroutineWorker
import androidx.work.WorkerParameters
import kotlinx.coroutines.delay

class DeferredSyncWorker(context: Context, params: WorkerParameters) :
    CoroutineWorker(context, params) {

    override suspend fun doWork(): Result {
        return try {
            Log.d("DeferredSyncWorker", "Executing deferred sync operations...")
            delay(5000)
            Log.d("DeferredSyncWorker", "Deferred sync completed successfully.")
            Result.success()
        } catch (e: Exception) {
            Log.e("DeferredSyncWorker", "Deferred sync failed: ${e.message}")
            Result.failure()
        }
    }
}
