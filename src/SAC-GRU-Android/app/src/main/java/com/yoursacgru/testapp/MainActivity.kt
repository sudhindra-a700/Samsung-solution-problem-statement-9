package com.yoursacgru.testapp

import android.app.AppOpsManager
import android.content.Context
import android.content.Intent
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SACGRUTheme {
                SACGRUApp()
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun SACGRUApp() {
        val context = LocalContext.current
        var serviceRunning by remember { mutableStateOf(false) }
        val statusHistory by QoSOptimizer.statusHistory.collectAsState()
        val hasUsageAccess = remember { mutableStateOf(hasUsageStatsPermission(context)) }

        Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                Text(
                    text = "VULCAN Real-Time Optimizer",
                    style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold)
                )

                if (!hasUsageAccess.value) {
                    PermissionCard {
                        context.startActivity(Intent(Settings.ACTION_USAGE_ACCESS_SETTINGS))
                    }
                }

                if (statusHistory.isNotEmpty()) {
                    LiveStatusCard(status = statusHistory[0])
                    HistoricActivityCard(status = statusHistory.getOrNull(1))
                    LiveBandwidthChart(history = statusHistory)
                    HistoricActivitiesTimeline(history = statusHistory)
                }

                OptimizationProfileCard()

                Card(modifier = Modifier.fillMaxWidth()) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text("Monitoring Service Control", fontWeight = FontWeight.Bold, modifier = Modifier.align(Alignment.CenterHorizontally))
                        Spacer(Modifier.height(8.dp))
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.SpaceEvenly
                        ) {
                            Button(onClick = {
                                context.startService(Intent(context, EnhancedOptimizationService::class.java))
                                serviceRunning = true
                            }, enabled = !serviceRunning && hasUsageAccess.value) {
                                Text("Start Service")
                            }
                            Button(onClick = {
                                context.stopService(Intent(context, EnhancedOptimizationService::class.java))
                                QoSOptimizer.resetState()
                                serviceRunning = false
                            }, enabled = serviceRunning) {
                                Text("Stop Service")
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun LiveStatusCard(status: LiveStatus) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            elevation = CardDefaults.cardElevation(4.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)
        ) {
            StatusItem(status = status, title = "Live Status")
        }
    }

    @Composable
    fun HistoricActivityCard(status: LiveStatus?) {
        status?.let {
            Card(
                modifier = Modifier.fillMaxWidth(),
                elevation = CardDefaults.cardElevation(2.dp),
                colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.secondaryContainer)
            ) {
                StatusItem(status = it, title = "Previous Activity")
            }
        }
    }

    @Composable
    fun StatusItem(status: LiveStatus, title: String) {
        Column(
            modifier = Modifier.padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Text(
                text = title,
                style = MaterialTheme.typography.titleMedium,
                fontWeight = FontWeight.Bold,
                modifier = Modifier.align(Alignment.CenterHorizontally)
            )
            HorizontalDivider(modifier = Modifier.padding(vertical = 4.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Action Taken:", fontWeight = FontWeight.SemiBold)
                Text(status.action)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Detected Traffic Type:", fontWeight = FontWeight.SemiBold)
                val label = status.prediction?.label ?: "N/A"
                val color = when (label) {
                    "REEL" -> Color(0xFFD32F2F)
                    "NON-REEL" -> Color(0xFF388E3C)
                    else -> LocalContentColor.current
                }
                Text(label, color = color, fontWeight = FontWeight.Bold)
            }
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("Model Confidence:", fontWeight = FontWeight.SemiBold)
                status.prediction?.let {
                    LinearProgressIndicator(
                        progress = { it.confidence },
                        modifier = Modifier
                            .weight(1f)
                            .padding(horizontal = 8.dp),
                        color = if (it.label == "REEL") Color(0xFFD32F2F) else Color(0xFF388E3C)
                    )
                    Text("${"%.1f".format(it.confidence * 100)}%")
                } ?: Text("N/A")
            }
        }
    }

    @Composable
    fun LiveBandwidthChart(history: List<LiveStatus>) {
        val bandwidthData = history.map { it.bandwidthBps / 1024f }.reversed()
        val chartColor = MaterialTheme.colorScheme.primary

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Live Bandwidth (KB/s)", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, modifier = Modifier.align(Alignment.CenterHorizontally))
                Spacer(Modifier.height(8.dp))
                Canvas(modifier = Modifier
                    .fillMaxWidth()
                    .height(100.dp)) {
                    if (bandwidthData.size > 1) {
                        val maxBw = bandwidthData.maxOrNull()?.coerceAtLeast(1f) ?: 1f
                        val path = Path()
                        path.moveTo(0f, size.height - (bandwidthData[0] / maxBw) * size.height)
                        for (i in 1 until bandwidthData.size) {
                            val x = (i.toFloat() / (bandwidthData.size - 1)) * size.width
                            val y = size.height - (bandwidthData[i] / maxBw) * size.height
                            path.lineTo(x, y)
                        }
                        drawPath(path, color = chartColor, style = Stroke(width = 3.dp.toPx()))
                    }
                }
            }
        }
    }

    @Composable
    fun HistoricActivitiesTimeline(history: List<LiveStatus>) {
        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Activity Timeline (Most Recent First)", style = MaterialTheme.typography.titleMedium, fontWeight = FontWeight.Bold, modifier = Modifier.align(Alignment.CenterHorizontally))
                Spacer(Modifier.height(8.dp))
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                    val timelineItems = (history + List(10) { null }).take(10)
                    timelineItems.forEach { status ->
                        val color = when (status?.prediction?.label) {
                            "REEL" -> Color(0xFFD32F2F)
                            "NON-REEL" -> Color(0xFF388E3C)
                            else -> Color.Gray.copy(alpha = 0.3f)
                        }
                        Surface(
                            modifier = Modifier
                                .weight(1f)
                                .height(20.dp),
                            color = color,
                            shape = RoundedCornerShape(4.dp)
                        ) {}
                    }
                }
            }
        }
    }

    @OptIn(ExperimentalMaterial3Api::class)
    @Composable
    fun OptimizationProfileCard() {
        var expanded by remember { mutableStateOf(false) }
        val profiles = OptimizationProfile.values()
        var selectedProfile by remember { mutableStateOf(QoSOptimizer.currentProfile) }

        Card(modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text(
                    "Optimization Profile",
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier.align(Alignment.CenterHorizontally)
                )
                Spacer(Modifier.height(8.dp))
                ExposedDropdownMenuBox(
                    expanded = expanded,
                    onExpandedChange = { expanded = !expanded }
                ) {
                    TextField(
                        value = selectedProfile.name,
                        onValueChange = {},
                        readOnly = true,
                        trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded = expanded) },
                        modifier = Modifier.menuAnchor()
                    )
                    ExposedDropdownMenu(
                        expanded = expanded,
                        onDismissRequest = { expanded = false }
                    ) {
                        profiles.forEach { profile ->
                            DropdownMenuItem(
                                text = { Text(profile.name) },
                                onClick = {
                                    selectedProfile = profile
                                    QoSOptimizer.currentProfile = profile
                                    expanded = false
                                }
                            )
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun PermissionCard(onRequestPermission: () -> Unit) {
        Card(
            modifier = Modifier.fillMaxWidth(),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)
        ) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("Permission Required", fontWeight = FontWeight.Bold, modifier = Modifier.align(Alignment.CenterHorizontally))
                Text("To identify apps using the network, this app needs 'Usage Access' permission.", textAlign = TextAlign.Center)
                Spacer(Modifier.height(8.dp))
                Button(onClick = onRequestPermission, modifier = Modifier.align(Alignment.CenterHorizontally)) {
                    Text("Grant Permission")
                }
            }
        }
    }

    private fun hasUsageStatsPermission(context: Context): Boolean {
        val appOps = context.getSystemService(Context.APP_OPS_SERVICE) as AppOpsManager
        val mode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            appOps.unsafeCheckOpNoThrow(
                AppOpsManager.OPSTR_GET_USAGE_STATS,
                android.os.Process.myUid(),
                context.packageName
            )
        } else {
            @Suppress("DEPRECATION")
            appOps.checkOpNoThrow(
                AppOpsManager.OPSTR_GET_USAGE_STATS,
                android.os.Process.myUid(),
                context.packageName
            )
        }
        return mode == AppOpsManager.MODE_ALLOWED
    }
}
