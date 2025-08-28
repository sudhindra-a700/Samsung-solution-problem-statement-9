package com.yoursacgru.testapp

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalSoftwareKeyboardController
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {

    private val TAG: String = "SACGRU_Final"
    private val MODEL_FILE: String = "sac_actor_model.tflite"
    private var actorInterpreter: Interpreter? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SACGRUTheme {
                SACGRUApp()
            }
        }
    }

    data class PredictionResult(val label: String, val confidence: Float)

    @Composable
    fun SACGRUApp() {
        val modelLoaded = remember { mutableStateOf(false) }
        val isLoading = remember { mutableStateOf(true) }
        val inputValue = remember { mutableStateOf("1080,60,15000,0,5000,0,0") } // Default REEL example
        val predictionResult = remember { mutableStateOf<PredictionResult?>(null) }
        val keyboardController = LocalSoftwareKeyboardController.current
        val coroutineScope = rememberCoroutineScope()

        LaunchedEffect(Unit) {
            isLoading.value = true
            modelLoaded.value = withContext(Dispatchers.IO) { loadModel() }
            isLoading.value = false
        }

        Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
            Column(
                modifier = Modifier.fillMaxSize().padding(innerPadding).padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                if (isLoading.value) {
                    CircularProgressIndicator()
                    Text("Loading Model...", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 16.dp))
                } else if (!modelLoaded.value) {
                    Text("Error: Model could not be loaded.", color = Color.Red)
                } else {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Text(
                            text = "Real-Time Traffic Classification",
                            style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold)
                        )

                        ResultCards(predictionResult.value)

                        TextField(
                            value = inputValue.value,
                            onValueChange = { inputValue.value = it },
                            label = { Text("Enter 7 comma-separated features") },
                            modifier = Modifier.fillMaxWidth(),
                            singleLine = true,
                            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number, imeAction = ImeAction.Done),
                            keyboardActions = KeyboardActions(onDone = {
                                keyboardController?.hide()
                                coroutineScope.launch {
                                    predictionResult.value = runInference(inputValue.value)
                                }
                            })
                        )

                        Button(
                            onClick = {
                                keyboardController?.hide()
                                coroutineScope.launch {
                                    predictionResult.value = runInference(inputValue.value)
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text("Classify")
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun ResultCards(result: PredictionResult?) {
        val reelConfidence = if (result?.label == "REEL") result.confidence * 100 else 100 - (result?.confidence ?: 0f) * 100
        val nonReelConfidence = 100 - reelConfidence

        Card(modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(4.dp)) {
            Row(modifier = Modifier.padding(16.dp).fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("REEL Confidence", fontWeight = FontWeight.SemiBold)
                Text("${"%.1f".format(reelConfidence)}%", fontWeight = FontWeight.Bold)
            }
        }
        Card(modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(4.dp)) {
            Row(modifier = Modifier.padding(16.dp).fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text("NON-REEL Confidence", fontWeight = FontWeight.SemiBold)
                Text("${"%.1f".format(nonReelConfidence)}%", fontWeight = FontWeight.Bold)
            }
        }
    }

    private fun loadModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(MODEL_FILE)
            actorInterpreter = Interpreter(modelBuffer)
            Log.d(TAG, "Model loaded successfully.")
            true
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model: ${e.message}")
            false
        }
    }

    private fun runInference(inputString: String): PredictionResult? {
        if (actorInterpreter == null) return null
        
        val features = inputString.split(",").mapNotNull { it.trim().toFloatOrNull() }
        if (features.size != 7) {
            Log.e(TAG, "Invalid input: Expected 7 features, got ${features.size}")
            return null
        }

        // The model expects an input shape of (1, 1, 7)
        val inputArray = arrayOf(arrayOf(features.toFloatArray()))
        val outputArray = Array(1) { FloatArray(2) } // Output is [prob_non_reel, prob_reel]

        try {
            actorInterpreter?.run(inputArray, outputArray)
            val nonReelProb = outputArray[0][0]
            val reelProb = outputArray[0][1]
            val label = if (reelProb > nonReelProb) "REEL" else "NON-REEL"
            val confidence = if (label == "REEL") reelProb else nonReelProb
            return PredictionResult(label, confidence)
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference: ${e.message}")
            return null
        }
    }

    @Throws(IOException::class)
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
