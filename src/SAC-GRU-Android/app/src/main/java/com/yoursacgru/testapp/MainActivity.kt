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
import org.tensorflow.lite.flex.FlexDelegate  // ✅ CRITICAL CHANGE 1: Added FlexDelegate import
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.random.Random

class MainActivity : ComponentActivity() {

    private val TAG: String = "SACGRU_Final_RL"
    private val MODEL_FILE: String = "sac_actor_model.tflite"
    private var actorInterpreter: Interpreter? = null
    private var flexDelegate: FlexDelegate? = null  // ✅ CRITICAL CHANGE 2: Added FlexDelegate field

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            SACGRUTheme {
                SACGRUApp()
            }
        }
    }

    // Data class to hold the result of a prediction
    data class PredictionResult(val label: String, val confidence: Float)
    // Data class to hold the result of a simulated learning step
    data class LearningStep(val prediction: String, val reward: Double, val feedback: String)

    @Composable
    fun SACGRUApp() {
        // State variables for the UI
        val modelLoaded = remember { mutableStateOf(false) }
        val isLoading = remember { mutableStateOf(true) }
        val inputValue = remember { mutableStateOf("1080,60,1,15000,0,5000,0,0,0") }  // ✅ CRITICAL CHANGE 6: Updated to 9 values
        val predictionResult = remember { mutableStateOf<PredictionResult?>(null) }
        val lastLearningStep = remember { mutableStateOf<LearningStep?>(null) }
        val keyboardController = LocalSoftwareKeyboardController.current
        val coroutineScope = rememberCoroutineScope()

        // Asynchronously load the model when the app starts
        LaunchedEffect(Unit) {
            isLoading.value = true
            modelLoaded.value = withContext(Dispatchers.IO) { loadModel() }
            isLoading.value = false
        }

        Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
            Column(
                modifier = Modifier
                    .fillMaxSize()
                    .padding(innerPadding)
                    .padding(16.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.Center
            ) {
                // Show a loading indicator, an error message, or the main UI
                when {
                    isLoading.value -> {
                        CircularProgressIndicator()
                        Text("Loading Model...", style = MaterialTheme.typography.bodyLarge, modifier = Modifier.padding(top = 16.dp))
                    }
                    !modelLoaded.value -> {
                        Text("Error: Model could not be loaded.", color = Color.Red)
                    }
                    else -> {
                        // Main UI content
                        Column(
                            modifier = Modifier.fillMaxWidth(),
                            verticalArrangement = Arrangement.spacedBy(16.dp),
                            horizontalAlignment = Alignment.CenterHorizontally
                        ) {
                            Text(
                                text = "Real-Time Traffic Classification",
                                style = MaterialTheme.typography.headlineSmall.copy(fontWeight = FontWeight.Bold)
                            )

                            // Display prediction results
                            ResultCards(predictionResult.value)

                            // Display the on-device learning simulation card if a prediction has been made
                            predictionResult.value?.let {
                                OnDeviceLearningCard(it) { learningStep ->
                                    lastLearningStep.value = learningStep
                                }
                            }
                            
                            // Display feedback from the last learning step
                            lastLearningStep.value?.let {
                                Text(
                                    "Last Learning Step: ${it.feedback} (Reward: ${it.reward})",
                                    style = MaterialTheme.typography.bodySmall
                                )
                            }

                            // Text field for user input
                            TextField(
                                value = inputValue.value,
                                onValueChange = { inputValue.value = it },
                                label = { Text("Enter 9 comma-separated features") },  // ✅ CRITICAL CHANGE 4: Changed from 7 to 9
                                modifier = Modifier.fillMaxWidth(),
                                singleLine = true,
                                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number, imeAction = ImeAction.Done),
                                keyboardActions = KeyboardActions(onDone = {
                                    keyboardController?.hide()
                                    coroutineScope.launch {
                                        predictionResult.value = runInference(inputValue.value)
                                        lastLearningStep.value = null // Reset learning step on new inference
                                    }
                                })
                            )
                        }
                    }
                }
            }
        }
    }

    @Composable
    fun OnDeviceLearningCard(result: PredictionResult, onLearningStep: (LearningStep) -> Unit) {
        Card(modifier = Modifier.fillMaxWidth(), elevation = CardDefaults.cardElevation(2.dp)) {
            Column(modifier = Modifier.padding(16.dp)) {
                Text("On-Device Learning Simulation", fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.height(8.dp))
                Text("Simulate user experience to generate a reward for on-device adaptation.", style = MaterialTheme.typography.bodySmall)
                Spacer(modifier = Modifier.height(8.dp))
                Button(onClick = {
                    // Simulate user experience (e.g., video stalling)
                    val didStall = Random.nextBoolean()
                    
                    // Calculate reward based on the prediction and the simulated experience
                    val reward = if (result.label == "REEL") {
                        if (didStall) -1.0 else 1.0 // Punish for stalling, reward for smoothness
                    } else {
                        0.0 // Neutral reward for non-reel traffic
                    }
                    
                    val feedback = if (didStall) "Video Stalled" else "Playback Smooth"
                    onLearningStep(LearningStep(result.label, reward, feedback))
                    
                    // In a full implementation, this is where you would trigger
                    // an on-device training step using the collected reward.
                }) {
                    Text("Simulate User Experience")
                }
            }
        }
    }

    @Composable
    fun ResultCards(result: PredictionResult?) {
        val reelConfidence = if (result?.label == "REEL") result.confidence * 100 else 100 - ((result?.confidence ?: 1f) * 100)
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

    // ✅ CRITICAL CHANGE 3: Complete loadModel() function rewrite for FlexDelegate
    private fun loadModel(): Boolean {
        return try {
            val modelBuffer = loadModelFile(MODEL_FILE)
            
            // Initialize FlexDelegate for GRU support
            flexDelegate = FlexDelegate()
            val options = Interpreter.Options().apply {
                addDelegate(flexDelegate!!)
            }
            
            actorInterpreter = Interpreter(modelBuffer, options)
            Log.d(TAG, "Model loaded successfully.")
            true
        } catch (e: IOException) {
            Log.e(TAG, "Error loading model: ${e.message}")
            false
        }
    }

    private fun runInference(inputString: String): PredictionResult? {
        if (actorInterpreter == null) return null
        
        // Parse the input string into a float array
        val features = inputString.split(",").mapNotNull { it.trim().toFloatOrNull() }
        if (features.size != 9) {  // ✅ CRITICAL CHANGE 4: Changed from 7 to 9
            Log.e(TAG, "Invalid input: Expected 9 features, got ${features.size}")  // ✅ CRITICAL CHANGE 4: Updated error message
            return null // Handle invalid input gracefully
        }

        // Prepare input and output buffers for the TFLite model
        val inputArray = arrayOf(arrayOf(features.toFloatArray()))
        val outputArray = Array(1) { FloatArray(2) }

        try {
            actorInterpreter?.run(inputArray, outputArray)
            val nonReelProb = outputArray[0][0]
            val reelProb = outputArray[0][1]
            
            // Determine the label and confidence from the output probabilities
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

    // ✅ CRITICAL CHANGE 5: Updated onDestroy() to close FlexDelegate
    override fun onDestroy() {
        super.onDestroy()
        // Close the interpreter to free up resources
        actorInterpreter?.close()
        flexDelegate?.close()  // Added FlexDelegate cleanup
    }
}

