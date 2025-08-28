package com.yoursacgru.testapp;

import android.os.Bundle;
import android.widget.Button;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Random;

/**
 * SAC-GRU Traffic Analyzer - Actor-Only Inference
 * ===============================================
 * 
 * This Android activity performs real-time traffic classification using only
 * the trained SAC actor network deployed from laptop training.
 * 
 * Key Features:
 * - Lightweight actor-only inference (no critic networks)
 * - Optimized for mobile performance
 * - Real-time REEL vs NON-REEL classification
 * - Comprehensive testing and benchmarking
 * 
 * Model Input: 11 normalized traffic features
 * Model Output: REEL probability (0.0 to 1.0)
 * 
 * Author: Enhanced by Manus AI for Actor-Only Deployment
 */
public class MainActivity extends AppCompatActivity {
    
    private static final String TAG = "SACActorInference";
    private static final String MODEL_FILENAME = "sac_actor_model.tflite";
    
    // UI Components
    private Button testButton;
    private TextView resultTextView;
    private ScrollView scrollView;
    
    // TensorFlow Lite Components
    private Interpreter tfliteInterpreter;
    private boolean isModelLoaded = false;
    
    // Model specifications
    private static final int INPUT_SIZE = 11;  // 11 traffic features
    private static final int OUTPUT_SIZE = 1;  // REEL probability
    
    // Feature names for logging
    private static final String[] FEATURE_NAMES = {
        "Format (Resolution)", "FPS", "Buffer Health", "Stalling", 
        "Quality Changes", "Session Length", "App Type", "Device Type", 
        "Network Type", "Battery Level", "Time Phase"
    };
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        
        initializeUI();
        loadActorModel();
    }
    
    private void initializeUI() {
        testButton = findViewById(R.id.testButton);
        resultTextView = findViewById(R.id.resultTextView);
        scrollView = findViewById(R.id.scrollView);
        
        testButton.setText("Run SAC Actor Tests");
        testButton.setOnClickListener(v -> runActorInferenceTests());
        
        appendResult("SAC-GRU Traffic Analyzer - Actor-Only Inference\n");
        appendResult("=================================================\n\n");
        appendResult("This app uses only the trained SAC actor network for lightweight inference.\n");
        appendResult("The complete SAC-GRU training happens on laptop/desktop.\n\n");
    }
    
    private void loadActorModel() {
        try {
            appendResult("Loading SAC actor model...\n");
            
            // Load TFLite model
            MappedByteBuffer tfliteModel = loadModelFile();
            
            // Create interpreter with optimizations
            Interpreter.Options options = new Interpreter.Options();
            options.setNumThreads(4); // Use multiple threads for better performance
            options.setUseNNAPI(true); // Use Android Neural Networks API if available
            
            tfliteInterpreter = new Interpreter(tfliteModel, options);
            
            // Verify model input/output shapes
            int[] inputShape = tfliteInterpreter.getInputTensor(0).shape();
            int[] outputShape = tfliteInterpreter.getOutputTensor(0).shape();
            
            appendResult(String.format("✅ SAC Actor model loaded successfully!\n"));
            appendResult(String.format("   Input shape: [%d, %d]\n", inputShape[0], inputShape[1]));
            appendResult(String.format("   Output shape: [%d, %d]\n", outputShape[0], outputShape[1]));
            appendResult(String.format("   Model size: %.2f KB\n\n", tfliteModel.capacity() / 1024.0));
            
            isModelLoaded = true;
            testButton.setEnabled(true);
            
        } catch (IOException e) {
            appendResult("❌ Failed to load SAC actor model: " + e.getMessage() + "\n\n");
            testButton.setEnabled(false);
        }
    }
    
    private MappedByteBuffer loadModelFile() throws IOException {
        FileInputStream inputStream = new FileInputStream(getAssets().openFd(MODEL_FILENAME).getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = getAssets().openFd(MODEL_FILENAME).getStartOffset();
        long declaredLength = getAssets().openFd(MODEL_FILENAME).getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    private void runActorInferenceTests() {
        if (!isModelLoaded) {
            appendResult("❌ Model not loaded. Cannot run tests.\n\n");
            return;
        }
        
        appendResult("Starting SAC Actor Inference Tests...\n");
        appendResult("=====================================\n\n");
        
        // Test 1: Basic inference functionality
        testBasicInference();
        
        // Test 2: REEL vs NON-REEL scenarios
        testReelVsNonReelScenarios();
        
        // Test 3: Performance benchmarking
        testPerformanceBenchmark();
        
        // Test 4: Edge cases and robustness
        testEdgeCases();
        
        // Test 5: Real-time simulation
        testRealTimeSimulation();
        
        appendResult("All SAC Actor tests completed! ✅\n\n");
    }
    
    private void testBasicInference() {
        appendResult("Test 1: Basic Actor Inference\n");
        appendResult("-----------------------------\n");
        
        try {
            // Create sample traffic features
            float[][] input = new float[1][INPUT_SIZE];
            float[][] output = new float[1][OUTPUT_SIZE];
            
            // Typical REEL traffic pattern
            input[0][0] = 0.3f;  // Lower resolution (360p)
            input[0][1] = 0.5f;  // 30 FPS
            input[0][2] = 0.6f;  // Moderate buffer health
            input[0][3] = 0.0f;  // No stalling
            input[0][4] = 0.1f;  // Few quality changes
            input[0][5] = 0.2f;  // Short session
            input[0][6] = 0.8f;  // TikTok-like app
            input[0][7] = 0.5f;  // Mobile device
            input[0][8] = 0.7f;  // WiFi network
            input[0][9] = 0.8f;  // Good battery
            input[0][10] = 0.5f; // Mid-session
            
            // Run inference
            long startTime = System.nanoTime();
            tfliteInterpreter.run(input, output);
            long inferenceTime = (System.nanoTime() - startTime) / 1_000_000; // Convert to ms
            
            float reelProbability = output[0][0];
            String classification = reelProbability > 0.5f ? "REEL" : "NON-REEL";
            
            appendResult(String.format("   Input features: [%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]\n",
                input[0][0], input[0][1], input[0][2], input[0][3], input[0][4], 
                input[0][5], input[0][6], input[0][7], input[0][8], input[0][9], input[0][10]));
            appendResult(String.format("   REEL Probability: %.4f\n", reelProbability));
            appendResult(String.format("   Classification: %s\n", classification));
            appendResult(String.format("   Inference Time: %d ms\n", inferenceTime));
            appendResult("   ✅ Basic inference working!\n\n");
            
        } catch (Exception e) {
            appendResult("   ❌ Basic inference failed: " + e.getMessage() + "\n\n");
        }
    }
    
    private void testReelVsNonReelScenarios() {
        appendResult("Test 2: REEL vs NON-REEL Scenarios\n");
        appendResult("----------------------------------\n");
        
        // Test typical REEL patterns
        appendResult("Testing REEL patterns:\n");
        testScenario("TikTok Short Video", createReelPattern("tiktok"));
        testScenario("Instagram Reel", createReelPattern("instagram"));
        testScenario("YouTube Shorts", createReelPattern("youtube_shorts"));
        
        appendResult("\nTesting NON-REEL patterns:\n");
        testScenario("YouTube Long Video", createNonReelPattern("youtube_long"));
        testScenario("Netflix Movie", createNonReelPattern("netflix"));
        testScenario("Zoom Call", createNonReelPattern("zoom"));
        
        appendResult("\n");
    }
    
    private void testScenario(String scenarioName, float[] features) {
        try {
            float[][] input = {features};
            float[][] output = new float[1][OUTPUT_SIZE];
            
            tfliteInterpreter.run(input, output);
            
            float reelProbability = output[0][0];
            String classification = reelProbability > 0.5f ? "REEL" : "NON-REEL";
            
            appendResult(String.format("   %s: %.3f (%s)\n", scenarioName, reelProbability, classification));
            
        } catch (Exception e) {
            appendResult(String.format("   %s: Error - %s\n", scenarioName, e.getMessage()));
        }
    }
    
    private float[] createReelPattern(String platform) {
        float[] features = new float[INPUT_SIZE];
        Random random = new Random();
        
        switch (platform) {
            case "tiktok":
                features[0] = 0.2f + random.nextFloat() * 0.3f; // 240p-480p
                features[1] = 0.4f + random.nextFloat() * 0.2f; // 24-30 FPS
                features[2] = 0.7f + random.nextFloat() * 0.2f; // Good buffer
                features[3] = random.nextFloat() * 0.1f;         // Minimal stalling
                features[4] = random.nextFloat() * 0.2f;         // Few quality changes
                features[5] = 0.1f + random.nextFloat() * 0.2f; // 15-60 seconds
                features[6] = 0.9f;                              // TikTok app
                break;
            case "instagram":
                features[0] = 0.3f + random.nextFloat() * 0.3f; // 360p-720p
                features[1] = 0.4f + random.nextFloat() * 0.2f; // 24-30 FPS
                features[2] = 0.6f + random.nextFloat() * 0.3f; // Variable buffer
                features[3] = random.nextFloat() * 0.2f;         // Some stalling
                features[4] = random.nextFloat() * 0.3f;         // Some quality changes
                features[5] = 0.1f + random.nextFloat() * 0.3f; // 15-90 seconds
                features[6] = 0.7f;                              // Instagram app
                break;
            case "youtube_shorts":
                features[0] = 0.3f + random.nextFloat() * 0.4f; // 360p-720p
                features[1] = 0.4f + random.nextFloat() * 0.4f; // 24-60 FPS
                features[2] = 0.5f + random.nextFloat() * 0.4f; // Variable buffer
                features[3] = random.nextFloat() * 0.2f;         // Some stalling
                features[4] = random.nextFloat() * 0.3f;         // Quality adaptation
                features[5] = 0.1f + random.nextFloat() * 0.2f; // 15-60 seconds
                features[6] = 0.4f;                              // YouTube app
                break;
        }
        
        // Common REEL characteristics
        features[7] = 0.5f + random.nextFloat() * 0.3f; // Mobile device
        features[8] = 0.3f + random.nextFloat() * 0.6f; // Various networks
        features[9] = 0.2f + random.nextFloat() * 0.7f; // Battery level
        features[10] = random.nextFloat();               // Time phase
        
        return features;
    }
    
    private float[] createNonReelPattern(String platform) {
        float[] features = new float[INPUT_SIZE];
        Random random = new Random();
        
        switch (platform) {
            case "youtube_long":
                features[0] = 0.5f + random.nextFloat() * 0.5f; // 720p-1080p+
                features[1] = 0.4f + random.nextFloat() * 0.6f; // 24-60 FPS
                features[2] = 0.3f + random.nextFloat() * 0.6f; // Variable buffer
                features[3] = random.nextFloat() * 0.3f;         // Some stalling
                features[4] = random.nextFloat() * 0.5f;         // Quality adaptation
                features[5] = 0.5f + random.nextFloat() * 0.5f; // 10+ minutes
                features[6] = 0.4f;                              // YouTube app
                break;
            case "netflix":
                features[0] = 0.7f + random.nextFloat() * 0.3f; // 1080p-4K
                features[1] = 0.4f + random.nextFloat() * 0.4f; // 24-60 FPS
                features[2] = 0.6f + random.nextFloat() * 0.3f; // Good buffer
                features[3] = random.nextFloat() * 0.2f;         // Minimal stalling
                features[4] = random.nextFloat() * 0.4f;         // Adaptive quality
                features[5] = 0.8f + random.nextFloat() * 0.2f; // 45+ minutes
                features[6] = 0.6f;                              // Netflix app
                break;
            case "zoom":
                features[0] = 0.3f + random.nextFloat() * 0.4f; // 360p-720p
                features[1] = 0.4f + random.nextFloat() * 0.2f; // 24-30 FPS
                features[2] = 0.4f + random.nextFloat() * 0.4f; // Variable buffer
                features[3] = random.nextFloat() * 0.4f;         // Network issues
                features[4] = random.nextFloat() * 0.6f;         // Quality adaptation
                features[5] = 0.6f + random.nextFloat() * 0.4f; // 30+ minutes
                features[6] = 0.2f;                              // Video call app
                break;
        }
        
        // Common NON-REEL characteristics
        features[7] = random.nextFloat();                // Various devices
        features[8] = 0.3f + random.nextFloat() * 0.6f; // Various networks
        features[9] = 0.2f + random.nextFloat() * 0.7f; // Battery level
        features[10] = random.nextFloat();               // Time phase
        
        return features;
    }
    
    private void testPerformanceBenchmark() {
        appendResult("Test 3: Performance Benchmark\n");
        appendResult("---------------------------\n");
        
        int numIterations = 100;
        long totalTime = 0;
        long minTime = Long.MAX_VALUE;
        long maxTime = 0;
        
        float[][] input = new float[1][INPUT_SIZE];
        float[][] output = new float[1][OUTPUT_SIZE];
        
        // Warm up
        for (int i = 0; i < 10; i++) {
            fillRandomFeatures(input[0]);
            tfliteInterpreter.run(input, output);
        }
        
        // Benchmark
        for (int i = 0; i < numIterations; i++) {
            fillRandomFeatures(input[0]);
            
            long startTime = System.nanoTime();
            tfliteInterpreter.run(input, output);
            long inferenceTime = System.nanoTime() - startTime;
            
            totalTime += inferenceTime;
            minTime = Math.min(minTime, inferenceTime);
            maxTime = Math.max(maxTime, inferenceTime);
        }
        
        double avgTimeMs = (totalTime / (double) numIterations) / 1_000_000;
        double minTimeMs = minTime / 1_000_000.0;
        double maxTimeMs = maxTime / 1_000_000.0;
        double throughputFps = 1000.0 / avgTimeMs;
        
        appendResult(String.format("   Iterations: %d\n", numIterations));
        appendResult(String.format("   Average time: %.2f ms\n", avgTimeMs));
        appendResult(String.format("   Min time: %.2f ms\n", minTimeMs));
        appendResult(String.format("   Max time: %.2f ms\n", maxTimeMs));
        appendResult(String.format("   Throughput: %.1f FPS\n", throughputFps));
        appendResult("   ✅ Performance benchmark completed!\n\n");
    }
    
    private void testEdgeCases() {
        appendResult("Test 4: Edge Cases and Robustness\n");
        appendResult("--------------------------------\n");
        
        // Test with extreme values
        testEdgeCase("All zeros", new float[INPUT_SIZE]);
        
        float[] allOnes = new float[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) allOnes[i] = 1.0f;
        testEdgeCase("All ones", allOnes);
        
        // Test with mixed extreme values
        float[] mixed = new float[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) {
            mixed[i] = (i % 2 == 0) ? 0.0f : 1.0f;
        }
        testEdgeCase("Alternating 0/1", mixed);
        
        appendResult("   ✅ Edge case testing completed!\n\n");
    }
    
    private void testEdgeCase(String caseName, float[] features) {
        try {
            float[][] input = {features};
            float[][] output = new float[1][OUTPUT_SIZE];
            
            tfliteInterpreter.run(input, output);
            
            float reelProbability = output[0][0];
            appendResult(String.format("   %s: %.4f\n", caseName, reelProbability));
            
        } catch (Exception e) {
            appendResult(String.format("   %s: Error - %s\n", caseName, e.getMessage()));
        }
    }
    
    private void testRealTimeSimulation() {
        appendResult("Test 5: Real-Time Traffic Simulation\n");
        appendResult("------------------------------------\n");
        
        Random random = new Random();
        int correctPredictions = 0;
        int totalPredictions = 20;
        
        for (int i = 0; i < totalPredictions; i++) {
            boolean isActuallyReel = random.nextBoolean();
            float[] features = isActuallyReel ? 
                createReelPattern("tiktok") : createNonReelPattern("youtube_long");
            
            float[][] input = {features};
            float[][] output = new float[1][OUTPUT_SIZE];
            
            tfliteInterpreter.run(input, output);
            
            float reelProbability = output[0][0];
            boolean predictedReel = reelProbability > 0.5f;
            
            if (predictedReel == isActuallyReel) {
                correctPredictions++;
            }
            
            String actual = isActuallyReel ? "REEL" : "NON-REEL";
            String predicted = predictedReel ? "REEL" : "NON-REEL";
            String result = (predictedReel == isActuallyReel) ? "✅" : "❌";
            
            appendResult(String.format("   Sample %d: Actual=%s, Predicted=%s (%.3f) %s\n", 
                i + 1, actual, predicted, reelProbability, result));
        }
        
        double accuracy = (double) correctPredictions / totalPredictions;
        appendResult(String.format("\n   Simulation Accuracy: %.1f%% (%d/%d)\n", 
            accuracy * 100, correctPredictions, totalPredictions));
        appendResult("   ✅ Real-time simulation completed!\n\n");
    }
    
    private void fillRandomFeatures(float[] features) {
        Random random = new Random();
        for (int i = 0; i < features.length; i++) {
            features[i] = random.nextFloat();
        }
    }
    
    private void appendResult(String text) {
        runOnUiThread(() -> {
            resultTextView.append(text);
            scrollView.post(() -> scrollView.fullScroll(ScrollView.FOCUS_DOWN));
        });
    }
    
    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (tfliteInterpreter != null) {
            tfliteInterpreter.close();
        }
    }
}

