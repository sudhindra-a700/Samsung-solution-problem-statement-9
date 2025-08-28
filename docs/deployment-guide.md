# Deployment Guide - SAC-GRU Traffic Analyzer

## Overview

This guide provides step-by-step instructions for deploying the SAC-GRU Traffic Analyzer in both development and production environments. The system follows a **laptop-to-Android deployment pipeline** where training occurs on powerful machines and inference runs on mobile devices.

## System Requirements

### Development Environment

#### Laptop/Server (Training)
- **OS**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **CPU**: Intel i5/AMD Ryzen 5 or better (8+ cores recommended)
- **RAM**: 16GB minimum, 32GB recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 10GB free space
- **Python**: 3.11 or higher

#### Android Development
- **Android Studio**: Arctic Fox (2020.3.1) or newer
- **Java**: JDK 17
- **Android SDK**: API level 24+ (Android 7.0+)
- **Gradle**: 8.4 (included in project)

### Production Environment

#### Android Devices
- **OS**: Android 7.0+ (API level 24+)
- **RAM**: 4GB minimum, 6GB+ recommended
- **Storage**: 100MB available space
- **Processor**: ARM64 or x86_64 architecture
- **Network**: Wi-Fi or cellular data connection

## Installation Guide

### 1. Environment Setup

#### Clone the Repository
```bash
# Clone the submission repository
git clone https://github.com/your-team/sac-gru-ennovatex-submission.git
cd sac-gru-ennovatex-submission
```

#### Python Environment Setup
```bash
# Navigate to source directory
cd src/

# Create virtual environment
python -m venv sac_gru_env

# Activate virtual environment
# On Windows:
sac_gru_env\\Scripts\\activate
# On macOS/Linux:
source sac_gru_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run environment setup script
python setup_environment.py --install-deps --verify-installation
```

#### Verify Installation
```bash
# Test the installation
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
python -c "import numpy as np; print(f'NumPy version: {np.__version__}')"

# Test SAC-GRU components
python test_sac_gru_rl_implementation.py
```

### 2. Training Pipeline Deployment

#### Quick Training (Pre-configured)
```bash
# Run the complete training pipeline
python laptop_training_pipeline_fixed.py --train --extract-actor --deploy-android

# This will:
# 1. Generate synthetic training data
# 2. Train the SAC-GRU model
# 3. Extract the actor network
# 4. Convert to TensorFlow Lite
# 5. Copy to Android assets directory
```

#### Custom Training Configuration
```bash
# Advanced training with custom parameters
python laptop_training_pipeline_fixed.py \
    --train \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.0003 \
    --hidden-units 128 \
    --sequence-length 50 \
    --extract-actor \
    --optimize-mobile \
    --deploy-android
```

#### Training Monitoring
```bash
# Monitor training progress
tail -f training_logs.txt

# View training metrics
python -c "
import json
with open('training_metrics.json', 'r') as f:
    metrics = json.load(f)
    print(f'Final Accuracy: {metrics[\"accuracy\"]:.4f}')
    print(f'Training Time: {metrics[\"training_time\"]:.2f} minutes')
"
```

### 3. Android Application Deployment

#### Open in Android Studio
1. Launch Android Studio
2. Select "Open an existing project"
3. Navigate to `src/Your-SAC-GRU-Android/`
4. Click "OK" to open the project

#### Project Configuration
```bash
# Verify Gradle configuration
cd src/Your-SAC-GRU-Android/
./gradlew --version

# Clean and sync project
./gradlew clean
./gradlew build
```

#### Build Configuration
```gradle
// Verify build.gradle (Module: app) settings
android {
    compileSdk 34
    
    defaultConfig {
        applicationId "com.yoursacgru.testapp"
        minSdk 24
        targetSdk 34
        versionCode 1
        versionName "1.0"
    }
    
    buildFeatures {
        compose true
    }
    
    composeOptions {
        kotlinCompilerExtensionVersion '1.5.8'
    }
}
```

#### Device Setup
1. **Enable Developer Options**:
   - Go to Settings → About Phone
   - Tap "Build Number" 7 times
   - Developer Options will be enabled

2. **Enable USB Debugging**:
   - Go to Settings → Developer Options
   - Enable "USB Debugging"
   - Enable "Install via USB"

3. **Connect Device**:
   - Connect Android device via USB
   - Accept debugging permissions on device

#### Build and Deploy
```bash
# Build debug APK
./gradlew assembleDebug

# Install on connected device
./gradlew installDebug

# Or build and install in one command
./gradlew installDebug
```

#### Alternative: Build Release APK
```bash
# Build release APK (for distribution)
./gradlew assembleRelease

# APK will be generated at:
# app/build/outputs/apk/release/app-release-unsigned.apk
```

## Configuration Options

### Training Configuration

#### Model Hyperparameters
```python
# Edit laptop_training_pipeline_fixed.py
TRAINING_CONFIG = {
    'sequence_length': 50,      # Input sequence length
    'feature_dim': 15,          # Number of features
    'hidden_units': 128,        # GRU hidden units
    'learning_rate': 3e-4,      # Learning rate
    'batch_size': 32,           # Training batch size
    'epochs': 50,               # Training epochs
    'gamma': 0.99,              # SAC discount factor
    'tau': 0.005,               # Soft update coefficient
    'alpha': 0.2,               # Entropy regularization
}
```

#### Data Generation Settings
```python
# Edit fast_massive_generator.py
DATA_CONFIG = {
    'num_samples': 10000,       # Total samples to generate
    'reel_ratio': 0.5,          # Ratio of REEL vs NON-REEL
    'noise_level': 0.1,         # Data noise level
    'feature_correlation': 0.8,  # Feature correlation strength
}
```

### Android Configuration

#### Model Settings
```kotlin
// Edit MainActivity.kt
class ModelConfig {
    companion object {
        const val MODEL_NAME = "sac_actor_model.tflite"
        const val INPUT_SIZE = 750  // 50 * 15 features
        const val OUTPUT_SIZE = 2   // REEL vs NON-REEL
        const val INFERENCE_THREADS = 4
        const val USE_NNAPI = true
        const val USE_GPU = true
    }
}
```

#### UI Configuration
```kotlin
// Edit theme and colors in ui/theme/
val PrimaryColor = Color(0xFF1976D2)      // Blue
val SecondaryColor = Color(0xFF00BCD4)    // Teal
val BackgroundColor = Color(0xFFF5F5F5)   // Light gray
```

## Performance Optimization

### Training Optimization

#### GPU Acceleration
```python
# Enable GPU training (if available)
import tensorflow as tf

# Check GPU availability
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Configure GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
```

#### Memory Optimization
```python
# Optimize memory usage during training
import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

# Use mixed precision training
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

### Android Optimization

#### Model Optimization
```bash
# Optimize TensorFlow Lite model
python -c "
import tensorflow as tf

# Load and optimize model
converter = tf.lite.TFLiteConverter.from_saved_model('models/sac_actor')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

# Convert and save
tflite_model = converter.convert()
with open('optimized_model.tflite', 'wb') as f:
    f.write(tflite_model)
"
```

#### Runtime Optimization
```kotlin
// Optimize inference performance
class InferenceOptimizer {
    fun optimizeInterpreter(interpreter: Interpreter) {
        interpreter.setNumThreads(4)
        interpreter.setUseNNAPI(true)
        
        // Use GPU delegate if available
        val gpuDelegate = GpuDelegate()
        interpreter.modifyGraphWithDelegate(gpuDelegate)
    }
}
```

## Monitoring and Logging

### Training Monitoring
```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.INFO)

# Monitor training progress
class TrainingMonitor:
    def __init__(self):
        self.metrics = []
    
    def log_epoch(self, epoch, loss, accuracy):
        self.metrics.append({
            'epoch': epoch,
            'loss': loss,
            'accuracy': accuracy,
            'timestamp': datetime.now()
        })
        
        # Save metrics
        with open('training_metrics.json', 'w') as f:
            json.dump(self.metrics, f, default=str, indent=2)
```

### Android Monitoring
```kotlin
// Performance monitoring
class PerformanceMonitor {
    fun logInference(inferenceTime: Long, prediction: String, confidence: Float) {
        Log.d("Performance", """
            Inference Time: ${inferenceTime}ms
            Prediction: $prediction
            Confidence: ${confidence}%
            Memory Usage: ${getMemoryUsage()}MB
        """.trimIndent())
    }
    
    private fun getMemoryUsage(): Long {
        val runtime = Runtime.getRuntime()
        return (runtime.totalMemory() - runtime.freeMemory()) / 1024 / 1024
    }
}
```

## Troubleshooting

### Common Issues

#### Training Issues
```bash
# Issue: CUDA out of memory
# Solution: Reduce batch size
python laptop_training_pipeline_fixed.py --batch-size 16

# Issue: Slow training
# Solution: Enable mixed precision
export TF_ENABLE_AUTO_MIXED_PRECISION=1

# Issue: Model not converging
# Solution: Adjust learning rate
python laptop_training_pipeline_fixed.py --learning-rate 0.001
```

#### Android Build Issues
```bash
# Issue: Gradle sync failed
# Solution: Clean and rebuild
./gradlew clean
./gradlew build --refresh-dependencies

# Issue: SDK not found
# Solution: Update local.properties
echo "sdk.dir=/path/to/android/sdk" > local.properties

# Issue: Out of memory during build
# Solution: Increase Gradle heap size
echo "org.gradle.jvmargs=-Xmx4g" >> gradle.properties
```

#### Runtime Issues
```kotlin
// Issue: Model loading failed
// Solution: Verify model path and permissions
private fun verifyModel(): Boolean {
    return try {
        val assetManager = assets
        val inputStream = assetManager.open("sac_actor_model.tflite")
        inputStream.close()
        true
    } catch (e: Exception) {
        Log.e("Model", "Model verification failed: ${e.message}")
        false
    }
}
```

## Production Deployment

### APK Distribution
```bash
# Build signed release APK
./gradlew assembleRelease

# Sign APK (if not using Android Studio)
jarsigner -verbose -sigalg SHA1withRSA -digestalg SHA1 \
    -keystore my-release-key.keystore \
    app-release-unsigned.apk alias_name

# Align APK
zipalign -v 4 app-release-unsigned.apk app-release.apk
```

### Continuous Integration
```yaml
# .github/workflows/ci.yml
name: CI/CD Pipeline

on: [push, pull_request]

jobs:
  train-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r src/requirements.txt
      - name: Train model
        run: |
          cd src/
          python laptop_training_pipeline_fixed.py --train --extract-actor
      
  build-android:
    needs: train-model
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Setup JDK
        uses: actions/setup-java@v2
        with:
          java-version: '17'
      - name: Build APK
        run: |
          cd src/Your-SAC-GRU-Android/
          ./gradlew assembleDebug
```

## Security Considerations

### Data Privacy
- All processing occurs on-device
- No user data is transmitted to external servers
- Models are stored locally and encrypted

### Model Security
```python
# Model integrity verification
import hashlib

def verify_model_integrity(model_path: str, expected_hash: str) -> bool:
    """Verify model hasn't been tampered with"""
    with open(model_path, 'rb') as f:
        model_data = f.read()
    
    actual_hash = hashlib.sha256(model_data).hexdigest()
    return actual_hash == expected_hash
```

### Network Security
```kotlin
// Secure network configuration
class NetworkSecurityConfig {
    companion object {
        const val CERTIFICATE_PINNING = true
        const val REQUIRE_HTTPS = true
        const val VERIFY_HOSTNAME = true
    }
}
```

This deployment guide ensures a smooth and secure deployment of the SAC-GRU Traffic Analyzer across all environments.

