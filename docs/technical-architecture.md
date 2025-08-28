# Technical Architecture - SAC-GRU Traffic Analyzer

## Overview

The SAC-GRU Traffic Analyzer employs a novel hybrid architecture combining **Soft Actor-Critic (SAC) reinforcement learning** with **Gated Recurrent Units (GRU)** for intelligent network traffic classification. The system is designed with a **laptop-to-Android deployment pipeline** that separates heavy training from lightweight inference.

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SAC-GRU Traffic Analyzer                     │
├─────────────────────────────────────────────────────────────────┤
│  Training Environment (Laptop/Server)                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Data Generation │  │ SAC-GRU Training│  │ Model Export    │ │
│  │ & Preprocessing │─▶│ (Actor+Critic)  │─▶│ (Actor Only)    │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│  Deployment Environment (Android)                              │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ TensorFlow Lite │  │ Real-time       │  │ UI & Results    │ │
│  │ Actor Inference │─▶│ Classification  │─▶│ Visualization   │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. SAC-GRU Neural Network Architecture

#### Soft Actor-Critic (SAC) Framework
```python
class SACGRUClassifier:
    def __init__(self, sequence_length, feature_dim, hidden_units):
        # Actor Network (Policy)
        self.actor = self._build_actor_network()
        
        # Critic Networks (Q-functions)
        self.critic_1 = self._build_critic_network()
        self.critic_2 = self._build_critic_network()
        
        # Target Networks
        self.target_critic_1 = self._build_critic_network()
        self.target_critic_2 = self._build_critic_network()
```

#### GRU Sequential Processing
```python
def _build_actor_network(self):
    return tf.keras.Sequential([
        tf.keras.layers.GRU(self.hidden_units, return_sequences=True),
        tf.keras.layers.GRU(self.hidden_units // 2),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')  # REEL vs NON-REEL
    ])
```

### 2. Training Pipeline Architecture

#### Data Flow
```
Raw Network Data → FastMassiveGenerator → Feature Extraction → 
Sequence Formation → SAC Training → Actor Extraction → TFLite Conversion
```

#### Training Components
- **FastMassiveGenerator**: Synthetic traffic data generation
- **Feature Extractor**: Network packet feature engineering
- **SAC Trainer**: Reinforcement learning optimization
- **Model Converter**: TensorFlow to TensorFlow Lite conversion

### 3. Android Application Architecture

#### MVVM Pattern with Jetpack Compose
```
┌─────────────────┐
│   Compose UI    │ ← User Interface Layer
├─────────────────┤
│   ViewModel     │ ← Business Logic Layer
├─────────────────┤
│   Repository    │ ← Data Access Layer
├─────────────────┤
│ TensorFlow Lite │ ← ML Inference Layer
└─────────────────┘
```

#### Key Android Components
- **MainActivity.kt**: Main application entry with Compose UI
- **TFLiteInferenceEngine**: Model loading and inference
- **NetworkTrafficAnalyzer**: Real-time traffic processing
- **ResultsViewModel**: State management and data binding

## Technical Specifications

### Model Architecture Details

#### SAC-GRU Network Topology
```
Input Layer (15 features) →
├─ Actor Branch:
│  ├─ GRU(128 units, return_sequences=True)
│  ├─ GRU(64 units)
│  ├─ Dense(64, ReLU)
│  └─ Dense(2, Softmax) → Action Probabilities
│
├─ Critic Branch 1:
│  ├─ GRU(128 units, return_sequences=True)
│  ├─ GRU(64 units)
│  ├─ Dense(64, ReLU)
│  └─ Dense(1, Linear) → Q-Value 1
│
└─ Critic Branch 2:
   ├─ GRU(128 units, return_sequences=True)
   ├─ GRU(64 units)
   ├─ Dense(64, ReLU)
   └─ Dense(1, Linear) → Q-Value 2
```

#### Feature Engineering
```python
NETWORK_FEATURES = [
    'packet_size', 'inter_arrival_time', 'protocol_type',
    'source_port', 'destination_port', 'tcp_flags',
    'payload_entropy', 'packet_direction', 'flow_duration',
    'bytes_per_second', 'packets_per_second', 'connection_state',
    'application_layer_protocol', 'geographic_location', 'time_of_day'
]
```

### Performance Optimizations

#### Training Optimizations
- **Experience Replay Buffer**: 10,000 samples
- **Soft Target Updates**: τ = 0.005
- **Entropy Regularization**: α = 0.2 (auto-tuned)
- **Learning Rate**: 3e-4 with Adam optimizer

#### Mobile Optimizations
- **Quantization**: INT8 quantization for 4x size reduction
- **Pruning**: Remove 30% of least important connections
- **Layer Fusion**: Combine consecutive operations
- **Memory Mapping**: Efficient model loading

## Data Architecture

### Training Data Pipeline
```
Synthetic Data Generation:
├─ Network Topology Simulation
├─ Traffic Pattern Generation
├─ Realistic Noise Injection
└─ Label Assignment (REEL/NON-REEL)

Feature Engineering:
├─ Packet-level Features
├─ Flow-level Statistics
├─ Temporal Patterns
└─ Behavioral Indicators

Sequence Formation:
├─ Sliding Window (sequence_length=50)
├─ Normalization (Min-Max scaling)
├─ Padding/Truncation
└─ Batch Formation
```

### Real-time Inference Pipeline
```
Network Traffic Capture:
├─ Packet Interception
├─ Feature Extraction
├─ Sequence Buffer Management
└─ Real-time Processing

TensorFlow Lite Inference:
├─ Model Loading (Memory Mapped)
├─ Input Preprocessing
├─ Forward Pass (<5ms)
└─ Output Post-processing
```

## Security & Privacy Architecture

### Privacy-First Design
- **On-device Processing**: No data leaves the device
- **Encrypted Storage**: Local model and data encryption
- **Minimal Permissions**: Only necessary Android permissions
- **Transparent AI**: Explainable decision making

### Security Measures
- **Model Integrity**: Cryptographic model verification
- **Secure Communication**: HTTPS for any external communication
- **Input Validation**: Robust input sanitization
- **Error Handling**: Graceful failure modes

## Scalability Architecture

### Horizontal Scaling
- **Distributed Training**: Multi-GPU SAC training support
- **Model Versioning**: A/B testing infrastructure
- **Edge Deployment**: Support for edge computing devices
- **Cloud Integration**: Optional cloud analytics

### Vertical Scaling
- **Memory Optimization**: Efficient memory usage patterns
- **CPU Optimization**: Multi-threading for inference
- **Battery Optimization**: Power-aware processing
- **Storage Optimization**: Compressed model storage

## Integration Architecture

### External Integrations
```python
# YouTube Integration
def handle_youtube_url(url: str) -> ClassificationResult:
    features = extract_youtube_features(url)
    return classify_traffic(features)

# Instagram Integration  
def handle_instagram_url(url: str) -> ClassificationResult:
    features = extract_instagram_features(url)
    return classify_traffic(features)
```

### API Architecture
```python
# Internal API Structure
class TrafficAnalyzerAPI:
    def classify_traffic(self, features: np.ndarray) -> Dict
    def get_model_info(self) -> ModelInfo
    def update_model(self, model_path: str) -> bool
    def get_performance_metrics(self) -> PerformanceMetrics
```

## Deployment Architecture

### Development Environment
- **Python 3.11+** with TensorFlow 2.14.0
- **Android Studio** with Kotlin 1.9.22
- **Gradle 8.4** with Android Gradle Plugin 8.2.2
- **Java 17** runtime environment

### Production Environment
- **Android 7.0+** (API level 24+)
- **ARM64/x86_64** processor architectures
- **4GB+ RAM** recommended
- **100MB** storage space

### CI/CD Pipeline
```
Code Commit → Automated Testing → Model Training → 
Model Validation → TFLite Conversion → Android Build → 
APK Generation → Deployment
```

This architecture ensures **scalable**, **efficient**, and **privacy-focused** network traffic analysis with state-of-the-art AI performance on mobile devices.

