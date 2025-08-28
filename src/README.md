# SAC-GRU Traffic Analyzer

**A complete system for real-time network traffic classification using Soft Actor-Critic (SAC) and Gated Recurrent Units (GRU) with a native Android application for mobile deployment.**

This project provides a full-stack solution for analyzing network traffic, classifying it as REEL vs NON-REEL, and deploying the trained model on Android devices for real-time inference. The system is designed with a modular architecture, dependency injection, and comprehensive testing for reliability and scalability.

## Features

- **Advanced Machine Learning**: SAC-GRU reinforcement learning for high-accuracy traffic classification.
- **Modular Architecture**: High cohesion, low coupling design for maintainability.
- **Dependency Injection**: Flexible and testable components with IoC container.
- **Android Application**: Native Android app for real-time inference and testing.
- **Comprehensive Testing**: Unit tests, integration tests, and performance benchmarks.
- **Data Generation**: Fast massive dataset generator for realistic traffic simulation.
- **Model Export**: Script to convert trained models to TensorFlow Lite for mobile deployment.
- **Environment Setup**: Automated script for setting up the complete development environment.

## System Architecture

The system is divided into two main components:

1. **Python Backend**: For model training, data processing, and feature extraction.
2. **Android Frontend**: For real-time traffic classification on mobile devices.

### Python Backend Architecture

- **`sac_gru_rl_classifier.py`**: Core SAC-GRU implementation with TensorFlow.
- **`dependency_injection_system.py`**: Inversion of Control (IoC) container for managing services.
- **`modular_architecture_design.py`**: Interfaces and protocols for system components.
- **`core_modules_implementation.py`**: Implementation of core services (data ingestion, feature extraction, etc.).
- **`fast_massive_generator.py`**: Generates realistic training data.
- **`test_sac_gru_rl_implementation.py`**: Comprehensive test suite for the backend.

### Android Application Architecture

- **`MainActivity.java`**: Main activity with TensorFlow Lite integration and test cases.
- **`your_sac_gru_model.tflite`**: Pre-trained TensorFlow Lite model in the `assets` directory.
- **`activity_main.xml`**: UI layout for the Android application.
- **`build.gradle`**: Gradle build configuration with all dependencies.

## Getting Started

### Prerequisites

- **Python**: 3.8+ (3.9 recommended)
- **Java**: JDK 11+
- **Android Studio**: Latest version
- **Android SDK**: API level 34+

### 1. Setup Python Environment

First, set up the Python environment using the provided setup script. This will install all necessary dependencies from `requirements.txt`.

```bash
# Navigate to the project directory
cd /path/to/SAC-GRU-Traffic-Analyzer

# Run the environment setup script
python setup_environment.py --install-deps
```

This will validate your environment and ensure all required packages are installed.

### 2. Train the SAC-GRU Model

Before deploying to Android, you need to train the SAC-GRU model. Use the provided training scripts to start the training process.

```bash
# (Optional) Generate a new dataset
python fast_massive_generator.py

# Start the training process
python train_sac_gru_model.py --config config.ini
```

This will generate a trained model file (e.g., `trained_model.h5`).

### 3. Export Model to TensorFlow Lite

Once the model is trained, export it to TensorFlow Lite format for use in the Android app.

```bash
python export_model_to_tflite.py \
    --model_path trained_model.h5 \
    --output_path Your-SAC-GRU-Android/app/src/main/assets/your_sac_gru_model.tflite
```

This will create the `your_sac_gru_model.tflite` file in the Android project's `assets` directory.

### 4. Build and Run Android Application

Now you can open the Android project in Android Studio and run it on an emulator or physical device.

1. **Open Android Studio**.
2. **Select "Open an existing Android Studio project"**.
3. **Navigate to the `Your-SAC-GRU-Android` directory** and click **OK**.
4. **Wait for Gradle to sync**.
5. **Click the "Run" button** to build and run the app.

## Usage

### Python Backend

- **Training**: `python train_sac_gru_model.py`
- **Inference**: `python run_inference.py --pcap_file your_traffic.pcap`
- **Testing**: `pytest`

### Android Application

- **Run Tests**: Click the "Run SAC-GRU Tests" button to see real-time classification results.
- **View Results**: The results will be displayed in the scrollable text view.

## Documentation

- **`component_analysis.md`**: Detailed analysis of all system components.
- **`README.md`**: This file.
- **Python Docstrings**: All Python modules are fully documented with docstrings.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


