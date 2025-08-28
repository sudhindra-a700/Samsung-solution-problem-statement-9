#!/usr/bin/env python3
"""
SAC-GRU Traffic Analyzer Environment Setup
==========================================

This script sets up the complete environment for the SAC-GRU Traffic Analyzer,
including Python dependencies, model training, and Android integration.

Features:
- Automatic dependency installation
- Environment validation
- Model training pipeline setup
- Android integration verification

Usage:
    python setup_environment.py [--install-deps] [--validate] [--train-model]
"""
import os
import sys
import subprocess
import argparse
import logging
import platform
from pathlib import Path
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    logger.info("Checking Python version...")
    
    version = sys.version_info
    if version.major != 3 or version.minor < 8:
        logger.error(f"Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    
    logger.info(f"Python {version.major}.{version.minor}.{version.micro} - OK")
    return True

def install_dependencies():
    """Install Python dependencies from requirements.txt"""
    logger.info("Installing Python dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        logger.error("requirements.txt not found!")
        return False
    
    try:
        # Upgrade pip first
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        logger.info("pip upgraded successfully")
        
        # Install requirements
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                      check=True, capture_output=True)
        logger.info("Dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False

def check_dependency(package_name: str, import_name: str = None) -> bool:
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        logger.info(f"✅ {package_name} - OK")
        return True
    except ImportError:
        logger.warning(f"❌ {package_name} - Missing")
        return False

def validate_environment():
    """Validate the complete environment setup"""
    logger.info("Validating environment...")
    
    # Check core dependencies
    dependencies = [
        ("tensorflow", "tensorflow"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("scapy", "scapy"),
        ("matplotlib", "matplotlib"),
        ("psutil", "psutil"),
    ]
    
    all_good = True
    for package, import_name in dependencies:
        if not check_dependency(package, import_name):
            all_good = False
    
    # Check TensorFlow GPU support
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            logger.info("✅ TensorFlow GPU support - Available")
        else:
            logger.info("ℹ️ TensorFlow GPU support - Not available (CPU only)")
    except Exception as e:
        logger.warning(f"Could not check GPU support: {e}")
    
    # Check system requirements
    logger.info(f"Platform: {platform.system()} {platform.release()}")
    logger.info(f"Architecture: {platform.machine()}")
    
    return all_good

def check_android_integration():
    """Check Android project integration"""
    logger.info("Checking Android integration...")
    
    android_project = Path(__file__).parent / "Your-SAC-GRU-Android"
    if not android_project.exists():
        logger.warning("Android project directory not found")
        return False
    
    # Check for TFLite model
    tflite_model = android_project / "app" / "src" / "main" / "assets" / "your_sac_gru_model.tflite"
    if tflite_model.exists():
        logger.info("✅ TensorFlow Lite model found in Android assets")
        logger.info(f"Model size: {tflite_model.stat().st_size / 1024:.2f} KB")
    else:
        logger.warning("❌ TensorFlow Lite model not found in Android assets")
    
    # Check for app icons
    icon_dirs = [
        android_project / "app" / "src" / "main" / "res" / "mipmap-hdpi",
        android_project / "app" / "src" / "main" / "res" / "mipmap-mdpi",
        android_project / "app" / "src" / "main" / "res" / "mipmap-xhdpi",
        android_project / "app" / "src" / "main" / "res" / "mipmap-xxhdpi",
        android_project / "app" / "src" / "main" / "res" / "mipmap-xxxhdpi",
    ]
    
    icons_found = 0
    for icon_dir in icon_dirs:
        if icon_dir.exists():
            launcher_icon = icon_dir / "ic_launcher.png"
            launcher_round = icon_dir / "ic_launcher_round.png"
            if launcher_icon.exists() and launcher_round.exists():
                icons_found += 1
    
    if icons_found == len(icon_dirs):
        logger.info("✅ All Android app icons found")
    else:
        logger.warning(f"❌ Missing app icons: {len(icon_dirs) - icons_found} directories incomplete")
    
    return True

def test_sac_gru_imports():
    """Test SAC-GRU specific imports"""
    logger.info("Testing SAC-GRU module imports...")
    
    try:
        # Test core modules
        from sac_gru_rl_classifier import SACGRUClassifier, SACActorNetwork, SACCriticNetwork
        logger.info("✅ SAC-GRU classifier imports - OK")
        
        from dependency_injection_system import ServiceFactory, DependencyContainer
        logger.info("✅ Dependency injection system - OK")
        
        from modular_architecture_design import ProcessingResult, PacketSequence
        logger.info("✅ Modular architecture design - OK")
        
        from core_modules_implementation import DataIngestionService, MLModelService
        logger.info("✅ Core modules implementation - OK")
        
        return True
        
    except ImportError as e:
        logger.error(f"SAC-GRU import failed: {e}")
        return False

def create_sample_config():
    """Create sample configuration files"""
    logger.info("Creating sample configuration...")
    
    config_content = """# SAC-GRU Traffic Analyzer Configuration
# ========================================

[model]
sequence_length = 20
hidden_units = 64
learning_rate = 0.001
batch_size = 32
epochs = 100

[data]
input_dir = ./data/pcap_files
output_dir = ./results
temp_dir = ./temp

[training]
train_split = 0.8
validation_split = 0.1
test_split = 0.1
early_stopping_patience = 10

[inference]
confidence_threshold = 0.5
batch_size = 1

[android]
model_path = ./Your-SAC-GRU-Android/app/src/main/assets/your_sac_gru_model.tflite
quantize = true
optimize_for_size = true
"""
    
    config_file = Path(__file__).parent / "config.ini"
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    logger.info(f"Sample configuration created: {config_file}")

def run_basic_tests():
    """Run basic functionality tests"""
    logger.info("Running basic functionality tests...")
    
    try:
        import numpy as np
        import tensorflow as tf
        
        # Test TensorFlow
        logger.info("Testing TensorFlow...")
        x = tf.constant([[1.0, 2.0, 3.0]])
        y = tf.constant([[4.0], [5.0], [6.0]])
        result = tf.matmul(x, y)
        logger.info(f"TensorFlow test result: {result.numpy()}")
        
        # Test model creation
        logger.info("Testing model creation...")
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation='relu', input_shape=(11,)),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Test prediction
        test_input = np.random.random((1, 11)).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        logger.info(f"Model test prediction: {prediction[0][0]:.4f}")
        
        logger.info("✅ Basic tests passed")
        return True
        
    except Exception as e:
        logger.error(f"Basic tests failed: {e}")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Setup SAC-GRU Traffic Analyzer environment')
    parser.add_argument('--install-deps', action='store_true',
                       help='Install Python dependencies')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate environment setup')
    parser.add_argument('--test', action='store_true', default=True,
                       help='Run basic functionality tests')
    parser.add_argument('--create-config', action='store_true',
                       help='Create sample configuration file')
    
    args = parser.parse_args()
    
    logger.info("SAC-GRU Traffic Analyzer Environment Setup")
    logger.info("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies if requested
    if args.install_deps:
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            sys.exit(1)
    
    # Validate environment
    if args.validate:
        if not validate_environment():
            logger.warning("Environment validation found issues")
        
        # Check Android integration
        check_android_integration()
        
        # Test SAC-GRU imports
        test_sac_gru_imports()
    
    # Run basic tests
    if args.test:
        if not run_basic_tests():
            logger.error("Basic tests failed")
            sys.exit(1)
    
    # Create sample config
    if args.create_config:
        create_sample_config()
    
    logger.info("Environment setup completed!")
    logger.info("Next steps:")
    logger.info("1. Train your SAC-GRU model using the training scripts")
    logger.info("2. Export the model to TensorFlow Lite using export_model_to_tflite.py")
    logger.info("3. Build and test the Android application")

if __name__ == "__main__":
    main()

