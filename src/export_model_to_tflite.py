#!/usr/bin/env python3
"""
SAC-GRU Model Export to TensorFlow Lite
======================================

This script exports the trained SAC-GRU model to TensorFlow Lite format
for deployment on Android devices.

Features:
- Model conversion with optimization
- Quantization for mobile deployment
- Model validation and testing
- Performance benchmarking

Usage:
    python export_model_to_tflite.py --model_path trained_model.h5 --output_path your_sac_gru_model.tflite

Author: Enhanced by Manus AI for SAC-GRU Traffic Analyzer
"""

import argparse
import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
import logging
import time
from typing import Optional, Tuple, List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_sac_gru_model(model_path: str) -> tf.keras.Model:
    """
    Load the trained SAC-GRU model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    logger.info(f"Loading SAC-GRU model from: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load model with custom objects if needed
    try:
        model = keras.models.load_model(model_path)
        logger.info(f"Model loaded successfully. Input shape: {model.input_shape}")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

def create_representative_dataset(num_samples: int = 100) -> callable:
    """
    Create representative dataset for quantization
    
    Args:
        num_samples: Number of samples for calibration
        
    Returns:
        Generator function for representative data
    """
    logger.info(f"Creating representative dataset with {num_samples} samples")
    
    def representative_data_gen():
        # Generate realistic traffic feature vectors
        # Features: [fmt, fps, bh, stalling, qc, phase, app, device, network, battery, timestamp]
        for _ in range(num_samples):
            # Simulate normalized traffic features
            sample = np.array([[
                np.random.uniform(0.1, 1.0),    # fmt (normalized resolution)
                np.random.uniform(0.4, 1.0),    # fps (normalized frame rate)
                np.random.uniform(0.2, 1.0),    # bh (normalized buffer health)
                np.random.choice([0.0, 1.0]),   # stalling (binary)
                np.random.uniform(0.0, 1.0),    # qc (quality changes)
                np.random.uniform(0.0, 1.0),    # phase (session phase)
                np.random.uniform(0.0, 1.0),    # app (app type)
                np.random.uniform(0.0, 1.0),    # device (device type)
                np.random.uniform(0.0, 1.0),    # network (network type)
                np.random.uniform(0.1, 1.0),    # battery (battery level)
                np.random.uniform(0.0, 1.0),    # timestamp (normalized)
            ]], dtype=np.float32)
            yield [sample]
    
    return representative_data_gen

def convert_to_tflite(model: tf.keras.Model, 
                     output_path: str,
                     quantize: bool = True,
                     optimize_for_size: bool = True) -> str:
    """
    Convert Keras model to TensorFlow Lite format
    
    Args:
        model: Trained Keras model
        output_path: Output path for TFLite model
        quantize: Whether to apply quantization
        optimize_for_size: Whether to optimize for model size
        
    Returns:
        Path to the converted model
    """
    logger.info("Converting model to TensorFlow Lite format")
    
    # Create TFLite converter
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set optimization flags
    if optimize_for_size:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        logger.info("Enabled default optimizations for size reduction")
    
    # Apply quantization if requested
    if quantize:
        logger.info("Applying post-training quantization")
        converter.representative_dataset = create_representative_dataset()
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    
    # Convert model
    try:
        logger.info("Starting model conversion...")
        start_time = time.time()
        tflite_model = converter.convert()
        conversion_time = time.time() - start_time
        
        logger.info(f"Model conversion completed in {conversion_time:.2f} seconds")
        logger.info(f"TFLite model size: {len(tflite_model) / 1024:.2f} KB")
        
        # Save the model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Model conversion failed: {e}")
        raise

def validate_tflite_model(tflite_path: str, 
                         original_model: tf.keras.Model,
                         num_test_samples: int = 10) -> Tuple[float, float]:
    """
    Validate the converted TFLite model
    
    Args:
        tflite_path: Path to TFLite model
        original_model: Original Keras model for comparison
        num_test_samples: Number of test samples
        
    Returns:
        Tuple of (accuracy_difference, avg_inference_time)
    """
    logger.info("Validating TFLite model")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    logger.info(f"TFLite input shape: {input_details[0]['shape']}")
    logger.info(f"TFLite output shape: {output_details[0]['shape']}")
    
    # Generate test data
    test_data = []
    for _ in range(num_test_samples):
        sample = np.array([[
            np.random.uniform(0.1, 1.0),    # fmt
            np.random.uniform(0.4, 1.0),    # fps
            np.random.uniform(0.2, 1.0),    # bh
            np.random.choice([0.0, 1.0]),   # stalling
            np.random.uniform(0.0, 1.0),    # qc
            np.random.uniform(0.0, 1.0),    # phase
            np.random.uniform(0.0, 1.0),    # app
            np.random.uniform(0.0, 1.0),    # device
            np.random.uniform(0.0, 1.0),    # network
            np.random.uniform(0.1, 1.0),    # battery
            np.random.uniform(0.0, 1.0),    # timestamp
        ]], dtype=np.float32)
        test_data.append(sample)
    
    # Compare predictions
    original_predictions = []
    tflite_predictions = []
    inference_times = []
    
    for sample in test_data:
        # Original model prediction
        orig_pred = original_model.predict(sample, verbose=0)
        original_predictions.append(orig_pred[0][0])
        
        # TFLite model prediction
        interpreter.set_tensor(input_details[0]['index'], sample)
        
        start_time = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        inference_times.append(inference_time)
        
        tflite_pred = interpreter.get_tensor(output_details[0]['index'])
        tflite_predictions.append(tflite_pred[0][0])
    
    # Calculate accuracy difference
    original_predictions = np.array(original_predictions)
    tflite_predictions = np.array(tflite_predictions)
    
    # Convert to binary classifications
    orig_binary = (original_predictions > 0.5).astype(int)
    tflite_binary = (tflite_predictions > 0.5).astype(int)
    
    accuracy_diff = np.mean(orig_binary == tflite_binary)
    avg_inference_time = np.mean(inference_times)
    
    logger.info(f"Model validation results:")
    logger.info(f"  Classification agreement: {accuracy_diff:.2%}")
    logger.info(f"  Average inference time: {avg_inference_time:.2f} ms")
    logger.info(f"  Prediction correlation: {np.corrcoef(original_predictions, tflite_predictions)[0,1]:.4f}")
    
    return accuracy_diff, avg_inference_time

def benchmark_model(tflite_path: str, num_iterations: int = 100) -> dict:
    """
    Benchmark the TFLite model performance
    
    Args:
        tflite_path: Path to TFLite model
        num_iterations: Number of benchmark iterations
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking model with {num_iterations} iterations")
    
    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    
    # Prepare test data
    test_sample = np.array([[
        0.5, 0.5, 0.5, 0.0, 0.2, 0.5, 0.3, 0.5, 0.8, 0.7, 0.5
    ]], dtype=np.float32)
    
    # Warm up
    for _ in range(10):
        interpreter.set_tensor(input_details[0]['index'], test_sample)
        interpreter.invoke()
    
    # Benchmark
    inference_times = []
    for _ in range(num_iterations):
        interpreter.set_tensor(input_details[0]['index'], test_sample)
        
        start_time = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start_time) * 1000
        inference_times.append(inference_time)
    
    # Calculate statistics
    inference_times = np.array(inference_times)
    results = {
        'mean_time_ms': np.mean(inference_times),
        'std_time_ms': np.std(inference_times),
        'min_time_ms': np.min(inference_times),
        'max_time_ms': np.max(inference_times),
        'p95_time_ms': np.percentile(inference_times, 95),
        'p99_time_ms': np.percentile(inference_times, 99),
        'throughput_fps': 1000 / np.mean(inference_times)
    }
    
    logger.info("Benchmark results:")
    for key, value in results.items():
        logger.info(f"  {key}: {value:.2f}")
    
    return results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Export SAC-GRU model to TensorFlow Lite')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained Keras model')
    parser.add_argument('--output_path', type=str, default='your_sac_gru_model.tflite',
                       help='Output path for TFLite model')
    parser.add_argument('--quantize', action='store_true', default=True,
                       help='Apply post-training quantization')
    parser.add_argument('--no-quantize', dest='quantize', action='store_false',
                       help='Disable quantization')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate converted model')
    parser.add_argument('--benchmark', action='store_true', default=True,
                       help='Benchmark model performance')
    
    args = parser.parse_args()
    
    try:
        # Load original model
        original_model = load_sac_gru_model(args.model_path)
        
        # Convert to TFLite
        tflite_path = convert_to_tflite(
            original_model, 
            args.output_path,
            quantize=args.quantize,
            optimize_for_size=True
        )
        
        # Validate model
        if args.validate:
            accuracy, avg_time = validate_tflite_model(tflite_path, original_model)
            logger.info(f"Validation completed: {accuracy:.2%} agreement, {avg_time:.2f}ms avg time")
        
        # Benchmark model
        if args.benchmark:
            benchmark_results = benchmark_model(tflite_path)
            logger.info("Benchmarking completed")
        
        logger.info("Model export completed successfully!")
        logger.info(f"TFLite model ready for Android deployment: {tflite_path}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

