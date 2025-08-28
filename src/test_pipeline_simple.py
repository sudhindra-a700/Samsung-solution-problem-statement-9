#!/usr/bin/env python3
"""Simple test of the training pipeline components"""

import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Reduce TF logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def test_data_generation():
    """Test data generation with proper type handling"""
    print("Testing data generation...")
    
    from fast_massive_generator import FastMassiveGenerator
    generator = FastMassiveGenerator()
    
    # Generate small test data
    test_data = generator.generate_chunk(100, 0)
    print(f"✅ Generated {len(test_data)} rows")
    
    # Extract features manually with proper type handling
    features = []
    labels = []
    
    for _, row in test_data.iterrows():
        # Handle string phases
        phase_map = {'startup': 0.1, 'steady': 0.5, 'depletion': 0.9}
        phase_val = phase_map.get(row['phase'], 0.5)
        
        # Handle string apps
        app_map = {'youtube': 0.4, 'instagram': 0.7, 'tiktok': 0.9, 
                   'facebook': 0.6, 'twitter': 0.5, 'snapchat': 0.8}
        app_val = app_map.get(row['app'], 0.5)
        
        # Extract normalized features
        feature_vector = np.array([
            row['fmt'] / 2160.0,                    # Normalized resolution
            row['fps'] / 60.0,                      # Normalized FPS
            row['bh'] / 25000.0,                    # Normalized buffer health
            1.0 if row['stalling'] > 0 else 0.0,   # Stalling (binary)
            row['qc'] / 10.0,                       # Quality changes
            phase_val,                              # Phase (mapped)
            app_val,                                # App (mapped)
            0.5,                                    # Device type (mobile)
            0.7,                                    # Network type (WiFi)
            np.random.uniform(0.2, 1.0),           # Battery level
            phase_val                               # Time phase
        ], dtype=np.float32)
        
        features.append(feature_vector)
        labels.append(float(row['label']))
    
    features = np.array(features)
    labels = np.array(labels)
    
    print(f"✅ Features shape: {features.shape}")
    print(f"✅ Labels shape: {labels.shape}")
    print(f"✅ REEL ratio: {labels.mean():.2%}")
    
    return features, labels

def test_simple_model():
    """Test a simple TensorFlow model"""
    print("\nTesting simple model creation...")
    
    # Create a simple model for binary classification
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(11,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✅ Model created successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    
    return model

def test_tflite_conversion(model, features):
    """Test TensorFlow Lite conversion"""
    print("\nTesting TFLite conversion...")
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    tflite_path = "test_actor_model.tflite"
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ TFLite model created: {len(tflite_model) / 1024:.2f} KB")
    
    # Test inference
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Test with sample data
    sample = features[:1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], sample)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"✅ TFLite inference test passed: {output[0][0]:.4f}")
    
    return tflite_path

def main():
    """Run all tests"""
    print("SAC-GRU Pipeline Component Tests")
    print("=" * 40)
    
    try:
        # Test 1: Data generation
        features, labels = test_data_generation()
        
        # Test 2: Simple model
        model = test_simple_model()
        
        # Test 3: TFLite conversion
        tflite_path = test_tflite_conversion(model, features)
        
        print("\n" + "=" * 40)
        print("✅ All tests passed successfully!")
        print(f"✅ TFLite model ready: {tflite_path}")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
