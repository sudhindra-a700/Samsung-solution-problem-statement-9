#!/usr/bin/env python3
"""
SAC-GRU Laptop Training Pipeline with Actor Extraction
=====================================================

This script provides a complete training pipeline for SAC-GRU on laptop/desktop,
with the ability to extract only the trained actor network for Android deployment.

Key Features:
- Full SAC-GRU training on laptop with GPU support
- Actor network extraction for mobile deployment
- Automatic TensorFlow Lite conversion
- Model validation and performance testing
- Deployment-ready actor model generation

Usage:
    python laptop_training_pipeline.py --train --extract-actor --deploy-android

Author: Enhanced by Manus AI for SAC-GRU Traffic Analyzer
"""

import os
import sys
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pickle
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import time

# Import SAC-GRU components
from sac_gru_rl_classifier import SACGRUClassifier, SACActorNetwork, GRUEncoder
from fast_massive_generator import FastMassiveGenerator

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LaptopTrainingPipeline:
    """Complete training pipeline for laptop with actor extraction"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_dir = Path(config.get('model_dir', './models'))
        self.data_dir = Path(config.get('data_dir', './data'))
        self.android_dir = Path(config.get('android_dir', './Your-SAC-GRU-Android/app/src/main/assets'))
        
        # Create directories
        self.model_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)
        
        # Training components
        self.sac_gru_classifier = None
        self.actor_network = None
        self.training_history = {}
        
        logger.info(f"Training pipeline initialized")
        logger.info(f"Model directory: {self.model_dir}")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Android assets: {self.android_dir}")
    
    def generate_training_data(self, num_samples: int = 100000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate realistic training data for REEL vs NON-REEL classification"""
        logger.info(f"Generating {num_samples} training samples...")
        
        generator = FastMassiveGenerator()
        
        # Generate data in chunks to manage memory
        chunk_size = 10000
        all_features = []
        all_labels = []
        
        for i in range(0, num_samples, chunk_size):
            current_chunk_size = min(chunk_size, num_samples - i)
            chunk_data = generator.generate_chunk(current_chunk_size, i // chunk_size)
            
            # Extract features for each session
            for _, session_data in chunk_data.groupby('session_id'):
                # Aggregate session features
                features = self._extract_session_features(session_data)
                label = 1.0 if session_data['is_reel'].iloc[0] else 0.0
                
                all_features.append(features)
                all_labels.append(label)
        
        features = np.array(all_features, dtype=np.float32)
        labels = np.array(all_labels, dtype=np.float32)
        
        logger.info(f"Generated {len(features)} feature vectors")
        logger.info(f"Feature shape: {features.shape}")
        logger.info(f"REEL samples: {np.sum(labels):.0f} ({np.mean(labels):.2%})")
        
        return features, labels
    
    def _extract_session_features(self, session_data) -> np.ndarray:
        """Extract normalized features from session data"""
        # Aggregate session statistics
        features = np.array([
            session_data['fmt'].mean() / 2160.0,           # Normalized resolution
            session_data['fps'].mean() / 60.0,             # Normalized FPS
            session_data['buffer_health'].mean() / 25000.0, # Normalized buffer health
            session_data['stalling'].mean(),               # Stalling rate
            session_data['quality_changes'].mean(),        # Quality change rate
            len(session_data) / 20.0,                      # Session length (normalized)
            session_data['app'].iloc[0] / 5.0,             # App type (normalized)
            session_data['device'].iloc[0] / 3.0,          # Device type (normalized)
            session_data['network'].iloc[0] / 2.0,         # Network type (normalized)
            session_data['battery'].mean() / 100.0,       # Battery level (normalized)
            session_data['timestamp'].mean() / session_data['timestamp'].max() # Time phase
        ], dtype=np.float32)
        
        return features
    
    def create_sac_gru_model(self) -> SACGRUClassifier:
        """Create and configure SAC-GRU model for training"""
        logger.info("Creating SAC-GRU model...")
        
        # Model configuration
        config = {
            'state_dim': 11,  # 11 traffic features
            'action_dim': 2,  # Binary classification (REEL vs NON-REEL)
            'hidden_units': self.config.get('hidden_units', 64),
            'learning_rate': self.config.get('learning_rate', 0.001),
            'buffer_size': self.config.get('buffer_size', 100000),
            'batch_size': self.config.get('batch_size', 32),
            'tau': self.config.get('tau', 0.005),
            'gamma': self.config.get('gamma', 0.99),
            'alpha': self.config.get('alpha', 0.2)
        }
        
        # Create SAC-GRU classifier
        self.sac_gru_classifier = SACGRUClassifier(**config)
        
        logger.info("SAC-GRU model created successfully")
        return self.sac_gru_classifier
    
    def train_model(self, features: np.ndarray, labels: np.ndarray, 
                   epochs: int = 100, validation_split: float = 0.2) -> Dict[str, Any]:
        """Train the SAC-GRU model"""
        logger.info(f"Starting SAC-GRU training for {epochs} epochs...")
        
        # Split data
        split_idx = int(len(features) * (1 - validation_split))
        train_features, val_features = features[:split_idx], features[split_idx:]
        train_labels, val_labels = labels[:split_idx], labels[split_idx:]
        
        logger.info(f"Training samples: {len(train_features)}")
        logger.info(f"Validation samples: {len(val_features)}")
        
        # Training loop
        training_history = {
            'epoch': [],
            'actor_loss': [],
            'critic_loss': [],
            'reward': [],
            'accuracy': [],
            'val_accuracy': []
        }
        
        best_val_accuracy = 0.0
        patience_counter = 0
        patience = self.config.get('early_stopping_patience', 10)
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train for one epoch
            epoch_metrics = self._train_epoch(train_features, train_labels)
            
            # Validate
            val_accuracy = self._validate_model(val_features, val_labels)
            
            # Update history
            training_history['epoch'].append(epoch + 1)
            training_history['actor_loss'].append(epoch_metrics['actor_loss'])
            training_history['critic_loss'].append(epoch_metrics['critic_loss'])
            training_history['reward'].append(epoch_metrics['reward'])
            training_history['accuracy'].append(epoch_metrics['accuracy'])
            training_history['val_accuracy'].append(val_accuracy)
            
            # Early stopping
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, val_accuracy)
            else:
                patience_counter += 1
            
            logger.info(f"Validation accuracy: {val_accuracy:.4f} (best: {best_val_accuracy:.4f})")
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        self.training_history = training_history
        logger.info("Training completed!")
        return training_history
    
    def _train_epoch(self, features: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Train for one epoch"""
        batch_size = self.config.get('batch_size', 32)
        num_batches = len(features) // batch_size
        
        epoch_actor_loss = 0.0
        epoch_critic_loss = 0.0
        epoch_reward = 0.0
        correct_predictions = 0
        
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_features = features[start_idx:end_idx]
            batch_labels = labels[start_idx:end_idx]
            
            # Train SAC-GRU
            metrics = self.sac_gru_classifier.train_step(batch_features, batch_labels)
            
            epoch_actor_loss += metrics.get('actor_loss', 0.0)
            epoch_critic_loss += metrics.get('critic_loss', 0.0)
            epoch_reward += metrics.get('reward', 0.0)
            
            # Calculate accuracy
            predictions = self.sac_gru_classifier.predict(batch_features)
            correct_predictions += np.sum((predictions > 0.5) == (batch_labels > 0.5))
        
        return {
            'actor_loss': epoch_actor_loss / num_batches,
            'critic_loss': epoch_critic_loss / num_batches,
            'reward': epoch_reward / num_batches,
            'accuracy': correct_predictions / (num_batches * batch_size)
        }
    
    def _validate_model(self, features: np.ndarray, labels: np.ndarray) -> float:
        """Validate model and return accuracy"""
        predictions = self.sac_gru_classifier.predict(features)
        accuracy = np.mean((predictions > 0.5) == (labels > 0.5))
        return accuracy
    
    def _save_checkpoint(self, epoch: int, accuracy: float):
        """Save model checkpoint"""
        checkpoint_path = self.model_dir / f"sac_gru_checkpoint_epoch_{epoch}_acc_{accuracy:.4f}.h5"
        self.sac_gru_classifier.save_model(str(checkpoint_path))
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def extract_actor_network(self) -> keras.Model:
        """Extract only the actor network for deployment"""
        logger.info("Extracting actor network for deployment...")
        
        if self.sac_gru_classifier is None:
            raise ValueError("Model must be trained before extracting actor")
        
        # Get the trained actor network
        self.actor_network = self.sac_gru_classifier.actor
        
        # Create a standalone inference model
        input_layer = keras.layers.Input(shape=(11,), name='traffic_features')
        actor_output = self.actor_network(input_layer)
        
        # For binary classification, we want the probability of REEL
        # The actor outputs action probabilities, we need classification probability
        classification_output = keras.layers.Dense(1, activation='sigmoid', name='reel_probability')(actor_output)
        
        inference_model = keras.Model(inputs=input_layer, outputs=classification_output, name='SAC_Actor_Inference')
        
        logger.info(f"Actor network extracted successfully")
        logger.info(f"Input shape: {inference_model.input_shape}")
        logger.info(f"Output shape: {inference_model.output_shape}")
        
        return inference_model
    
    def convert_actor_to_tflite(self, actor_model: keras.Model, 
                               output_path: Optional[str] = None) -> str:
        """Convert actor network to TensorFlow Lite for Android"""
        logger.info("Converting actor to TensorFlow Lite...")
        
        if output_path is None:
            output_path = self.android_dir / "sac_actor_model.tflite"
        
        # Create TFLite converter
        converter = tf.lite.TFLiteConverter.from_keras_model(actor_model)
        
        # Optimization for mobile deployment
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        # Representative dataset for quantization
        def representative_data_gen():
            for _ in range(100):
                # Generate realistic traffic features
                sample = np.random.random((1, 11)).astype(np.float32)
                # Normalize to realistic ranges
                sample[0, 0] *= 1.0    # fmt
                sample[0, 1] *= 1.0    # fps
                sample[0, 2] *= 1.0    # buffer_health
                sample[0, 3] = np.random.choice([0.0, 1.0])  # stalling
                sample[0, 4] *= 1.0    # quality_changes
                sample[0, 5] *= 1.0    # session_length
                sample[0, 6] *= 1.0    # app
                sample[0, 7] *= 1.0    # device
                sample[0, 8] *= 1.0    # network
                sample[0, 9] *= 1.0    # battery
                sample[0, 10] *= 1.0   # timestamp
                yield [sample]
        
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
        
        # Convert
        tflite_model = converter.convert()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite actor model saved: {output_path}")
        logger.info(f"Model size: {len(tflite_model) / 1024:.2f} KB")
        
        return str(output_path)
    
    def validate_tflite_actor(self, tflite_path: str, test_features: np.ndarray, 
                             test_labels: np.ndarray) -> Dict[str, float]:
        """Validate the TFLite actor model"""
        logger.info("Validating TFLite actor model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Test predictions
        predictions = []
        inference_times = []
        
        for i in range(len(test_features)):
            sample = test_features[i:i+1].astype(np.float32)
            
            interpreter.set_tensor(input_details[0]['index'], sample)
            
            start_time = time.time()
            interpreter.invoke()
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            output = interpreter.get_tensor(output_details[0]['index'])
            predictions.append(output[0][0])
        
        predictions = np.array(predictions)
        
        # Calculate metrics
        accuracy = np.mean((predictions > 0.5) == (test_labels > 0.5))
        avg_inference_time = np.mean(inference_times)
        
        metrics = {
            'accuracy': accuracy,
            'avg_inference_time_ms': avg_inference_time,
            'min_inference_time_ms': np.min(inference_times),
            'max_inference_time_ms': np.max(inference_times),
            'throughput_fps': 1000 / avg_inference_time
        }
        
        logger.info("TFLite validation results:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return metrics
    
    def save_training_artifacts(self):
        """Save all training artifacts"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training history
        history_path = self.model_dir / f"training_history_{timestamp}.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save configuration
        config_path = self.model_dir / f"training_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Training artifacts saved with timestamp: {timestamp}")

def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='SAC-GRU Laptop Training Pipeline')
    parser.add_argument('--train', action='store_true', help='Train the SAC-GRU model')
    parser.add_argument('--extract-actor', action='store_true', help='Extract actor network')
    parser.add_argument('--deploy-android', action='store_true', help='Convert to TFLite for Android')
    parser.add_argument('--config', type=str, default='training_config.json', help='Configuration file')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--samples', type=int, default=100000, help='Number of training samples')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'hidden_units': 64,
        'learning_rate': 0.001,
        'batch_size': 32,
        'buffer_size': 100000,
        'early_stopping_patience': 10,
        'model_dir': './models',
        'data_dir': './data',
        'android_dir': './Your-SAC-GRU-Android/app/src/main/assets'
    }
    
    # Load configuration if exists
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    
    # Create pipeline
    pipeline = LaptopTrainingPipeline(config)
    
    if args.train:
        # Generate training data
        features, labels = pipeline.generate_training_data(args.samples)
        
        # Create and train model
        pipeline.create_sac_gru_model()
        training_history = pipeline.train_model(features, labels, args.epochs)
        
        # Save artifacts
        pipeline.save_training_artifacts()
    
    if args.extract_actor:
        # Extract actor network
        actor_model = pipeline.extract_actor_network()
        
        # Save actor model
        actor_path = pipeline.model_dir / "sac_actor_standalone.h5"
        actor_model.save(str(actor_path))
        logger.info(f"Standalone actor saved: {actor_path}")
    
    if args.deploy_android:
        if pipeline.actor_network is None:
            # Load actor if not already extracted
            actor_model = pipeline.extract_actor_network()
        else:
            actor_model = pipeline.actor_network
        
        # Convert to TFLite
        tflite_path = pipeline.convert_actor_to_tflite(actor_model)
        
        # Validate TFLite model
        features, labels = pipeline.generate_training_data(1000)  # Small test set
        metrics = pipeline.validate_tflite_actor(tflite_path, features, labels)
        
        logger.info("Android deployment ready!")
        logger.info(f"TFLite model: {tflite_path}")

if __name__ == "__main__":
    main()

