#!/usr/bin/env python3
"""
SAC-GRU Reinforcement Learning Classifier for Traffic Analysis
=============================================================

This module implements a Soft Actor-Critic (SAC) reinforcement learning algorithm
combined with GRU (Gated Recurrent Unit) for network traffic classification.

The system learns to classify "Reel" vs "Non-Reel" traffic patterns through
reward-based learning, making it robust across varying network conditions.

Key Components:
- SAC Actor: Policy network that outputs classification probabilities
- SAC Critics: Q-value networks that estimate expected rewards
- GRU Encoder: Sequence modeling for temporal traffic patterns
- Replay Buffer: Stores experiences for off-policy learning
- Environment: Traffic stream simulation for training

Designed for deployment on Samsung phones with TensorFlow Lite.

Author: Enhanced by Manus AI based on SAC-GRU hybrid approach
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import logging
import pickle
import os
from collections import deque
import random
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class GRUEncoder(layers.Layer):
    """GRU-based sequence encoder for traffic features"""
    
    def __init__(self, hidden_units=64, dropout_rate=0.2, **kwargs):
        super(GRUEncoder, self).__init__(**kwargs)
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        
        # GRU layers for sequence processing
        self.gru1 = layers.GRU(hidden_units, return_sequences=True, dropout=dropout_rate)
        self.gru2 = layers.GRU(hidden_units, dropout=dropout_rate)
        self.batch_norm = layers.BatchNormalization()
        
    def call(self, inputs, training=None):
        """
        Process traffic sequence through GRU layers
        
        Args:
            inputs: Traffic sequence [batch_size, sequence_length, features]
            
        Returns:
            Hidden state representing sequence encoding
        """
        x = self.gru1(inputs, training=training)
        x = self.gru2(x, training=training)
        x = self.batch_norm(x, training=training)
        return x

class SACActorNetwork(Model):
    """SAC Actor Network - Policy for traffic classification"""
    
    def __init__(self, state_dim, action_dim=2, hidden_units=64):
        super(SACActorNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # GRU encoder for sequence processing
        self.gru_encoder = GRUEncoder(hidden_units)
        
        # Policy network layers
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.2)
        
        # Output layers for mean and log_std of policy
        self.mean_layer = layers.Dense(action_dim)
        self.log_std_layer = layers.Dense(action_dim)
        
        # For discrete actions (Reel vs Non-Reel)
        self.action_probs = layers.Dense(action_dim, activation='softmax')
        
    def call(self, state_sequence, training=None):
        """
        Forward pass through actor network
        
        Args:
            state_sequence: Traffic sequence [batch_size, seq_len, features]
            
        Returns:
            Action probabilities for Reel vs Non-Reel classification
        """
        # Encode sequence with GRU
        encoded_state = self.gru_encoder(state_sequence, training=training)
        
        # Policy network
        x = self.dense1(encoded_state)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        
        # Output action probabilities
        action_probs = self.action_probs(x)
        
        return action_probs
    
    def sample_action(self, state_sequence, training=True):
        """
        Sample action from policy with entropy regularization
        
        Args:
            state_sequence: Traffic sequence
            
        Returns:
            action, log_prob, entropy
        """
        action_probs = self(state_sequence, training=training)
        
        # Add small epsilon for numerical stability
        action_probs = tf.clip_by_value(action_probs, 1e-8, 1.0)
        
        # Sample action from categorical distribution
        dist = tf.random.categorical(tf.math.log(action_probs), 1)
        action = tf.squeeze(dist, axis=1)
        
        # Calculate log probability and entropy
        log_prob = tf.math.log(tf.gather(action_probs, action, batch_dims=1))
        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs), axis=1)
        
        return action, log_prob, entropy

class SACCriticNetwork(Model):
    """SAC Critic Network - Q-value estimation"""
    
    def __init__(self, state_dim, action_dim=2, hidden_units=64):
        super(SACCriticNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # GRU encoder for sequence processing
        self.gru_encoder = GRUEncoder(hidden_units)
        
        # Q-value network layers
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')
        self.dropout1 = layers.Dropout(0.3)
        self.dropout2 = layers.Dropout(0.2)
        
        # Output Q-values for each action
        self.q_values = layers.Dense(action_dim)
        
    def call(self, state_sequence, training=None):
        """
        Forward pass through critic network
        
        Args:
            state_sequence: Traffic sequence [batch_size, seq_len, features]
            
        Returns:
            Q-values for each action
        """
        # Encode sequence with GRU
        encoded_state = self.gru_encoder(state_sequence, training=training)
        
        # Q-value network
        x = self.dense1(encoded_state)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)
        x = self.dense3(x)
        
        # Output Q-values
        q_vals = self.q_values(x)
        
        return q_vals

class ReplayBuffer:
    """Experience replay buffer for SAC training"""
    
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample batch of experiences"""
        batch = random.sample(self.buffer, batch_size)
        
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class TrafficEnvironment:
    """Traffic stream environment for SAC training"""
    
    def __init__(self, sequence_length=20, feature_dim=5):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.current_sequence = None
        self.current_label = None
        
    def reset(self, traffic_sequence, true_label):
        """
        Reset environment with new traffic sequence
        
        Args:
            traffic_sequence: Packet sequence data
            true_label: Ground truth (0=Non-Reel, 1=Reel)
            
        Returns:
            Initial state
        """
        self.current_sequence = self._preprocess_sequence(traffic_sequence)
        self.current_label = true_label
        return self.current_sequence
    
    def step(self, action):
        """
        Take action in environment
        
        Args:
            action: Classification decision (0=Non-Reel, 1=Reel)
            
        Returns:
            next_state, reward, done, info
        """
        # Calculate reward based on correct classification
        if action == self.current_label:
            reward = 1.0  # Correct classification
        else:
            reward = -1.0  # Incorrect classification
        
        # Episode ends after one classification
        done = True
        next_state = self.current_sequence  # Same state since episode ends
        
        info = {
            'true_label': self.current_label,
            'predicted_label': action,
            'correct': action == self.current_label
        }
        
        return next_state, reward, done, info
    
    def _preprocess_sequence(self, traffic_sequence):
        """Preprocess traffic sequence to fixed length"""
        if len(traffic_sequence) == 0:
            return np.zeros((self.sequence_length, self.feature_dim))
        
        sequence = np.array(traffic_sequence)
        
        # Truncate or pad to sequence_length
        if len(sequence) > self.sequence_length:
            sequence = sequence[:self.sequence_length]
        elif len(sequence) < self.sequence_length:
            padding = np.zeros((self.sequence_length - len(sequence), self.feature_dim))
            sequence = np.vstack([sequence, padding])
        
        return sequence.astype(np.float32)

class SACGRUClassifier:
    """SAC-GRU Reinforcement Learning Classifier"""
    
    def __init__(self, sequence_length=20, feature_dim=5, hidden_units=64):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.hidden_units = hidden_units
        
        # SAC hyperparameters
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Soft update rate
        self.alpha = 0.2   # Entropy coefficient
        self.lr = 3e-4     # Learning rate
        
        # Networks
        self.actor = None
        self.critic1 = None
        self.critic2 = None
        self.target_critic1 = None
        self.target_critic2 = None
        
        # Training components
        self.replay_buffer = ReplayBuffer(capacity=100000)
        self.environment = TrafficEnvironment(sequence_length, feature_dim)
        
        # Optimizers
        self.actor_optimizer = None
        self.critic1_optimizer = None
        self.critic2_optimizer = None
        self.alpha_optimizer = None
        
        # Automatic entropy tuning
        self.target_entropy = -2.0  # For 2 actions
        self.log_alpha = tf.Variable(0.0, trainable=True)
        
        self.is_trained = False
        
    def build_networks(self):
        """Build SAC networks"""
        logger.info("Building SAC-GRU networks...")
        
        # Actor network
        self.actor = SACActorNetwork(
            state_dim=self.sequence_length * self.feature_dim,
            action_dim=2,
            hidden_units=self.hidden_units
        )
        
        # Critic networks (double Q-learning)
        self.critic1 = SACCriticNetwork(
            state_dim=self.sequence_length * self.feature_dim,
            action_dim=2,
            hidden_units=self.hidden_units
        )
        
        self.critic2 = SACCriticNetwork(
            state_dim=self.sequence_length * self.feature_dim,
            action_dim=2,
            hidden_units=self.hidden_units
        )
        
        # Target critics
        self.target_critic1 = SACCriticNetwork(
            state_dim=self.sequence_length * self.feature_dim,
            action_dim=2,
            hidden_units=self.hidden_units
        )
        
        self.target_critic2 = SACCriticNetwork(
            state_dim=self.sequence_length * self.feature_dim,
            action_dim=2,
            hidden_units=self.hidden_units
        )
        
        # Initialize target networks
        dummy_input = tf.random.normal((1, self.sequence_length, self.feature_dim))
        self.critic1(dummy_input)
        self.critic2(dummy_input)
        self.target_critic1(dummy_input)
        self.target_critic2(dummy_input)
        
        # Copy weights to target networks
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.target_critic2.set_weights(self.critic2.get_weights())
        
        # Optimizers
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.critic1_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.critic2_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.alpha_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        
        logger.info("SAC-GRU networks built successfully")
    
    def generate_training_data(self, num_episodes=10000):
        """Generate training data with Reel vs Non-Reel patterns"""
        logger.info(f"Generating {num_episodes} training episodes...")
        
        # Define traffic patterns for Reel vs Non-Reel
        reel_patterns = {
            'video_streaming': {
                'size_range': (800, 1500),  # Large packets for video
                'iat_range': (0.01, 0.05),  # Regular timing
                'direction_bias': 0.8,      # Mostly downlink
                'burst_probability': 0.7,   # Bursty video data
                'protocol': 6               # TCP for video
            },
            'short_video_requests': {
                'size_range': (64, 300),    # Small request packets
                'iat_range': (0.001, 0.01), # Fast requests
                'direction_bias': 0.3,      # Mostly uplink requests
                'burst_probability': 0.9,   # Frequent requests
                'protocol': 6               # TCP
            }
        }
        
        non_reel_patterns = {
            'web_browsing': {
                'size_range': (200, 800),   # Mixed sizes
                'iat_range': (0.1, 1.0),    # Slower, irregular
                'direction_bias': 0.6,      # Balanced
                'burst_probability': 0.3,   # Less bursty
                'protocol': 6               # TCP
            },
            'background_sync': {
                'size_range': (64, 200),    # Small packets
                'iat_range': (1.0, 10.0),   # Very slow
                'direction_bias': 0.5,      # Balanced
                'burst_probability': 0.1,   # Rare bursts
                'protocol': 6               # TCP
            }
        }
        
        training_data = []
        
        for episode in range(num_episodes):
            # Randomly choose Reel or Non-Reel
            is_reel = random.choice([True, False])
            
            if is_reel:
                pattern_name = random.choice(list(reel_patterns.keys()))
                pattern = reel_patterns[pattern_name]
                label = 1  # Reel
            else:
                pattern_name = random.choice(list(non_reel_patterns.keys()))
                pattern = non_reel_patterns[pattern_name]
                label = 0  # Non-Reel
            
            # Generate traffic sequence based on pattern
            sequence = self._generate_sequence_from_pattern(pattern)
            
            training_data.append((sequence, label))
            
            if episode % 1000 == 0:
                logger.info(f"Generated {episode}/{num_episodes} episodes...")
        
        logger.info(f"Training data generation complete: {len(training_data)} episodes")
        return training_data
    
    def _generate_sequence_from_pattern(self, pattern):
        """Generate traffic sequence based on pattern"""
        sequence = []
        current_time = 0
        
        # Generate sequence of packets
        for i in range(self.sequence_length):
            # Packet size
            size = random.randint(pattern['size_range'][0], pattern['size_range'][1])
            
            # Inter-arrival time
            iat = random.uniform(pattern['iat_range'][0], pattern['iat_range'][1])
            
            # Direction (0=downlink, 1=uplink)
            direction = 1 if random.random() < (1 - pattern['direction_bias']) else 0
            
            # Protocol
            protocol = pattern['protocol']
            
            current_time += iat
            
            # Features: [size, iat, direction, protocol, timestamp]
            sequence.append([size, iat, direction, protocol, current_time])
        
        return sequence
    
    def train(self, num_episodes=10000, batch_size=64, update_frequency=4):
        """Train SAC-GRU classifier"""
        if self.actor is None:
            self.build_networks()
        
        logger.info("Starting SAC-GRU training...")
        
        # Generate training data
        training_data = self.generate_training_data(num_episodes)
        
        # Training metrics
        episode_rewards = []
        episode_accuracies = []
        
        for episode in range(num_episodes):
            # Sample training episode
            sequence, true_label = random.choice(training_data)
            
            # Reset environment
            state = self.environment.reset(sequence, true_label)
            state = np.expand_dims(state, axis=0)  # Add batch dimension
            
            # Actor selects action
            action, log_prob, entropy = self.actor.sample_action(state)
            action_val = int(action.numpy()[0])
            
            # Environment step
            next_state, reward, done, info = self.environment.step(action_val)
            next_state = np.expand_dims(next_state, axis=0)
            
            # Store experience in replay buffer
            self.replay_buffer.push(
                state[0], action_val, reward, next_state[0], done
            )
            
            # Track metrics
            episode_rewards.append(reward)
            episode_accuracies.append(1.0 if info['correct'] else 0.0)
            
            # Update networks
            if len(self.replay_buffer) > batch_size and episode % update_frequency == 0:
                self._update_networks(batch_size)
            
            # Logging
            if episode % 1000 == 0:
                avg_reward = np.mean(episode_rewards[-1000:]) if episode_rewards else 0
                avg_accuracy = np.mean(episode_accuracies[-1000:]) if episode_accuracies else 0
                logger.info(f"Episode {episode}: Avg Reward = {avg_reward:.3f}, "
                           f"Avg Accuracy = {avg_accuracy:.3f}")
        
        self.is_trained = True
        logger.info("SAC-GRU training completed!")
        
        # Final metrics
        final_accuracy = np.mean(episode_accuracies[-1000:])
        final_reward = np.mean(episode_rewards[-1000:])
        
        return {
            'final_accuracy': final_accuracy,
            'final_reward': final_reward,
            'episode_rewards': episode_rewards,
            'episode_accuracies': episode_accuracies
        }
    
    def _update_networks(self, batch_size):
        """Update SAC networks"""
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Update critics
        self._update_critics(states, actions, rewards, next_states, dones)
        
        # Update actor
        self._update_actor(states)
        
        # Update target networks
        self._soft_update_targets()
        
        # Update alpha (entropy coefficient)
        self._update_alpha(states)
    
    @tf.function
    def _update_critics(self, states, actions, rewards, next_states, dones):
        """Update critic networks"""
        # Target Q-values
        next_actions, next_log_probs, _ = self.actor.sample_action(next_states)
        target_q1 = self.target_critic1(next_states)
        target_q2 = self.target_critic2(next_states)
        
        # Take minimum for double Q-learning
        target_q = tf.minimum(target_q1, target_q2)
        target_q_values = tf.gather(target_q, next_actions, batch_dims=1)
        
        # Add entropy term
        alpha = tf.exp(self.log_alpha)
        target_q_values = target_q_values - alpha * next_log_probs
        
        # Bellman backup
        target_values = rewards + self.gamma * (1 - dones) * target_q_values
        
        # Update critic 1
        with tf.GradientTape() as tape:
            q1_values = self.critic1(states)
            q1_selected = tf.gather(q1_values, actions, batch_dims=1)
            critic1_loss = tf.reduce_mean(tf.square(q1_selected - target_values))
        
        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        
        # Update critic 2
        with tf.GradientTape() as tape:
            q2_values = self.critic2(states)
            q2_selected = tf.gather(q2_values, actions, batch_dims=1)
            critic2_loss = tf.reduce_mean(tf.square(q2_selected - target_values))
        
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
    
    @tf.function
    def _update_actor(self, states):
        """Update actor network"""
        with tf.GradientTape() as tape:
            actions, log_probs, _ = self.actor.sample_action(states)
            
            q1_values = self.critic1(states)
            q2_values = self.critic2(states)
            q_values = tf.minimum(q1_values, q2_values)
            q_selected = tf.gather(q_values, actions, batch_dims=1)
            
            alpha = tf.exp(self.log_alpha)
            actor_loss = tf.reduce_mean(alpha * log_probs - q_selected)
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
    
    @tf.function
    def _update_alpha(self, states):
        """Update entropy coefficient"""
        with tf.GradientTape() as tape:
            _, log_probs, _ = self.actor.sample_action(states)
            alpha_loss = -tf.reduce_mean(self.log_alpha * (log_probs + self.target_entropy))
        
        alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
    
    def _soft_update_targets(self):
        """Soft update target networks"""
        # Update target critic 1
        for target_param, param in zip(self.target_critic1.trainable_variables, 
                                     self.critic1.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
        
        # Update target critic 2
        for target_param, param in zip(self.target_critic2.trainable_variables, 
                                     self.critic2.trainable_variables):
            target_param.assign(self.tau * param + (1 - self.tau) * target_param)
    
    def predict(self, traffic_sequence):
        """
        Predict Reel vs Non-Reel for traffic sequence
        
        Args:
            traffic_sequence: Packet sequence data
            
        Returns:
            Dictionary with prediction results
        """
        if not self.is_trained:
            logger.warning("Model not trained yet, training with default data...")
            self.train(num_episodes=5000)
        
        # Preprocess sequence
        state = self.environment._preprocess_sequence(traffic_sequence)
        state = np.expand_dims(state, axis=0)
        
        # Get action probabilities
        action_probs = self.actor(state, training=False)
        
        # Extract probabilities
        prob_non_reel = float(action_probs[0, 0])
        prob_reel = float(action_probs[0, 1])
        
        # Determine prediction
        predicted_action = 1 if prob_reel > prob_non_reel else 0
        confidence = max(prob_reel, prob_non_reel)
        
        # Map to labels
        traffic_label = "Reel" if predicted_action == 1 else "Non-Reel"
        
        return {
            'traffic_label': traffic_label,
            'confidence_score': confidence,
            'prob_reel': prob_reel,
            'prob_non_reel': prob_non_reel,
            'predicted_action': predicted_action,
            'sac_gru_prediction': True,
            'analysis_method': 'SAC-GRU-RL'
        }
    
    def export_actor_for_deployment(self, export_path):
        """
        Export only the actor network for deployment (TensorFlow Lite ready)
        
        Args:
            export_path: Path to save the actor model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before export")
        
        logger.info(f"Exporting actor network to {export_path}")
        
        # Save actor model
        self.actor.save(f"{export_path}_actor.h5")
        
        # Convert to TensorFlow Lite for mobile deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(self.actor)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(f"{export_path}_actor.tflite", "wb") as f:
            f.write(tflite_model)
        
        logger.info(f"Actor exported: {export_path}_actor.h5 and {export_path}_actor.tflite")
        
        # Save metadata
        metadata = {
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'hidden_units': self.hidden_units,
            'model_type': 'SAC-GRU-RL',
            'export_timestamp': datetime.now().isoformat(),
            'deployment_ready': True
        }
        
        with open(f"{export_path}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return f"{export_path}_actor.tflite"
    
    def save_full_model(self, filepath):
        """Save complete SAC-GRU model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        # Save all networks
        self.actor.save(f"{filepath}_actor.h5")
        self.critic1.save(f"{filepath}_critic1.h5")
        self.critic2.save(f"{filepath}_critic2.h5")
        
        # Save hyperparameters and metadata
        config = {
            'sequence_length': self.sequence_length,
            'feature_dim': self.feature_dim,
            'hidden_units': self.hidden_units,
            'gamma': self.gamma,
            'tau': self.tau,
            'alpha': self.alpha,
            'lr': self.lr,
            'is_trained': self.is_trained,
            'model_type': 'SAC-GRU-RL'
        }
        
        with open(f"{filepath}_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Full SAC-GRU model saved to {filepath}")

# Factory function
def create_sac_gru_classifier(sequence_length=20, feature_dim=5, hidden_units=64):
    """
    Factory function to create SAC-GRU classifier
    
    Args:
        sequence_length: Length of traffic sequences
        feature_dim: Number of features per packet
        hidden_units: Hidden units in GRU layers
        
    Returns:
        SACGRUClassifier instance
    """
    return SACGRUClassifier(sequence_length, feature_dim, hidden_units)

