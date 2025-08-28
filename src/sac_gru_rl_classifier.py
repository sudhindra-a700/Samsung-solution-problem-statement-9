import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import logging
from collections import deque
import random

logger = logging.getLogger(__name__)

class GRUEncoder(layers.Layer):
    """GRU-based sequence encoder for traffic features."""
    def __init__(self, hidden_units=64, dropout_rate=0.2, **kwargs):
        super(GRUEncoder, self).__init__(**kwargs)
        self.gru1 = layers.GRU(hidden_units, return_sequences=True, dropout=dropout_rate)
        self.gru2 = layers.GRU(hidden_units, dropout=dropout_rate)
        self.batch_norm = layers.BatchNormalization()
        
    def call(self, inputs, training=None):
        x = self.gru1(inputs, training=training)
        x = self.gru2(x, training=training)
        return self.batch_norm(x, training=training)

class SACActorNetwork(Model):
    """SAC Actor Network - The policy model for classification."""
    def __init__(self, state_dim, action_dim=2, hidden_units=64):
        super(SACActorNetwork, self).__init__()
        self.gru_encoder = GRUEncoder(hidden_units)
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.action_probs = layers.Dense(action_dim, activation='softmax')
        
    def call(self, state_sequence, training=None):
        encoded_state = self.gru_encoder(state_sequence, training=training)
        x = self.dense1(encoded_state)
        x = self.dense2(x)
        return self.action_probs(x)
    
    def sample_action(self, state_sequence, training=True):
        action_probs = self(state_sequence, training=training)
        action_probs = tf.clip_by_value(action_probs, 1e-8, 1.0)
        dist = tf.random.categorical(tf.math.log(action_probs), 1)
        action = tf.squeeze(dist, axis=1)
        log_prob = tf.math.log(tf.gather(action_probs, action, batch_dims=1))
        return action, log_prob

class SACCriticNetwork(Model):
    """SAC Critic Network - Q-value estimation."""
    def __init__(self, state_dim, action_dim=2, hidden_units=64):
        super(SACCriticNetwork, self).__init__()
        self.gru_encoder = GRUEncoder(hidden_units)
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.q_values = layers.Dense(action_dim)
        
    def call(self, state_sequence, training=None):
        encoded_state = self.gru_encoder(state_sequence, training=training)
        x = self.dense1(encoded_state)
        x = self.dense2(x)
        return self.q_values(x)

class ReplayBuffer:
    """Experience replay buffer for SAC training."""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)

class SACGRUClassifier:
    """Full SAC-GRU Reinforcement Learning Classifier."""
    
    def __init__(self, sequence_length=1, feature_dim=7, hidden_units=64):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.lr = 3e-4
        self.gamma = 0.99
        self.tau = 0.005
        self.is_trained = False
        
        state_dim = self.sequence_length * self.feature_dim
        self.actor = SACActorNetwork(state_dim, hidden_units=hidden_units)
        self.critic1 = SACCriticNetwork(state_dim, hidden_units=hidden_units)
        self.critic2 = SACCriticNetwork(state_dim, hidden_units=hidden_units)
        self.target_critic1 = SACCriticNetwork(state_dim, hidden_units=hidden_units)
        self.target_critic2 = SACCriticNetwork(state_dim, hidden_units=hidden_units)

        # Initialize target networks
        dummy_input = tf.random.normal((1, self.sequence_length, self.feature_dim))
        self.critic1(dummy_input)
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.critic2(dummy_input)
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.critic1_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        self.critic2_optimizer = keras.optimizers.Adam(learning_rate=self.lr)
        
        self.replay_buffer = ReplayBuffer()

    def train(self, features, labels, num_episodes, batch_size=64):
        """
        Pre-trains the RL agent using a labeled dataset.
        """
        logger.info(f"Starting supervised pre-training for {num_episodes} episodes...")

        for episode in range(num_episodes):
            idx = np.random.randint(0, len(features))
            feature_set = features[idx]
            true_label = labels[idx]
            
            state = np.reshape(feature_set, (1, self.sequence_length, self.feature_dim))
            
            action, _ = self.actor.sample_action(state)
            action_val = int(action.numpy()[0])
            
            # Use the label to generate a reward
            reward = 1.0 if action_val == true_label else -1.0
            
            self.replay_buffer.push(state[0], action_val, reward, state[0], True)
            
            if len(self.replay_buffer) > batch_size:
                self._update_networks(batch_size)

            if (episode + 1) % 5000 == 0:
                logger.info(f"Pre-training episode {episode + 1}/{num_episodes} completed.")

        self.is_trained = True
        logger.info("Supervised pre-training complete!")

    def _update_networks(self, batch_size):
        """Performs one step of SAC update."""
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            # Critic Loss
            next_actions, _ = self.actor.sample_action(next_states)
            target_q1 = self.target_critic1(next_states)
            target_q2 = self.target_critic2(next_states)
            target_q = tf.minimum(target_q1, target_q2)
            target_q_values = tf.gather(target_q, next_actions, batch_dims=1)
            target_values = rewards + self.gamma * (1 - dones) * tf.squeeze(target_q_values, axis=1)

            q1_values = self.critic1(states)
            q1_selected = tf.gather(q1_values, actions, batch_dims=1)
            critic1_loss = tf.reduce_mean(tf.square(tf.squeeze(q1_selected, axis=1) - target_values))

            q2_values = self.critic2(states)
            q2_selected = tf.gather(q2_values, actions, batch_dims=1)
            critic2_loss = tf.reduce_mean(tf.square(tf.squeeze(q2_selected, axis=1) - target_values))

            # Actor Loss
            actions_pred, log_probs = self.actor.sample_action(states)
            q1_pred = self.critic1(states)
            q2_pred = self.critic2(states)
            q_pred = tf.minimum(q1_pred, q2_pred)
            actor_loss = tf.reduce_mean(tf.gather(q_pred, actions_pred, batch_dims=1) - log_probs)

        critic1_grads = tape.gradient(critic1_loss, self.critic1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(critic1_grads, self.critic1.trainable_variables))
        
        critic2_grads = tape.gradient(critic2_loss, self.critic2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(critic2_grads, self.critic2.trainable_variables))
        
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
        
        del tape
        self._soft_update_targets()

    def _soft_update_targets(self):
        """Softly updates the weights of the target networks."""
        for target, source in zip(self.target_critic1.variables, self.critic1.variables):
            target.assign(self.tau * source + (1.0 - self.tau) * target)
        for target, source in zip(self.target_critic2.variables, self.critic2.variables):
            target.assign(self.tau * source + (1.0 - self.tau) * target)

    def export_actor_for_deployment(self, export_path):
        """Exports the trained actor network to a TFLite file."""
        if not self.is_trained:
            raise ValueError("Model must be trained before export")
        
        logger.info(f"Exporting actor network to {export_path}.tflite")
        
        converter = tf.lite.TFLiteConverter.from_keras_model(self.actor)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        with open(f"{export_path}.tflite", "wb") as f:
            f.write(tflite_model)
        
        logger.info(f"Actor exported successfully to {export_path}.tflite")
