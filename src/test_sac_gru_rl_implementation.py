#!/usr/bin/env python3
"""
Test Script for SAC-GRU Reinforcement Learning Implementation
===========================================================

This script comprehensively tests the SAC-GRU reinforcement learning implementation
for Reel vs Non-Reel traffic classification including:
- SAC (Soft Actor-Critic) algorithm validation
- GRU sequence modeling testing
- Reinforcement learning training pipeline
- Mobile deployment readiness
- Performance benchmarking

Author: Enhanced by Manus AI based on SAC-GRU RL approach
"""

import numpy as np
import tempfile
import os
import sys
import logging
import time
import tensorflow as tf

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sac_gru_rl_imports():
    """Test all SAC-GRU RL related imports"""
    logger.info("=== Testing SAC-GRU RL Imports ===")
    
    try:
        import tensorflow as tf
        logger.info(f"TensorFlow version: {tf.__version__}")
        
        from sac_gru_rl_classifier import (
            SACGRUClassifier, SACActorNetwork, SACCriticNetwork, 
            GRUEncoder, ReplayBuffer, TrafficEnvironment
        )
        logger.info("SAC-GRU RL classifier imports: SUCCESS")
        
        # Test if GPU is available
        if tf.config.list_physical_devices('GPU'):
            logger.info("GPU available for TensorFlow")
        else:
            logger.info("Running on CPU (no GPU detected)")
        
        return True
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        return False

def test_gru_encoder():
    """Test GRU encoder component"""
    logger.info("=== Testing GRU Encoder ===")
    
    try:
        from sac_gru_rl_classifier import GRUEncoder
        
        # Create GRU encoder
        encoder = GRUEncoder(hidden_units=64)
        
        # Test with dummy traffic sequence
        batch_size, seq_len, features = 10, 20, 5
        dummy_input = tf.random.normal((batch_size, seq_len, features))
        
        # Forward pass
        encoded_output = encoder(dummy_input)
        
        logger.info(f"Input shape: {dummy_input.shape}")
        logger.info(f"Encoded output shape: {encoded_output.shape}")
        
        # Verify output shape
        expected_shape = (batch_size, 64)  # hidden_units
        assert encoded_output.shape == expected_shape, f"Expected {expected_shape}, got {encoded_output.shape}"
        
        logger.info("GRU encoder test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"GRU encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sac_networks():
    """Test SAC Actor and Critic networks"""
    logger.info("=== Testing SAC Networks ===")
    
    try:
        from sac_gru_rl_classifier import SACActorNetwork, SACCriticNetwork
        
        state_dim = 20 * 5  # sequence_length * feature_dim
        action_dim = 2      # Reel vs Non-Reel
        
        # Create networks
        actor = SACActorNetwork(state_dim, action_dim)
        critic1 = SACCriticNetwork(state_dim, action_dim)
        critic2 = SACCriticNetwork(state_dim, action_dim)
        
        # Test with dummy input
        batch_size = 10
        dummy_state = tf.random.normal((batch_size, 20, 5))
        
        # Test actor
        action_probs = actor(dummy_state)
        logger.info(f"Actor output shape: {action_probs.shape}")
        logger.info(f"Action probabilities sample: {action_probs[0].numpy()}")
        
        # Verify probabilities sum to 1
        prob_sums = tf.reduce_sum(action_probs, axis=1)
        assert tf.reduce_all(tf.abs(prob_sums - 1.0) < 1e-6), "Action probabilities don't sum to 1"
        
        # Test action sampling
        actions, log_probs, entropy = actor.sample_action(dummy_state)
        logger.info(f"Sampled actions shape: {actions.shape}")
        logger.info(f"Log probabilities shape: {log_probs.shape}")
        logger.info(f"Entropy shape: {entropy.shape}")
        
        # Test critics
        q_values1 = critic1(dummy_state)
        q_values2 = critic2(dummy_state)
        
        logger.info(f"Critic 1 Q-values shape: {q_values1.shape}")
        logger.info(f"Critic 2 Q-values shape: {q_values2.shape}")
        
        # Verify Q-value shapes
        expected_q_shape = (batch_size, action_dim)
        assert q_values1.shape == expected_q_shape, f"Expected {expected_q_shape}, got {q_values1.shape}"
        assert q_values2.shape == expected_q_shape, f"Expected {expected_q_shape}, got {q_values2.shape}"
        
        logger.info("SAC networks test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"SAC networks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_replay_buffer():
    """Test replay buffer functionality"""
    logger.info("=== Testing Replay Buffer ===")
    
    try:
        from sac_gru_rl_classifier import ReplayBuffer
        
        # Create replay buffer
        buffer = ReplayBuffer(capacity=1000)
        
        # Add experiences
        for i in range(100):
            state = np.random.rand(20, 5)
            action = np.random.randint(0, 2)
            reward = np.random.rand()
            next_state = np.random.rand(20, 5)
            done = np.random.choice([True, False])
            
            buffer.push(state, action, reward, next_state, done)
        
        logger.info(f"Buffer size after adding 100 experiences: {len(buffer)}")
        
        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        logger.info(f"Sampled batch shapes:")
        logger.info(f"  States: {states.shape}")
        logger.info(f"  Actions: {actions.shape}")
        logger.info(f"  Rewards: {rewards.shape}")
        logger.info(f"  Next states: {next_states.shape}")
        logger.info(f"  Dones: {dones.shape}")
        
        # Verify batch shapes
        assert states.shape == (batch_size, 20, 5), f"Wrong states shape: {states.shape}"
        assert actions.shape == (batch_size,), f"Wrong actions shape: {actions.shape}"
        assert rewards.shape == (batch_size,), f"Wrong rewards shape: {rewards.shape}"
        assert next_states.shape == (batch_size, 20, 5), f"Wrong next_states shape: {next_states.shape}"
        assert dones.shape == (batch_size,), f"Wrong dones shape: {dones.shape}"
        
        logger.info("Replay buffer test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Replay buffer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_traffic_environment():
    """Test traffic environment for RL training"""
    logger.info("=== Testing Traffic Environment ===")
    
    try:
        from sac_gru_rl_classifier import TrafficEnvironment
        
        # Create environment
        env = TrafficEnvironment(sequence_length=20, feature_dim=5)
        
        # Generate test traffic sequence
        test_sequence = []
        for i in range(25):  # More than sequence_length to test truncation
            packet = [
                np.random.randint(64, 1500),    # size
                np.random.exponential(0.01),    # iat
                np.random.choice([0, 1]),       # direction
                np.random.choice([6, 17]),      # protocol
                i * 0.01                        # timestamp
            ]
            test_sequence.append(packet)
        
        # Test environment reset
        true_label = 1  # Reel
        state = env.reset(test_sequence, true_label)
        
        logger.info(f"Environment state shape: {state.shape}")
        logger.info(f"Expected shape: (20, 5)")
        
        assert state.shape == (20, 5), f"Wrong state shape: {state.shape}"
        
        # Test environment step
        action = 1  # Predict Reel
        next_state, reward, done, info = env.step(action)
        
        logger.info(f"Action: {action}")
        logger.info(f"Reward: {reward}")
        logger.info(f"Done: {done}")
        logger.info(f"Info: {info}")
        
        # Verify reward calculation
        expected_reward = 1.0 if action == true_label else -1.0
        assert reward == expected_reward, f"Wrong reward: {reward}, expected: {expected_reward}"
        
        # Test with wrong action
        env.reset(test_sequence, true_label)
        wrong_action = 0  # Predict Non-Reel when true label is Reel
        next_state, reward, done, info = env.step(wrong_action)
        
        assert reward == -1.0, f"Wrong reward for incorrect action: {reward}"
        assert not info['correct'], "Info should indicate incorrect prediction"
        
        logger.info("Traffic environment test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Traffic environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sac_gru_training():
    """Test SAC-GRU training pipeline"""
    logger.info("=== Testing SAC-GRU Training ===")
    
    try:
        from sac_gru_rl_classifier import SACGRUClassifier
        
        # Create classifier with small parameters for testing
        classifier = SACGRUClassifier(
            sequence_length=10,  # Smaller for faster testing
            feature_dim=5,
            hidden_units=32      # Smaller for faster testing
        )
        
        # Test training with few episodes
        logger.info("Starting small-scale training test...")
        training_stats = classifier.train(
            num_episodes=100,    # Small number for testing
            batch_size=16,       # Small batch size
            update_frequency=4
        )
        
        logger.info(f"Training completed!")
        logger.info(f"Final accuracy: {training_stats['final_accuracy']:.4f}")
        logger.info(f"Final reward: {training_stats['final_reward']:.4f}")
        
        # Verify training stats
        assert 'final_accuracy' in training_stats, "Missing final_accuracy in training stats"
        assert 'final_reward' in training_stats, "Missing final_reward in training stats"
        assert 'episode_rewards' in training_stats, "Missing episode_rewards in training stats"
        assert 'episode_accuracies' in training_stats, "Missing episode_accuracies in training stats"
        
        # Test prediction after training
        test_sequence = []
        for i in range(10):
            packet = [800, 0.02, 0, 6, i * 0.02]  # Video-like pattern
            test_sequence.append(packet)
        
        prediction = classifier.predict(test_sequence)
        
        logger.info(f"Test prediction: {prediction}")
        
        # Verify prediction structure
        required_fields = ['traffic_label', 'confidence_score', 'prob_reel', 'prob_non_reel']
        for field in required_fields:
            assert field in prediction, f"Missing field in prediction: {field}"
        
        assert prediction['traffic_label'] in ['Reel', 'Non-Reel'], f"Invalid traffic label: {prediction['traffic_label']}"
        assert 0 <= prediction['confidence_score'] <= 1, f"Invalid confidence score: {prediction['confidence_score']}"
        
        logger.info("SAC-GRU training test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"SAC-GRU training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_mobile_deployment():
    """Test mobile deployment export functionality"""
    logger.info("=== Testing Mobile Deployment Export ===")
    
    try:
        from sac_gru_rl_classifier import SACGRUClassifier
        
        # Create and train a small model
        classifier = SACGRUClassifier(sequence_length=10, feature_dim=5, hidden_units=32)
        
        # Quick training
        logger.info("Training model for deployment test...")
        classifier.train(num_episodes=50, batch_size=16)
        
        # Test export for mobile deployment
        export_path = "test_mobile_model"
        tflite_path = classifier.export_actor_for_deployment(export_path)
        
        logger.info(f"Model exported to: {tflite_path}")
        
        # Verify files were created
        assert os.path.exists(f"{export_path}_actor.h5"), "Actor H5 model not created"
        assert os.path.exists(f"{export_path}_actor.tflite"), "TensorFlow Lite model not created"
        assert os.path.exists(f"{export_path}_metadata.json"), "Metadata file not created"
        
        # Check TensorFlow Lite model size
        tflite_size = os.path.getsize(f"{export_path}_actor.tflite")
        logger.info(f"TensorFlow Lite model size: {tflite_size / 1024:.1f} KB")
        
        # Verify model can be loaded
        interpreter = tf.lite.Interpreter(model_path=f"{export_path}_actor.tflite")
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        logger.info(f"TFLite input shape: {input_details[0]['shape']}")
        logger.info(f"TFLite output shape: {output_details[0]['shape']}")
        
        # Test inference with TFLite model
        test_input = np.random.rand(1, 10, 5).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], test_input)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])
        
        logger.info(f"TFLite inference output: {output}")
        
        # Cleanup test files
        for ext in ['_actor.h5', '_actor.tflite', '_metadata.json']:
            try:
                os.remove(f"{export_path}{ext}")
            except:
                pass
        
        logger.info("Mobile deployment test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Mobile deployment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration_with_processor():
    """Test integration with main PCAP processor"""
    logger.info("=== Testing Integration with PCAP Processor ===")
    
    try:
        # Test imports from main processor
        from enhanced_pcap_processor_sac_gru_rl import SACGRUTrafficAnalyzer, Config
        
        logger.info("SAC-GRU RL processor imports: SUCCESS")
        
        # Create analyzer
        analyzer = SACGRUTrafficAnalyzer()
        
        # Generate test packet data (Reel-like pattern)
        test_packet_data = []
        current_time = 0
        for i in range(20):
            if i % 5 == 0:  # Video packets
                size = np.random.randint(800, 1500)
                iat = np.random.uniform(0.01, 0.05)
                direction = 0  # Downlink
            else:  # Request packets
                size = np.random.randint(64, 300)
                iat = np.random.uniform(0.001, 0.01)
                direction = 1  # Uplink
            
            protocol = 6  # TCP
            current_time += iat
            test_packet_data.append([size, iat, direction, protocol, current_time])
        
        # Test analysis (this will trigger model initialization and training)
        logger.info("Testing packet sequence analysis...")
        start_time = time.time()
        
        analysis_result = analyzer.analyze_packet_sequence(test_packet_data)
        
        end_time = time.time()
        analysis_time = end_time - start_time
        
        logger.info(f"Analysis completed in {analysis_time:.2f} seconds")
        logger.info(f"Analysis result: {analysis_result}")
        
        # Verify analysis result structure
        required_fields = [
            'traffic_label', 'confidence_score', 'analysis_method', 
            'model_type', 'prob_reel', 'prob_non_reel'
        ]
        for field in required_fields:
            assert field in analysis_result, f"Missing required field: {field}"
        
        assert analysis_result['analysis_method'] == 'SAC-GRU-RL', "Wrong analysis method"
        assert analysis_result['model_type'] == 'Reinforcement Learning', "Wrong model type"
        
        logger.info("Integration test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance_benchmarking():
    """Test performance of SAC-GRU RL classifier"""
    logger.info("=== Testing Performance Benchmarking ===")
    
    try:
        from sac_gru_rl_classifier import SACGRUClassifier
        
        # Create classifier
        classifier = SACGRUClassifier(sequence_length=20, feature_dim=5, hidden_units=64)
        
        # Quick training
        logger.info("Training model for performance testing...")
        classifier.train(num_episodes=200, batch_size=32)
        
        # Test prediction performance
        test_sizes = [1, 10, 50, 100]
        
        for size in test_sizes:
            # Generate test sequences
            test_sequences = []
            for _ in range(size):
                sequence = []
                for i in range(20):
                    packet = [
                        np.random.randint(64, 1500),
                        np.random.exponential(0.01),
                        np.random.choice([0, 1]),
                        np.random.choice([6, 17]),
                        i * 0.01
                    ]
                    sequence.append(packet)
                test_sequences.append(sequence)
            
            # Measure prediction time
            start_time = time.time()
            for sequence in test_sequences:
                result = classifier.predict(sequence)
            end_time = time.time()
            
            prediction_time = end_time - start_time
            rate = size / prediction_time if prediction_time > 0 else float('inf')
            
            logger.info(f"Batch size {size}: {prediction_time:.4f}s ({rate:.0f} sequences/second)")
        
        logger.info("Performance benchmarking test: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Performance benchmarking test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_comprehensive_tests():
    """Run all SAC-GRU RL tests"""
    logger.info("Starting Comprehensive SAC-GRU RL Tests...")
    
    tests = [
        ("Import Tests", test_sac_gru_rl_imports),
        ("GRU Encoder", test_gru_encoder),
        ("SAC Networks", test_sac_networks),
        ("Replay Buffer", test_replay_buffer),
        ("Traffic Environment", test_traffic_environment),
        ("SAC-GRU Training", test_sac_gru_training),
        ("Mobile Deployment", test_mobile_deployment),
        ("Integration", test_integration_with_processor),
        ("Performance", test_performance_benchmarking)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            start_time = time.time()
            result = test_func()
            end_time = time.time()
            
            results[test_name] = {
                'passed': result,
                'time': end_time - start_time
            }
            
            status = "PASSED" if result else "FAILED"
            logger.info(f"{test_name}: {status} ({end_time - start_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"{test_name}: FAILED with exception: {e}")
            results[test_name] = {
                'passed': False,
                'time': 0,
                'error': str(e)
            }
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed_tests = sum(1 for r in results.values() if r['passed'])
    total_tests = len(results)
    
    for test_name, result in results.items():
        status = "PASSED" if result['passed'] else "FAILED"
        time_str = f"({result['time']:.2f}s)" if result['time'] > 0 else ""
        logger.info(f"{test_name}: {status} {time_str}")
        
        if not result['passed'] and 'error' in result:
            logger.info(f"  Error: {result['error']}")
    
    logger.info(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        logger.info("🎉 All SAC-GRU RL tests PASSED!")
        return True
    else:
        logger.error(f"❌ {total_tests - passed_tests} tests FAILED")
        return False

def main():
    """Main test function"""
    logger.info("SAC-GRU Reinforcement Learning Implementation Test Suite")
    logger.info("=" * 60)
    
    success = run_comprehensive_tests()
    
    if success:
        logger.info("\n✅ SAC-GRU RL implementation is ready for production use!")
        logger.info("📱 Model can be deployed on Samsung phones with TensorFlow Lite")
    else:
        logger.error("\n❌ SAC-GRU RL implementation has issues that need to be addressed.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

