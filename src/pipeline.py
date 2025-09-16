import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset

from export_model_to_tflite import main as run_training_and_export

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def feature_engineering_step(hf_dataset_name='VULCAN/sns-app-traffic-dataset'):
    """
    Loads and preprocesses the training data from Hugging Face Hub.
    """
    logger.info(f"Step 1: Loading and processing dataset '{hf_dataset_name}' from Hugging Face Hub...")

    try:
        dataset = load_dataset(hf_dataset_name, data_files="massive_balanced_training.csv")
        df = dataset['train'].to_pandas()
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face Hub: {e}")
        return None, None

    # Add new labels for more granular classification
    df['granular_label'] = df['label'].apply(lambda x: 'REEL' if x == 1 else 'NON-REEL')
    live_stream_mask = (df['fps'] > 50) & (df['bh'] > 10000)
    video_call_mask = (df['stalling'] > 10) & (df['qc'] > 5)
    
    df.loc[live_stream_mask, 'granular_label'] = 'LIVE-STREAM'
    df.loc[video_call_mask, 'granular_label'] = 'VIDEO-CALL'

    features_cols = ['fmt', 'fps', 'bh', 'droppedFrames', 'playedFrames', 'stalling', 'qc']
    X = df[features_cols].values
    
    # Encode the new granular labels
    labels_map = {label: i for i, label in enumerate(df['granular_label'].unique())}
    y = df['granular_label'].map(labels_map).values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info(f"Step 1: Feature Engineering complete with new labels: {labels_map}")
    return X_scaled, y

def validate_exported_model(model_path="sac_actor_model.tflite"):
    """
    Validates the exported TFLite model.
    """
    logger.info(f"Step 3: Validating the exported model: {model_path}")

    if not os.path.exists(model_path):
        logger.error(f"Validation failed: Model file not found at '{model_path}'")
        return False

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        dummy_input = np.random.rand(1, 1, 7).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        logger.info(f"Validation successful. Model produced an output: {output_data}")
        return True

    except Exception as e:
        logger.error(f"Validation failed with an error: {e}")
        return False

def main_pipeline():
    """
    Main pipeline for training and deployment.
    """
    logger.info("--- Starting the Main Training and Deployment Pipeline ---")

    features, labels = feature_engineering_step(hf_dataset_name='VULCAN/sns-reel-traffic-dataset')
    if features is None:
        logger.error("Pipeline stopped due to error in feature engineering.")
        return

    logger.info("Step 2: Kicking off training and model export...")
    run_training_and_export(features, labels)
    logger.info("Step 2: Training and model export completed.")

    validation_passed = validate_exported_model()

    if validation_passed:
        logger.info("--- Pipeline Finished Successfully ---")
    else:
        logger.error("--- Pipeline Finished with Errors ---")

if __name__ == "__main__":
    main_pipeline()
