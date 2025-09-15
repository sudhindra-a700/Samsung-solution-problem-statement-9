import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from datasets import load_dataset

# Import the main function from your existing export script
from export_model_to_tflite import main as run_training_and_export

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def feature_engineering_step(hf_dataset_name='VULCAN/sns-reel-traffic-dataset'):
    """
    Loads the training data from the specified Hugging Face Hub repository.
    """
    logger.info(f"Step 1: Loading dataset '{hf_dataset_name}' from Hugging Face Hub...")

    try:
        # Load the specific CSV file from the repository
        dataset = load_dataset(hf_dataset_name, data_files="massive_balanced_training.csv")
        df = dataset['train'].to_pandas() # Convert to a pandas DataFrame
    except Exception as e:
        logger.error(f"Failed to load dataset from Hugging Face Hub: {e}")
        return None, None

    # Select the relevant features for the model
    features_cols = ['fmt', 'fps', 'bh', 'droppedFrames', 'playedFrames', 'stalling', 'qc']
    X = df[features_cols].values
    y = df['label'].values

    # Normalize the features to a range of [0, 1]
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    logger.info("Step 1: Feature Engineering complete.")
    return X_scaled, y


def validate_exported_model(model_path="sac_actor_model.tflite"):
    """
    Loads the exported TFLite model and runs a test inference to validate it.
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

        # Create a dummy input tensor that matches the model's expected input shape
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
    The main pipeline that orchestrates the entire workflow.
    """
    logger.info("--- Starting the Main Training and Deployment Pipeline ---")

    # --- Step 1: Feature Engineering from Hugging Face ---
    features, labels = feature_engineering_step(hf_dataset_name='VULCAN/sns-reel-traffic-dataset')
    if features is None:
        logger.error("Pipeline stopped due to error in feature engineering.")
        return

    # --- Step 2: Run the Training and Export Process ---
    logger.info("Step 2: Kicking off training and model export...")
    run_training_and_export(features, labels)
    logger.info("Step 2: Training and model export completed.")

    # --- Step 3: Validate the Exported Model ---
    validation_passed = validate_exported_model()

    if validation_passed:
        logger.info("--- Pipeline Finished Successfully ---")
    else:
        logger.error("--- Pipeline Finished with Errors ---")

if __name__ == "__main__":
    main_pipeline()
