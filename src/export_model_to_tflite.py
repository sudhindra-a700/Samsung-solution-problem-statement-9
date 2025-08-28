import logging
import numpy as np
from sac_gru_rl_classifier import SACGRUClassifier

# --- Setup logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(features, labels):
    """
    Trains the model using the provided data and exports the actor network.
    """
    logger.info("Training script initiated with preprocessed data...")

    if features is None or labels is None:
        logger.error("No data provided to the training script. Aborting.")
        return

    # The number of features is determined by the input data.
    feature_dim = features.shape[1]

    # Initialize the classifier
    classifier = SACGRUClassifier(sequence_length=1, feature_dim=feature_dim)
    
    # Train the model
    classifier.train(features, labels, num_episodes=len(features))

    # Export the trained actor model
    classifier.export_actor_for_deployment("sac_actor_model")
    
    logger.info("Training and export complete. 'sac_actor_model.tflite' is ready.")
