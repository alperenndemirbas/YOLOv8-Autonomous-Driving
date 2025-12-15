import os
import keras
import keras_cv
import numpy as np

class YoloModel:
    """
    A wrapper class for loading the trained YOLOv8 model and running predictions.
    It handles Keras 3 compatibility and Non-Max Suppression (NMS) configuration.
    """
    def __init__(self, model_path):
        """
        Initializes the model.
        
        Args:
            model_path (str): Path to the .keras model file.
        """
        # 1. Validate file existence
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")

        print(f"Loading model from {model_path}...")
        
        # 2. Load the Keras model
        # compile=False: We skip compiling since we only need inference (no training).
        self.model = keras.models.load_model(model_path, compile=False)
        
        # 3. Configure Non-Max Suppression (NMS)
        # This layer filters out overlapping boxes and low-confidence predictions.
        self.model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
            bounding_box_format="xyxy",
            from_logits=False,
            iou_threshold=0.5,       # Intersection over Union threshold
            confidence_threshold=0.25 # Minimum confidence score to keep a box
        )
        print("Model loaded successfully with NMS configuration.")

    def predict(self, input_tensor):
        """
        Runs inference on the input tensor.

        Args:
            input_tensor (np.array): Preprocessed image tensor of shape (1, 640, 640, 3).

        Returns:
            dict: A dictionary containing 'boxes', 'classes', and 'confidence' scores.
        """
        # verbose=0: Suppress progress bar for cleaner logs
        predictions = self.model.predict(input_tensor, verbose=0)
        return predictions