import os
import keras
import keras_cv
import numpy as np
import gdown

class YoloModel:
    """
    A wrapper class for loading the trained YOLOv8 model and running predictions.
    It handles Keras 3 compatibility, automatic model downloading for cloud deployment,
    and Non-Max Suppression (NMS) configuration.
    """
    def __init__(self, model_path):
        """
        Initializes the model. If the model file is missing (e.g., on a cloud server),
        it automatically downloads it from Google Drive.
        
        Args:
            model_path (str): Path to the .keras model file.
        """
        # 1. AUTO-DOWNLOAD LOGIC (Crucial for Render/Cloud Deployment)
        # Check if the model exists locally. If not, download it using gdown.
        if not os.path.exists(model_path):
            print(f"⚠️ Model file not found at: {model_path}")
            print("🌐 Downloading from Google Drive for deployment...")
            
            # Create the directory (e.g., 'models/') if it doesn't exist
            if os.path.dirname(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Google Drive File ID for the pre-trained YOLOv8 model
            # Note: This is a direct link to the large model file hosted on GDrive.
            file_id = '1JZ0OmNuOIK8l4xxo5KoThcNpykMzsCtq'
            url = f'https://drive.google.com/uc?id={file_id}'
            
            # Start the download
            try:
                gdown.download(url, model_path, quiet=False)
                print("✅ Model downloaded successfully.")
            except Exception as e:
                print(f"❌ Error downloading model: {e}")
                raise e

        print(f"Loading model from {model_path}...")
        
        # 2. LOAD THE KERAS MODEL
        # compile=False: We skip compilation since we only need inference (prediction), not training.
        try:
            self.model = keras.models.load_model(model_path, compile=False)
        except Exception as e:
            print(f"❌ Critical Error: Could not load the model file. Details: {e}")
            raise e
        
        # 3. CONFIGURE NON-MAX SUPPRESSION (NMS)
        # This layer filters out overlapping bounding boxes and removes low-confidence predictions.
        self.model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
            bounding_box_format="xyxy",
            from_logits=False,
            iou_threshold=0.5,        # Intersection over Union threshold (removes duplicates)
            confidence_threshold=0.25 # Minimum confidence score to keep a detection
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
        # verbose=0: Suppress Keras progress bar for cleaner logs in production
        predictions = self.model.predict(input_tensor, verbose=0)
        return predictions