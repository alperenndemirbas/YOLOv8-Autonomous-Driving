import os
import numpy as np
import tensorflow as tf
import gdown

class YoloModel:
    """
    A wrapper class for loading the optimized TFLite YOLOv8 model.
    This version uses TensorFlow Lite Interpreter for low-memory inference (RAM friendly).
    """
    def __init__(self, model_path="models/yolov8_quantized.tflite"):
        """
        Initializes the TFLite model. If the model file is missing,
        it automatically downloads the optimized version from Google Drive.
        
        Args:
            model_path (str): Path to the .tflite model file.
        """
        self.model_path = model_path

        # 1. AUTO-DOWNLOAD LOGIC (Optimized TFLite Model)
        if not os.path.exists(self.model_path):
            print(f"⚠️ Lite Model file not found at: {self.model_path}")
            print("🌐 Downloading optimized TFLite model from Google Drive...")
            
            # Create directory if needed
            if os.path.dirname(self.model_path):
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # --- NEW GOOGLE DRIVE ID (TFLite Version) ---
            file_id = '1bcVmcB-rynYoQOteDcRFzEfS0CD-AIjW'
            url = f'https://drive.google.com/uc?id={file_id}'
            
            try:
                gdown.download(url, self.model_path, quiet=False)
                print("✅ Lite Model downloaded successfully.")
            except Exception as e:
                print(f"❌ Error downloading model: {e}")
                raise e

        # 2. LOAD TFLITE INTERPRETER
        # This is much lighter than loading the full Keras model.
        print(f"Loading TFLite interpreter from {self.model_path}...")
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details to understand the model structure
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("✅ TFLite Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Critical Error loading TFLite model: {e}")
            raise e

    def predict(self, input_tensor):
        """
        Runs inference using the TFLite Interpreter.

        Args:
            input_tensor (np.array): Preprocessed image tensor of shape (1, 640, 640, 3).

        Returns:
            dict: A dictionary containing 'boxes', 'classes', and 'confidence'.
        """
        # 1. Set Input Tensor
        # TFLite expects float32
        input_data = input_tensor.astype(np.float32)
        
        # We assume the model has 1 input (index 0)
        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, input_data)

        # 2. Run Inference
        self.interpreter.invoke()

        # 3. Get Outputs and Map to Dictionary
        # KerasCV exported models typically output a dictionary structure,
        # but in TFLite, they become separate tensors.
        # We need to map them back to keys expected by Streamlit.
        
        results = {}
        
        # Iterate through outputs to find boxes, classes, and scores
        # Note: The order depends on how the model was exported.
        # We try to match by name or common output shapes.
        
        for output in self.output_details:
            name = output['name'].lower()
            data = self.interpreter.get_tensor(output['index'])
            
            # Heuristic matching based on KerasCV naming conventions or output shapes
            # Boxes: Usually shape (1, N, 4)
            # Classes: Usually shape (1, N)
            # Confidence: Usually shape (1, N)
            
            if 'boxes' in name:
                results['boxes'] = data
            elif 'classes' in name:
                results['classes'] = data
            elif 'confidence' in name:
                results['confidence'] = data
            
        # Fallback: If names are mangled (common in TFLite), try to guess by index
        # Standard KerasCV export order often: boxes, classes, confidence (or scores)
        if 'boxes' not in results and len(self.output_details) >= 2:
            # Dangerous assumption but necessary if names are lost
            # Let's verify output count. If we have 2 outputs (boxes, classes), use indices.
            try:
                # This part might need adjustment depending on exact export structure
                # For now, we return what we found. If empty, we map by index.
                if len(self.output_details) >= 2:
                     results['boxes'] = self.interpreter.get_tensor(self.output_details[0]['index'])
                     results['classes'] = self.interpreter.get_tensor(self.output_details[1]['index'])
                     # Some exports merge confidence into classes or have a 3rd output
                     if len(self.output_details) > 2:
                         results['confidence'] = self.interpreter.get_tensor(self.output_details[2]['index'])
                     else:
                         # If confidence is missing, generate dummy 1.0s (rare case)
                         results['confidence'] = np.ones_like(results['classes'], dtype=np.float32)
            except Exception as e:
                print(f"Warning: Output mapping failed: {e}")

        return results