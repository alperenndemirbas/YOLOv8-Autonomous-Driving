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
            
            # --- YENİ DRIVE ID (Fixlendi: Concrete Function Version) ---
            file_id = '1OOPSfxiYIv1WvO7P8U0trgnK2IE_8_W8'
            url = f'https://drive.google.com/uc?id={file_id}'
            
            try:
                gdown.download(url, self.model_path, quiet=False)
                print("✅ Lite Model downloaded successfully.")
            except Exception as e:
                print(f"❌ Error downloading model: {e}")
                # Hata olsa bile devam etmeye çalışmayalım, durduralım
                raise e

        # 2. LOAD TFLITE INTERPRETER
        print(f"Loading TFLite interpreter from {self.model_path}...")
        try:
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
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
        input_data = input_tensor.astype(np.float32)
        
        # Giriş indeksini al (Genelde 0)
        input_index = self.input_details[0]['index']
        self.interpreter.set_tensor(input_index, input_data)

        # 2. Run Inference
        self.interpreter.invoke()

        # 3. Get Outputs
        results = {}
        
        # Çıktıları işle
        # Model export şekline göre çıktı sıralaması değişebilir, ama isimlerden yakalamaya çalışalım.
        for output in self.output_details:
            name = output['name'].lower()
            data = self.interpreter.get_tensor(output['index'])
            
            if 'boxes' in name:
                results['boxes'] = data
            elif 'classes' in name:
                results['classes'] = data
            elif 'confidence' in name or 'score' in name:
                results['confidence'] = data
            
        # Eğer isimler karıştıysa ve sonuç boşsa, index sırasına göre manuel ata (Yedek Plan)
        if 'boxes' not in results and len(self.output_details) >= 2:
             try:
                 # Genellikle sıra: Boxes, Classes, Confidence
                 results['boxes'] = self.interpreter.get_tensor(self.output_details[0]['index'])
                 results['classes'] = self.interpreter.get_tensor(self.output_details[1]['index'])
                 if len(self.output_details) > 2:
                     results['confidence'] = self.interpreter.get_tensor(self.output_details[2]['index'])
                 else:
                     # Confidence yoksa hepsini 1 kabul et
                     results['confidence'] = np.ones_like(results['classes'], dtype=np.float32)
             except Exception as e:
                 print(f"Warning: Output mapping failed: {e}")

        return results