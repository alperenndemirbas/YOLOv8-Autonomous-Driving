import os
import numpy as np
import cv2
import gdown

# Attempt to import TFLite runtime, fall back to full TensorFlow if not available
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        raise ImportError("ERROR: Neither 'tflite-runtime' nor 'tensorflow' is installed!")

class YoloModel:
    """
    TFLite Interpreter Wrapper for YOLOv8.
    Handles model loading, inference, and dynamic output tensor parsing.
    """
    def __init__(self, model_path="yolov8_high_acc.tflite"):
        self.model_path = model_path

        # 1. Auto-Download Model if not present
        if not os.path.exists(self.model_path):
            print("🌐 Downloading TFLite model...")
            file_id = '1kkejEYyZvbMrenMJj_PUl3v5myQjccxw'
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, self.model_path, quiet=False)

        # 2. Initialize TFLite Interpreter
        print("⏳ Initializing TFLite Interpreter...")
        try:
            self.interpreter = tflite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            print(f"✅ Model ready. Input Shape: {self.input_details[0]['shape']}")
            
        except Exception as e:
            print(f"❌ Interpreter Error: {e}")
            raise e

    def predict(self, input_tensor):
        """
        Runs inference and parses the decoupled YOLO output heads.
        """
        # 1. Prepare Input (Ensure Float32)
        input_data = np.array(input_tensor, dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # 2. Run Inference
        self.interpreter.invoke()

        # 3. Parse Outputs (Dynamic Mapping)
        # TFLite output order can vary, so we identify tensors by shape and content.
        boxes = None
        scores = None
        classes = None
        
        for detail in self.output_details:
            output_data = self.interpreter.get_tensor(detail['index'])
            shape = output_data.shape
            
            # CASE A: Bounding Boxes (Shape: [1, 100, 4])
            if len(shape) == 3 and shape[2] == 4:
                boxes = output_data[0]
                
            # CASE B: Scores or Classes (Shape: [1, 100])
            elif len(shape) == 2 and shape[1] == 100:
                sample = output_data[0]
                
                # Filter non-zero elements to check for floating point values
                non_zeros = sample[sample > 0]
                
                if len(non_zeros) > 0:
                    # Check if values are integers (Classes) or floats (Scores)
                    # Modulo 1 != 0 implies floating point numbers (Probability Scores)
                    has_decimals = np.any(np.mod(non_zeros, 1) != 0)
                    
                    if has_decimals:
                        scores = output_data[0] 
                    else:
                        classes = output_data[0]
                else:
                    # If array is all zeros, assume it's scores (safe default)
                    if scores is None:
                        scores = output_data[0]
                    else:
                        classes = output_data[0]

        # 4. Handle Missing Outputs (Fallbacks)
        if boxes is None:
            return self._empty_result()
            
        if scores is None:
            scores = np.ones(len(boxes)) # Default confidence: 1.0
            
        if classes is None:
            classes = np.zeros(len(boxes)) # Default class: 0

        return self._post_process(boxes, scores, classes)

    def _post_process(self, boxes, scores, classes):
        """
        Filters detections based on confidence score (> 0.25).
        """
        mask = scores > 0.25
        
        final_boxes = boxes[mask]
        final_scores = scores[mask]
        final_classes = classes[mask]
        
        if len(final_boxes) == 0:
            return self._empty_result()

        # Format output
        output_boxes = []
        for box in final_boxes:
            output_boxes.append(box)

        return {
            'boxes': np.array([output_boxes]),
            'confidence': np.array([final_scores]),
            'classes': np.array([final_classes])
        }

    def _empty_result(self):
        """
        Returns empty numpy arrays when no objects are detected.
        """
        return {
            'boxes': np.array([[]]), 
            'confidence': np.array([[]]), 
            'classes': np.array([[]])
        }