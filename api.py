from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
import cv2
from PIL import Image
import io

# --- LOCAL MODULE IMPORTS ---
from app.inference import YoloModel
from app.utils import preprocess_image, CLASS_MAPPING

# 1. INITIALIZE API APPLICATION
app = FastAPI(
    title="YOLOv8 Object Detection API",
    description="Upload an image, and YOLO will return bounding box coordinates in JSON format.",
    version="1.0"
)

# 2. LOAD MODEL GLOBALLY
# The model is loaded once when the server starts to avoid reloading it for every request.
MODEL_PATH = 'models/yolov8_model_manuel_kayit.keras'

try:
    model_wrapper = YoloModel(MODEL_PATH)
    print(f"Server started. Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model! {e}")
    # In a real production environment, we might want to shut down the server here.

# --- ENDPOINTS ---

@app.get("/")   
def root():
    """
    Health Check Endpoint.
    """
    return {"message": "YOLO API is running! Send a POST request to /predict to detect objects."}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    Receives an image file, processes it through the YOLO model,
    and returns detected objects with their coordinates and confidence scores.
    """
    
    # 1. Validate Input File Type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # 2. Read and Convert Image (Bytes -> PIL -> Numpy)
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_np = np.array(image_pil)
        
        # Convert RGB (PIL) to BGR (OpenCV format)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 3. Preprocessing (Letterbox Resize)
        # Using the helper function from app.utils
        processed_image, meta = preprocess_image(image_bgr)
        
        # 4. Prepare Input Tensor
        # Add batch dimension: (640, 640, 3) -> (1, 640, 640, 3)
        input_tensor = np.expand_dims(processed_image, axis=0)

        # 5. Run Inference
        preds = model_wrapper.predict(input_tensor)

        # 6. Post-Processing (Format results to JSON)
        # Extract raw data from predictions
        boxes = preds['boxes'][0]
        classes = preds['classes'][0]
        confidence = preds['confidence'][0]
        
        # Get scaling factors to map back to original image size
        scale = meta['scale']
        pad_left = meta['pad_left']
        pad_top = meta['pad_top']
        
        detections = []
        # Filter out invalid detections (where class is -1)
        valid_indices = np.where(classes != -1)[0]

        for i in valid_indices:
            # Calculate original coordinates (Reverse Letterbox)
            x1 = int((boxes[i][0] - pad_left) / scale)
            y1 = int((boxes[i][1] - pad_top) / scale)
            x2 = int((boxes[i][2] - pad_left) / scale)
            y2 = int((boxes[i][3] - pad_top) / scale)
            
            class_id = int(classes[i])
            score = float(confidence[i])
            label = CLASS_MAPPING.get(class_id, "Unknown")

            # Append to result list
            detections.append({
                "class": label,
                "confidence": round(score, 2),
                "box": [x1, y1, x2, y2]  # Format: [x_min, y_min, x_max, y_max]
            })

        # 7. Return JSON Response
        return {"filename": file.filename, "detections": detections}

    except Exception as e:
        # Handle unexpected server errors
        return {"error": str(e)}