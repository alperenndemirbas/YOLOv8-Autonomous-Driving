from fastapi import FastAPI, File, UploadFile, HTTPException
import numpy as np
from PIL import Image
import io

# --- Local Modules ---
from app.inference import YoloModel
from app.utils import letterbox_image, recover_coordinates, CLASS_NAMES

# --- API Configuration ---
app = FastAPI(
    title="Autonomous Driving YOLOv8 API",
    description="Microservice for real-time object detection using TFLite.",
    version="2.0"
)

# --- Global Model Initialization ---
# Load the model once at startup to ensure high performance.
MODEL_PATH = "yolov8_high_acc.tflite"

try:
    model_wrapper = YoloModel(MODEL_PATH)
    print(f"✅ API Initialized. Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"❌ Critical Error: Could not load model: {e}")

# --- Endpoints ---

@app.get("/")
def root():
    """
    Health check endpoint to verify server status.
    """
    return {"status": "OK", "message": "YOLOv8 Autonomous Driving API is running 🚀"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Main prediction endpoint.
    Handles image upload, preprocessing, inference, and coordinate mapping.
    """
    
    # 1. Validate File Type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    try:
        # 2. Read and Convert Image
        contents = await file.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")

        # 3. Preprocessing (Letterbox Resize)
        # Resizes image to 640x640 while maintaining aspect ratio
        processed_image, meta = letterbox_image(image_pil, target_size=(640, 640))
        
        # Prepare Input Tensor (Add batch dimension: 1, 640, 640, 3)
        image_np = np.array(processed_image)
        input_tensor = np.expand_dims(image_np, axis=0)

        # 4. Inference
        results = model_wrapper.predict(input_tensor)

        # 5. Post-Processing
        raw_boxes = results['boxes'][0]
        raw_scores = results['confidence'][0]
        raw_classes = results['classes'][0]
        
        detections = []
        
        for i in range(len(raw_scores)):
            score = float(raw_scores[i])
            
            # Filter by Confidence Threshold
            if score > 0.25:
                raw_box = raw_boxes[i]
                
                # Recover original coordinates using metadata from preprocessing
                final_box = recover_coordinates(raw_box, meta)
                
                class_id = int(raw_classes[i])
                label = CLASS_NAMES.get(class_id, "Unknown")
                
                detections.append({
                    "box": final_box,      # Format: [x1, y1, x2, y2]
                    "score": score,
                    "class_id": class_id,
                    "label": label
                })

        # Return structured JSON response
        return {"filename": file.filename, "detections": detections}

    except Exception as e:
        return {"error": str(e)}