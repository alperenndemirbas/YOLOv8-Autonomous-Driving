import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras
import keras_cv
import tempfile
import os

# --- CONFIGURATION ---
MODEL_PATH = 'models/yolov8_model_manuel_kayit.keras'
TARGET_SIZE = (640, 640)  # Model input size
# Class mapping for the autonomous driving dataset
CLASS_MAPPING = {0: 'Car', 1: 'Truck', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Traffic Light'}
# Bounding box colors for visualization
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    """
    Loads the trained YOLOv8 model with Keras 3 compatibility.
    Configures the NonMaxSuppression decoder for inference.
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found: {MODEL_PATH}")
        st.stop()
    
    try:
        # Load model without compiling (optimizer state not needed for inference)
        model = keras.models.load_model(MODEL_PATH, compile=False)
        
        # Configure the prediction decoder
        model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
            bounding_box_format="xyxy",
            from_logits=False,
            iou_threshold=0.5,
            confidence_threshold=0.25,
        )
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# --- PREPROCESSING (LETTERBOX RESIZING) ---
def preprocess_image(image, target_size):
    """
    Resizes the image to target_size while maintaining aspect ratio (Letterbox).
    Adds padding (gray borders) to fit the square shape without distortion.
    
    Returns:
        padded_image: The processed image ready for the model.
        meta: Dictionary containing scaling factor and padding values for post-processing.
    """
    h, w = image.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    
    # Calculate new dimensions preserving aspect ratio
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h))
    
    # Calculate padding needed to reach target_size
    delta_w = target_size[0] - new_w
    delta_h = target_size[1] - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Add border (padding) with YOLO standard gray color
    padded_image = cv2.copyMakeBorder(
        resized_image, top, bottom, left, right, 
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    
    # Store metadata to map boxes back to original coordinates later
    meta = {
        'scale': scale,
        'pad_top': top,
        'pad_left': left,
        'original_dim': (h, w)
    }
    
    return padded_image, meta

# --- INFERENCE ---
def predict_frame(model, frame):
    """
    Runs prediction on a single frame.
    """
    # 1. Letterbox Preprocessing
    processed_frame, meta = preprocess_image(frame, TARGET_SIZE)
    
    # 2. Convert to RGB and add batch dimension
    input_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    input_frame = np.expand_dims(input_frame, axis=0) # Shape: (1, 640, 640, 3)
    
    # 3. Predict
    predictions = model.predict(input_frame, verbose=0)
    return predictions, meta

# --- VISUALIZATION ---
def draw_boxes(frame, predictions, meta):
    """
    Draws bounding boxes and labels on the original frame.
    Uses 'meta' to reverse the resizing/padding logic.
    """
    boxes = predictions['boxes'][0]
    classes = predictions['classes'][0]
    confidence = predictions['confidence'][0]
    
    # Retrieve preprocessing metadata
    scale = meta['scale']
    pad_left = meta['pad_left']
    pad_top = meta['pad_top']

    valid_indices = np.where(classes != -1)[0]
    
    for i in valid_indices:
        box = boxes[i]
        score = float(confidence[i])
        
        # --- COORDINATE MAPPING ---
        # 1. Subtract padding
        # 2. Divide by scale to restore original resolution
        x1 = int((box[0] - pad_left) / scale)
        y1 = int((box[1] - pad_top) / scale)
        x2 = int((box[2] - pad_left) / scale)
        y2 = int((box[3] - pad_top) / scale)

        # Get class info
        class_id = int(classes[i])
        label_text = CLASS_MAPPING.get(class_id, 'Unknown')
        label = f"{label_text} {score:.2f}"
        color = COLORS[class_id % len(COLORS)]

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background and text
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

# --- MAIN STREAMLIT APP ---
st.set_page_config(page_title="Autonomous Driving Detection", page_icon="🚗")
st.title("🚗 Autonomous Driving Object Detection")
st.caption("Model Loading")

# Load the model
model = load_model()
if model:
    st.success("Model Loaded Successfully!")

# File Uploader
uploaded_file = st.file_uploader("Upload a Video or Image", type=["mp4", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # --- IMAGE PROCESSING ---
    if uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        # Run inference
        preds, meta = predict_frame(model, frame)
        
        # Draw results
        result_frame = draw_boxes(frame, preds, meta)
        
        # Display result
        st.image(result_frame, channels="BGR", caption="Processed Image", use_container_width=True)

    # --- VIDEO PROCESSING ---
    elif uploaded_file.name.endswith('.mp4'):
        # Save uploaded video to a temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            # Run inference
            preds, meta = predict_frame(model, frame)
            
            # Draw results
            result_frame = draw_boxes(frame, preds, meta)
            
            # Display video frame
            stframe.image(result_frame, channels="BGR", use_container_width=True)
            
        vf.release()