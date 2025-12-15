import streamlit as st
import cv2
import numpy as np
import tempfile

# --- LOCAL MODULE IMPORTS ---
# Importing logic from our clean architecture structure
from app.inference import YoloModel
from app.utils import preprocess_image, draw_boxes

# --- CONFIGURATION ---
MODEL_PATH = 'models/yolov8_model_manuel_kayit.keras'

# --- MODEL INITIALIZATION ---
@st.cache_resource
def get_model():
    """
    Initializes and caches the YoloModel instance.
    This prevents reloading the model on every UI interaction.
    """
    try:
        return YoloModel(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# --- STREAMLIT UI SETUP ---
st.set_page_config(page_title="Autonomous Driving Detection", page_icon="🚗")
st.title("🚗 Autonomous Driving Object Detection")
st.caption("Powered by KerasCV & YOLOv8 | Modular Architecture")

# Load model once
model_wrapper = get_model()

if model_wrapper:
    st.success("System Ready")

    # File Uploader
    uploaded_file = st.file_uploader("Upload a Video or Image", type=["mp4", "jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # --- IMAGE PROCESSING PIPELINE ---
        if uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
            # Read image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, 1)
            
            # 1. Preprocess (Letterbox Resize)
            processed_frame, meta = preprocess_image(frame)
            
            # 2. Prepare Input Tensor (Add batch dimension)
            input_tensor = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(input_tensor, axis=0)
            
            # 3. Inference
            preds = model_wrapper.predict(input_tensor)
            
            # 4. Visualization (Draw Boxes)
            result_frame = draw_boxes(frame, preds, meta)
            
            # Display
            st.image(result_frame, channels="BGR", use_container_width=True, caption="Detected Objects")

        # --- VIDEO PROCESSING PIPELINE ---
        elif uploaded_file.name.endswith('.mp4'):
            # Save temp file for OpenCV to read
            tfile = tempfile.NamedTemporaryFile(delete=False) 
            tfile.write(uploaded_file.read())
            
            vf = cv2.VideoCapture(tfile.name)
            stframe = st.empty()
            
            while vf.isOpened():
                ret, frame = vf.read()
                if not ret: break
                
                # Pipeline: Preprocess -> Predict -> Draw
                processed_frame, meta = preprocess_image(frame)
                
                input_tensor = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                input_tensor = np.expand_dims(input_tensor, axis=0)
                
                preds = model_wrapper.predict(input_tensor)
                result_frame = draw_boxes(frame, preds, meta)
                
                # Update video frame
                stframe.image(result_frame, channels="BGR", use_container_width=True)
                
            vf.release()