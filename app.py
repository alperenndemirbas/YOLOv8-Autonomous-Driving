import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import keras_cv
import tempfile
import os
import gdown

# --- CONFIGURATION ---
# Google Drive File ID for the pre-trained model
MODEL_DRIVE_ID = '1JZ0OmNuOIK8l4xxo5KoThcNpykMzsCtq'
MODEL_FILENAME = 'yolov8_model_manuel_kayit.keras'

# Visualization Settings
CLASS_MAPPING = {0: 'Car', 1: 'Truck', 2: 'Pedestrian', 3: 'Cyclist', 4: 'Traffic Light'}
COLORS = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
IMAGE_SIZE = (640, 640)

# --- MODEL LOADING ---
@st.cache_resource
def load_model():
    # Download model from Drive if it doesn't exist locally
    if not os.path.exists(MODEL_FILENAME):
        url = f'https://drive.google.com/uc?id={MODEL_DRIVE_ID}'
        st.info("Downloading model from Drive... This might take a minute.")
        gdown.download(url, MODEL_FILENAME, quiet=False)
    
    # Load the Keras model with custom objects
    model = tf.keras.models.load_model(
        MODEL_FILENAME,
        compile=False,
        custom_objects={
            "YOLOV8Detector": keras_cv.models.YOLOV8Detector,
            "YOLOV8Backbone": keras_cv.models.YOLOV8Backbone
        }
    )
    
    # Configure NonMaxSuppression decoder for inference
    model.prediction_decoder = keras_cv.layers.NonMaxSuppression(
        bounding_box_format="xyxy",
        from_logits=False, # Ensure probabilities are not re-scaled
        iou_threshold=0.5, 
        confidence_threshold=0.2,
    )
    
    return model

# --- INFERENCE UTILS ---
def predict_frame(model, frame):
    # Preprocess frame: Resize, Pad, and Convert to Tensor
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = tf.image.resize_with_pad(input_frame, IMAGE_SIZE[0], IMAGE_SIZE[1])
    input_frame = tf.cast(input_frame, tf.float32)
    input_frame = tf.expand_dims(input_frame, axis=0)

    predictions = model.predict(input_frame, verbose=0)
    return predictions

def draw_boxes(frame, predictions, width, height):
    # Extract predictions
    boxes = predictions['boxes'][0]
    classes = predictions['classes'][0]
    confidence = predictions['confidence'][0]

    # Calculate scaling factors to map back to original resolution
    ratio_x = IMAGE_SIZE[0] / width
    ratio_y = IMAGE_SIZE[1] / height
    scale = min(ratio_x, ratio_y)
    offset_x = (IMAGE_SIZE[0] - width * scale) / 2
    offset_y = (IMAGE_SIZE[1] - height * scale) / 2

    valid_indices = np.where(classes != -1)[0]
    
    for i in valid_indices:
        box = boxes[i]
        score = float(confidence[i])
        
        # Rescale coordinates
        x1 = int((box[0] - offset_x) / scale)
        y1 = int((box[1] - offset_y) / scale)
        x2 = int((box[2] - offset_x) / scale)
        y2 = int((box[3] - offset_y) / scale)

        class_id = int(classes[i])
        label = f"{CLASS_MAPPING.get(class_id, 'Unknown')} {score:.2f}"
        color = COLORS[class_id % len(COLORS)]

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    return frame

# --- STREAMLIT APP ---
st.title("🚗 Autonomous Driving Object Detection")
st.write("Upload a video or image to detect cars, trucks, and pedestrians.")

# Load Model
try:
    model = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File Uploader
uploaded_file = st.file_uploader("Choose a file...", type=["mp4", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Handle Image Upload
    if uploaded_file.name.endswith(('.jpg', '.jpeg', '.png')):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)
        
        preds = predict_frame(model, frame)
        h, w, _ = frame.shape
        result_frame = draw_boxes(frame, preds, w, h)
        
        st.image(result_frame, channels="BGR", caption="Processed Image")

    # Handle Video Upload
    elif uploaded_file.name.endswith('.mp4'):
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        vf = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_count = 0
        
        while vf.isOpened():
            ret, frame = vf.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Frame Skipping for Performance (Process every 5th frame)
            if frame_count % 5 != 0:
                continue
            
            preds = predict_frame(model, frame)
            h, w, _ = frame.shape
            result_frame = draw_boxes(frame, preds, w, h)
            
            stframe.image(result_frame, channels="BGR")
            
        vf.release()
