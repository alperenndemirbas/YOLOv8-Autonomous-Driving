import streamlit as st
from PIL import Image
import numpy as np
from app.utils import draw_boxes  # Local utility for visualization
from app.inference import YoloModel # Direct import for standalone mode (No API)

# --- CONFIGURATION ---
# We are switching to Standalone Mode to save RAM on Render Free Tier.
# The model will be loaded directly into Streamlit's memory, bypassing the API overhead.

# --- UI SETUP ---
st.set_page_config(page_title="Autonomous Driving AI", page_icon="🚗", layout="wide")
st.title("🚗 Autonomous Driving - AI Detection (Standalone)")

# --- MODEL LOADING (CACHE) ---
# Use st.cache_resource to load the model only once and keep it in memory.
# This prevents reloading the heavy model (300MB+) on every user interaction,
# which is crucial for preventing "Out of Memory" crashes on Render.
@st.cache_resource
def load_model():
    model_path = "models/yolov8_quantized.tflite"
    return YoloModel(model_path)

# Show a spinner during the initial load (downloading + loading into RAM)
with st.spinner("Loading AI Model... (This may take 1-2 minutes on first run)"):
    try:
        model = load_model()
    except Exception as e:
        st.error(f"Critical Error: Could not load model. {e}")
        st.stop()

# File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Original Image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2. Button to Trigger Prediction
    if st.button("Analyze Image 🚀"):
        
        # Show spinner while processing
        with st.spinner("Running AI Inference..."):
            try:
                # --- PREPROCESSING ---
                # Resize image to 640x640 as expected by YOLOv8 architecture
                input_image = image.resize((640, 640))
                image_np = np.array(input_image)

                # Normalize to 0-1 range (Standard for Neural Networks)
                input_tensor = image_np.astype(np.float32) / 255.0

                # Add batch dimension: (640, 640, 3) -> (1, 640, 640, 3)
                input_tensor = np.expand_dims(input_tensor, axis=0)

                # --- INFERENCE ---
                # Call the model directly (no API request needed)
                # The 'predict' method returns the dictionary from KerasCV NMS layer
                results = model.predict(input_tensor)

                # --- POST-PROCESSING ---
                # Extract boxes, classes, and confidence scores from the dictionary
                # Expected keys from KerasCV NMS: 'boxes', 'classes', 'confidence'
                boxes = results['boxes'][0]
                classes = results['classes'][0]
                confidence = results['confidence'][0]

                # Filter out invalid detections (padding usually has -1 or 0 confidence)
                detections = []
                for i in range(len(confidence)):
                    if confidence[i] > 0:
                        detections.append({
                            "box": boxes[i], 
                            "class_id": int(classes[i]),
                            "score": float(confidence[i])
                        })

                # --- VISUALIZATION ---
                # Draw bounding boxes on the resized image (640x640)
                final_image = draw_boxes(image_np, detections)
                
                # Display the processed image
                st.image(final_image, caption=f"AI Detection Result ({len(detections)} Objects)", use_container_width=True)
                
                # Debug info
                with st.expander("View Technical Details"):
                    st.write(f"Detected {len(detections)} objects.")
                    st.write(detections)

            except Exception as e:
                st.error(f"An unexpected error occurred during inference: {e}")