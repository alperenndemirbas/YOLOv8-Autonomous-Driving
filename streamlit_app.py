import streamlit as st
import requests
from PIL import Image
import io
import numpy as np
from app.utils import draw_boxes  # Local utility for visualization

# --- CONFIGURATION ---
# The endpoint of the local FastAPI server
# In Docker/Render, localhost usually works, but 127.0.0.1 is more explicit.
API_URL = "http://127.0.0.1:8000/predict"

# --- UI SETUP ---
st.set_page_config(page_title="Autonomous Driving Client", page_icon="🚗", layout="wide")
st.title("🚗 Autonomous Driving - API Client")

# File Uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 1. Display Original Image
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Updated to fix warning: used generic width parameter for compatibility
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # 2. Button to Trigger Prediction
    if st.button("Analyze Image (Send to API) 🚀"):
        
        # --- PREPARE PAYLOAD ---
        # Format the file for the multipart/form-data request
        files = {
            "file": (
                uploaded_file.name,    # Filename
                image_bytes,           # File content (bytes)
                uploaded_file.type     # Content type (e.g., image/jpeg)
            )
        }

        # Show spinner while waiting for response
        with st.spinner("Connecting to API... (Note: First run takes longer due to model download!)"):
            try:
                # --- SEND REQUEST ---
                # Send the POST request to the FastAPI backend
                # IMPORTANT: Added timeout=120 seconds because Render free tier is slow
                # and model loading takes time.
                response = requests.post(API_URL, files=files, timeout=120)
                
                # --- HANDLE RESPONSE ---
                if response.status_code == 200:
                    # Success: Parse JSON response
                    result = response.json()
                    
                    st.success(f"Success! Detected {len(result['detections'])} objects.")
                    
                    # --- VISUALIZATION ---
                    # Convert PIL image to NumPy array for drawing
                    image_np = np.array(image)
                    
                    # Draw bounding boxes using the received coordinates
                    # Note: utils.draw_boxes now handles the updated JSON format
                    final_image = draw_boxes(image_np, result['detections'])
                    
                    # Display the processed image
                    st.image(final_image, caption="AI Detection Result", use_container_width=True)
                    
                    # Show raw JSON data for debugging/inspection
                    with st.expander("View Technical Details (JSON)"):
                        st.json(result)
                else:
                    # Handle API errors (e.g., 400, 422, 500)
                    st.error(f"API Error! Status Code: {response.status_code}")
                    st.error(response.text)

            except requests.exceptions.ConnectionError:
                # Handle connection failures (e.g., server not running)
                st.error("🚨 Connection Error: Could not reach the API.")
                st.warning("⚠️ Troubleshooting for Render:")
                st.info(
                    "1. The server might still be downloading the model (300MB+). Wait 1-2 minutes and try again.\n"
                    "2. Refresh the page."
                )
            except requests.exceptions.Timeout:
                st.error("⏰ Request Timed Out.")
                st.info("The server took too long to respond. The model might be loading into RAM. Please try again.")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")