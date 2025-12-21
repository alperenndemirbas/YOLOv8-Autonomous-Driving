import streamlit as st
import requests
from PIL import Image
import numpy as np
import io

# --- LOCAL MODULE IMPORTS ---
# Importing visualization logic only. No model inference happens here.
from app.utils import draw_boxes

# --- API CONFIGURATION ---
# Since Streamlit and FastAPI run in the same container, we target localhost.
API_URL = "http://localhost:8000/predict"

# --- UI SETUP ---
st.set_page_config(page_title="Autonomous Driving AI", page_icon="🚗", layout="wide")
st.title("🚗 Autonomous Driving - AI Detection (Client-Side)")
st.caption("Frontend interface. All analysis is performed by the FastAPI Backend.")

# --- 1. FILE UPLOAD ---
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --- 2. ANALYSIS TRIGGER ---
    if st.button("Analyze (Send to API) 🚀"):
        
        with st.spinner("Processing image via API..."):
            try:
                # A) PREPARE IMAGE FOR API
                # Reset file pointer to the beginning
                uploaded_file.seek(0)
                
                # Package the file for the HTTP POST request
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # B) SEND REQUEST
                response = requests.post(API_URL, files=files)
                
                # C) HANDLE RESPONSE
                if response.status_code == 200:
                    result_json = response.json()
                    detections = result_json.get("detections", [])
                    
                    # D) VISUALIZATION
                    # API returns corrected coordinates. We simply draw them on the original image.
                    orig_image_np = np.array(image)
                    final_image = draw_boxes(orig_image_np, detections)
                    
                    # Display Results
                    st.success(f"Processing Complete! Detected {len(detections)} objects.")
                    st.image(final_image, caption="Inference Result", use_container_width=True)
                    
                    # Show Technical Details
                    with st.expander("View Raw JSON Response"):
                        st.json(result_json)
                        
                else:
                    st.error(f"API Error: {response.status_code}")
                    st.error(response.text)

            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Could not reach the API.")
                st.warning("Ensure the backend (api.py) is running on port 8000.")
                st.info(f"Target URL: {API_URL}")
                
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")