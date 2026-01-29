import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
from PIL import Image
import av
import sys
import os
import time
import tempfile

# --- Path Configuration ---
# Add 'app' directory to system path to import local modules
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(root_dir, 'app'))

try:
    from inference import YoloModel
    from utils import letterbox_image, recover_coordinates, draw_boxes, CLASS_NAMES
except ImportError as e:
    st.error(f"Module Error: {e}. Please check the directory structure.")
    st.stop()

# --- Page & Model Setup ---
st.set_page_config(page_title="Traffic Analysis", page_icon="🚗", layout="wide")
st.title("🚗 Autonomous Driving - Vehicle Detection")

@st.cache_resource
def load_model():
    """Loads the TFLite model once and caches it."""
    return YoloModel(model_path="yolov8_high_acc.tflite")

try:
    yolo_model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Sidebar Settings ---
st.sidebar.title("Settings")
app_mode = st.sidebar.selectbox("Mode Selection", ["Video File (Offline)", "Live Camera (Live)"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 0.90, 0.40)


# ==========================================
# MODE A: VIDEO FILE ANALYSIS (Offline)
# Includes frame skipping for performance
# ==========================================
if app_mode == "Video File (Offline)":
    st.header("📁 Video Analysis")
    st.info("Processing video with frame skipping enabled (1 inference per 3 frames).")

    uploaded_file = st.file_uploader("Upload Video (.mp4, .avi)", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        # Save uploaded file to a temporary file for OpenCV
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        stop_btn = st.button("Stop")
        
        # Optimization variables
        frame_count = 0
        skip_rate = 3  # Inference every 3 frames
        detections = [] 
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                st.write("Video processing completed.")
                break
            
            frame_count += 1
            
            # --- Inference Logic (Frame Skipping) ---
            if frame_count % skip_rate == 0:
                try:
                    # Preprocessing
                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    resized_img, meta = letterbox_image(pil_image, (640, 640))
                    
                    input_tensor = np.array(resized_img, dtype=np.float32)
                    input_tensor = np.expand_dims(input_tensor, axis=0)
                    
                    # Run Inference
                    results = yolo_model.predict(input_tensor)
                    
                    # Parse Results
                    raw_boxes = results.get('boxes', [])[0]
                    raw_scores = results.get('confidence', [])[0]
                    raw_classes = results.get('classes', [])[0]
                    
                    detections = [] # Reset detections
                    if len(raw_scores) > 0:
                        for i in range(len(raw_scores)):
                            score = float(raw_scores[i])
                            if score > confidence_threshold:
                                raw_box = raw_boxes[i]
                                final_box = recover_coordinates(raw_box, meta)
                                class_id = int(raw_classes[i])
                                label = CLASS_NAMES.get(class_id, "Unknown")
                                detections.append({
                                    "box": final_box, "score": score, "class_id": class_id, "label": label
                                })
                except Exception as e:
                    print(f"Error during inference: {e}")

            # --- Drawing ---
            # Draw detections (either fresh or cached) on the current frame
            frame = draw_boxes(frame, detections)
            
            # Display
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame, channels="RGB", width="stretch")
        
        cap.release()


# ==========================================
# MODE B: LIVE CAMERA (WebRTC)
# Optimized for Real-Time CPU Processing
# ==========================================
elif app_mode == "Live Camera (Live)":
    st.header("📹 Live Camera Feed")
    st.warning("Running on CPU. FPS might be limited.")

    class YoloVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.prev_time = 0
            self.fps = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            current_time = time.time()
            img_bgr = frame.to_ndarray(format="bgr24")
            
            try:
                # 1. Preprocessing
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(img_rgb)
                resized_img, meta = letterbox_image(pil_image, (640, 640))

                input_tensor = np.array(resized_img, dtype=np.float32)
                input_tensor = np.expand_dims(input_tensor, axis=0)

                # 2. Inference
                results = yolo_model.predict(input_tensor)
                
                # 3. Post-processing
                raw_boxes = results.get('boxes', [])[0]
                raw_scores = results.get('confidence', [])[0]
                raw_classes = results.get('classes', [])[0]
                
                detections = []
                if len(raw_scores) > 0:
                    for i in range(len(raw_scores)):
                        score = float(raw_scores[i])
                        if score > confidence_threshold:
                            raw_box = raw_boxes[i]
                            final_box = recover_coordinates(raw_box, meta)
                            class_id = int(raw_classes[i])
                            label = CLASS_NAMES.get(class_id, "Unknown")
                            detections.append({
                                "box": final_box, "score": score, "class_id": class_id, "label": label
                            })

                # 4. Draw Boxes
                img_bgr = draw_boxes(img_bgr, detections)
            
            except Exception as e:
                print(f"Inference Error: {e}")
                pass

            # 5. FPS Calculation
            if current_time - self.prev_time > 0:
                self.fps = 1.0 / (current_time - self.prev_time)
            self.prev_time = current_time

            cv2.putText(img_bgr, f"FPS: {int(self.fps)}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            return av.VideoFrame.from_ndarray(img_bgr, format="bgr24")

    # --- WebRTC Configuration ---
    col1, col2 = st.columns([3, 1])

    with col1:
        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        webrtc_streamer(
            key="traffic-yolo-live",
            video_processor_factory=YoloVideoProcessor,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            media_stream_constraints={
                "video": {"width": 640, "height": 480}, # Low resolution for CPU performance
                "audio": False
            },
            async_processing=True
        )

    with col2:
        st.info("System Status")
        st.write("Active Mode: Live Stream")
        st.write("Tip: Switch to **Video File** mode for smoother offline analysis.")