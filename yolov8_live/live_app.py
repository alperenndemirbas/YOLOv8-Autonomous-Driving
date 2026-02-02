import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import cv2
import numpy as np
import av
import os
import time
import tempfile
import torch
from ultralytics import YOLO

# --- CONFIGURATION ---
MODEL_PATH = "best.pt"

CLASS_NAMES = {
    0: 'car', 
    1: 'truck', 
    2: 'pedestrian', 
    3: 'bicyclist', 
    4: 'light'
}

# --- GPU MODEL WRAPPER ---
class LocalGPUModel:
    def __init__(self, model_path):
        # Check for CUDA device
        if torch.cuda.is_available():
            self.device = 'cuda'
            self.device_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU ACTIVATED: {self.device_name}")
        else:
            self.device = 'cpu'
            self.device_name = "CPU"
            print("⚠️ GPU not found, falling back to CPU...")
            
        # Load YOLOv8 model
        self.model = YOLO(model_path)

    def predict(self, image, conf_threshold=0.25):
        """
        Runs inference on the GPU using Ultralytics.
        Handles resizing and normalization automatically.
        """
        results = self.model(image, device=self.device, conf=conf_threshold, verbose=False)
        result = results[0]

        # Convert GPU tensors to CPU numpy arrays
        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)

        return {
            'boxes': boxes,
            'confidence': scores,
            'classes': class_ids
        }

# --- DRAWING UTILS ---
def draw_detections(image, detections, fps=None):
    """Draws bounding boxes, labels, and FPS on the image."""
    
    # 1. Draw Boxes & Labels
    for i in range(len(detections['boxes'])):
        box = detections['boxes'][i]
        score = detections['confidence'][i]
        cls_id = detections['classes'][i]
        
        x1, y1, x2, y2 = map(int, box)
        label_text = f"{CLASS_NAMES.get(cls_id, 'Unknown')} {score:.2f}"
        
        # Draw Rectangle (Green)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw Label Background
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Draw Label Text
        cv2.putText(image, label_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 2. Draw FPS
    if fps is not None:
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(image, fps_text, (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
    return image

# --- STREAMLIT PAGE CONFIG ---
st.set_page_config(page_title="Traffic Analysis (GPU)", page_icon="🏎️", layout="wide")
st.title("🏎️ Autonomous Driving")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return LocalGPUModel(MODEL_PATH)

try:
    yolo_model = load_model()
    st.success(f"Model Loaded! Running on: **{yolo_model.device_name}**")
except Exception as e:
    st.error(f"Failed to load model. Ensure 'best.pt' is in the directory. Error: {e}")
    st.stop()

# --- SIDEBAR SETTINGS ---
st.sidebar.title("Settings")
app_mode = st.sidebar.selectbox("Mode Selection", ["Video File (Offline)", "Live Camera (Live)"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.25, 0.90, 0.40)

# ==========================================
# MODE A: VIDEO FILE ANALYSIS
# ==========================================
if app_mode == "Video File (Offline)":
    st.header("📁 Video Analysis (High Performance)")
    uploaded_file = st.file_uploader("Upload Video (.mp4, .avi)", type=['mp4', 'avi', 'mov'])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        stop_btn = st.button("Stop")
        
        prev_time = 0
        
        while cap.isOpened() and not stop_btn:
            ret, frame = cap.read()
            if not ret:
                break
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time > 0 else 0
            prev_time = curr_time

            # 1. Inference
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = yolo_model.predict(img_rgb, conf_threshold=confidence_threshold)
            
            # 2. Draw (Boxes + FPS)
            frame_drawn = draw_detections(img_rgb.copy(), results, fps=fps)
            
            # 3. Display
            st_frame.image(frame_drawn, channels="RGB", width="stretch")
        
        cap.release()
        st.write("Video processing completed.")

# ==========================================
# MODE B: LIVE CAMERA (WebRTC)
# ==========================================
elif app_mode == "Live Camera (Live)":
    st.header("📹 Live Camera Feed (GPU Accelerated)")

    class YoloVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = yolo_model
            self.conf_thresh = confidence_threshold
            self.prev_time = 0

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            
            # FPS Calculation
            curr_time = time.time()
            fps = 1 / (curr_time - self.prev_time) if self.prev_time > 0 else 0
            self.prev_time = curr_time

            # 1. Inference (Convert BGR to RGB for model)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.model.predict(img_rgb, conf_threshold=self.conf_thresh)
            
            # 2. Draw (Draw on BGR image for WebRTC return)
            img_drawn = draw_detections(img, results, fps=fps)

            return av.VideoFrame.from_ndarray(img_drawn, format="bgr24")

    # WebRTC Config
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    webrtc_streamer(
        key="traffic-yolo-live",
        video_processor_factory=YoloVideoProcessor,
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )