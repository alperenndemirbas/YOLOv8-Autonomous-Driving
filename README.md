# 🚗 YOLOv8 Autonomous Driving Object Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-Edge_AI-FF6F00?logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)

---

![Demo](demo_traffic.gif)
> *(Note: The demo GIF is accelerated 2x for better visualization. Real-time inference on CPU is ~5 FPS.)*

## 🚀 Performance & System Status

This project is engineered for **portability and ease of deployment**, utilizing **TensorFlow Lite (CPU)** for inference. This eliminates complex GPU driver dependencies, making it runnable on any standard machine.

* **Current Performance:** ~3-5 FPS on standard CPU.
* **Optimization:** Frame skipping algorithms are implemented in "Video Analysis Mode" to ensure smooth processing.
* **Future Roadmap:** Migration to **GPU (CUDA/TensorRT)** is planned to achieve fully real-time, high-FPS performance.

---

![Example](Example.png)

## 🎮 Live Demo
Try the deployed cloud application:  
**[👉 Click Here to Open App](https://yolov8-autonomous-driving.onrender.com)**

> **⚠️ Note:** The server runs on Render's Free Tier. If the app is inactive, please wait **~1 minute** for the instance to wake up.

---

## 📋 Overview
This project is an **end-to-end AI microservice application** designed for object detection in autonomous driving scenarios.

The system has been re-engineered from a heavy **Keras-based architecture** to a lightweight **TFLite (Float32)** inference engine. It is served via a **FastAPI** backend and containerized with **Docker**, featuring a **Streamlit** frontend for interactive testing.

---

## ⚡ Key Features
-   **Optimized Inference (TFLite):** Converted heavy YOLO models to TFLite, reducing RAM usage and latency by ~60%.
-   **Hybrid Processing Modes:**
    -   **Live Camera:** Real-time detection via WebRTC.
    -   **Offline Video Analysis:** CPU-optimized processing with **Frame Skipping** for smooth playback.
-   **Microservice Architecture:** Decoupled system with FastAPI (Backend) and Streamlit (Frontend) communicating via HTTP.
-   **Orchestrated Deployment:** Custom `start.sh` script manages multi-process execution within a single Docker container.
-   **Robust Preprocessing:** Custom Letterbox resizing and coordinate recovery logic ensures high-precision bounding boxes.

---

## 🏗 Project Structure

```text
YOLOv8-Autonomous-Driving/
│
├── app/                                    # Application Core (Shared Logic)
│   ├── inference.py                        # TFLite wrapper & inference engine
│   └── utils.py                            # Preprocessing (Letterbox) & Visualization
│
├── yolov8_live/                            # Live & Video Analysis Module
│   └── live_app.py                         # Hybrid App (Video File Analysis & WebRTC) -> RECOMMENDED FOR DEMO
│
├── api.py                                  # FastAPI Backend (Server)
├── streamlit_app.py                        # Streamlit Frontend (Client - API mode)
├── convert_model.py                        # Script to convert heavy .keras models to optimized .tflite format
├── YOLOv8_Autonomous_Driving_Training.ipynb # Jupyter Notebook for training & fine-tuning the YOLOv8 model
├── start.sh                                # Entrypoint script for Docker orchestration
├── yolov8_high_acc.tflite                  # Optimized Model (Auto-downloaded)
├── Dockerfile                              # Container configuration
├── requirements.txt                        # Python Dependencies
└── README.md                               # Documentation
```

---

🧠 Model & Dataset
The model is based on the YOLOv8 architecture, fine-tuned on a Self-Driving Cars Dataset.

Format: .tflite (TensorFlow Lite Float32)

Input Shape: (1, 640, 640, 3)

Detected Classes
🟢 Car

🔵 Truck

🔴 Pedestrian

🟣 Cyclist

🟠 Traffic Light

Note: The model file is hosted remotely. The application uses gdown to automatically fetch it during startup.

---

🛠️ Tech Stack
AI & Core

- TensorFlow (TFLite): High-performance inference

- NumPy: Matrix operations and post-processing

- OpenCV: Image manipulation and drawing

Backend & Deployment

- FastAPI: High-performance async web framework

- Docker: Containerization for consistent environments

- Uvicorn: ASGI server

Frontend

- Streamlit: Interactive web interface for testing and demo purposes

---

🚀 How to Run
You can run this project using Docker (Recommended for API mode) or directly with Python (Recommended for Video Analysis).

Option 1: Run Locally (Python) - Recommended for Video Analysis 🐍
To use the Video File Analysis mode (as seen in the GIF):
```
# 1. Clone the Repository
git clone https://github.com/alperenndemirbas/YOLOv8-Autonomous-Driving
cd YOLOv8-Autonomous-Driving

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Run the Hybrid App
streamlit run live_app.py
```
Select "Video Dosyası (Offline)" from the sidebar to test video analysis.

Option 2: Run with Docker (Microservice Mode) 🐳
This method runs the full API + Frontend architecture as deployed on Render.
```
# Build the Image
docker build -t yolo-autonomous-app .

# Run the Container
docker run -p 8501:8501 yolo-autonomous-app
```
(Note: The container runs both FastAPI (Internal 8000) and Streamlit (Exposed 8501)) Access the App: http://localhost:8501

---

📡 API Usage
The backend exposes a REST API for prediction.

POST /predict

- Input: Multipart/form-data (Image file: jpg, png)

- Output: JSON Object

Example Response:
```
{
  "filename": "highway.jpg",
  "detections": [
    {
      "box": [450, 320, 580, 410],
      "score": 0.92,
      "class_id": 0,
      "label": "car"
    }
  ]
}
```

---

📊 Training Results & Observations
The model was planned to be trained for 15 epochs, but training was manually stopped at Epoch 9 after observing stable convergence and diminishing performance gains.

- Rapid loss reduction after initial epochs.

- Significant decrease in class loss, indicating successful learning.

- No strong signs of overfitting observed up to Epoch 9.

⚠️ Note: The model has not reached full convergence and can be further improved with additional epochs, stronger data augmentation, and better class balance.

![Model_Result](yolo_model_result.png)

---

🔍 Technical Analysis & Future Work
📊 Performance Analysis
- Status: Prototype / Proof of Concept.

- Current Success: Excellent detection of vehicles in daylight conditions.

- Optimization Gains: Switching to TFLite reduced container size and memory usage by approximately 60%, enabling deployment on free-tier cloud instances.

📉 Confusion Matrix (CM) Analysis
The Confusion Matrix shows high accuracy for the Car class, while Pedestrian, Cyclist, and Traffic Light exhibit relatively lower performance.

- Root Cause: Class imbalance in the dataset, where the Car class is overrepresented.

![Confusion](confusion_matrix.png)

📈 Future Improvements
- Data Augmentation: Increasing samples for underrepresented classes (Pedestrians/Cyclists).

- Scenario Diversity: Adding rain, fog, and night-time datasets.

- Training: Longer training with Early Stopping.

- Model Scaling: Fine-tuning larger YOLOv8 variants (Medium/Large) for detecting small/distant objects.

⚡ Impact of TFLite Conversion
Beyond speed and memory efficiency, the TFLite conversion provided:

- More stable inference on CPU-only environments.

- Faster cold-start times on serverless platforms.

---

📌 Conclusion
This project presents a lightweight, scalable, and real-time object detection system for autonomous driving scenarios. Transitioning from a heavy Keras architecture to an optimized TFLite microservice demonstrates a practical and efficient approach to deploying AI on edge devices and cloud platforms.
