# 🚗 YOLOv8 Autonomous Driving Object Detection System

![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi&logoColor=white)
![TensorFlow Lite](https://img.shields.io/badge/TFLite-Edge_AI-FF6F00?logo=tensorflow&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?logo=streamlit&logoColor=white)

---

![Demo](demo_traffic.gif)
*(Demo GIF has been accelerated by 2x for better visualization.)*

## 🚀 Performance & Future Improvements

Currently, this project utilizes **TensorFlow Lite (CPU)** for inference to ensure maximum portability and ease of installation without complex driver dependencies.

* **Current Status:** Running on a standard CPU, the system achieves approximately **3-5 FPS**.
* **Future Roadmap:** Plans include migrating the inference pipeline to **GPU (CUDA/TensorRT)**. This transition is expected to significantly boost performance, enabling high-FPS, real-time detection capabilities.

---

![Example](Example.png)

## 🎮 Live Demo
Try the application live:  
**[👉 Click Here to Open App](https://yolov8-autonomous-driving.onrender.com)**

> **Note:** The server runs on a Render Free Tier. Please wait **~1 minute** for the instance to wake up if it is currently inactive.

---

## 📋 Overview
This project is an **end-to-end AI microservice application** optimized for real-time object detection in autonomous driving scenarios.

The system has been re-engineered from a heavy **Keras-based architecture** to a lightweight and high-performance **TFLite (TensorFlow Lite)** inference engine. The model is served via a **FastAPI** backend and containerized with **Docker** for scalable deployment, featuring a **Streamlit** frontend for interactive testing.

---

## ⚡ Key Features
- **Optimized Inference (TFLite):** Switched from heavy Keras models to **TFLite (Float32)**, significantly reducing RAM usage and latency.
- **Microservice Architecture:** Decoupled architecture with a FastAPI backend and Streamlit frontend communicating via HTTP.
- **Orchestrated Deployment:** Uses a custom `start.sh` script to manage multi-process (API + Frontend) execution within a single Docker container.
- **Auto-Model Fetching:** Model weights are automatically downloaded from the cloud upon the first run.
- **Robust Preprocessing:** Custom Letterbox resizing and coordinate recovery logic ensures high precision mapping.

---

## 🏗 Project Structure

```text
YOLOv8-Autonomous-Driving/
│
├── app/                        # Application Core
│   ├── inference.py            # TFLite wrapper & custom inference logic
│   └── utils.py                # Preprocessing (Letterbox) & Visualization
│
├── api.py                      # FastAPI Backend (Server)
├── streamlit_app.py            # Streamlit Frontend (Client)
├── start.sh                    # Entrypoint script for orchestration
├── yolov8_high_acc.tflite      # Optimized Model (Auto-downloaded)
├── Dockerfile                  # Container configuration
├── requirements.txt            # Dependencies
└── README.md                   # Documentation
```
🧠 Model & Dataset

The model is based on the YOLOv8 architecture, fine-tuned on a Self-Driving Cars Dataset.

- Format: .tflite (TensorFlow Lite Float32)

- Input Shape: (1, 640, 640, 3)

Detected Classes

🟢 Car

🔵 Truck

🔴 Pedestrian

🟣 Cyclist

🟠 Traffic Light

Note: The model file is hosted remotely. The application uses gdown to automatically fetch it during startup.

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

🚀 How to Run

You can run this project using Docker (Recommended) or directly with Python.

Option 1: Run with Docker (Recommended) 🐳

This method prevents dependency conflicts and ensures a consistent environment.

Build the Image
```
docker build -t yolo-autonomous-app .
```

Run the Container
```
docker run -p 8501:8501 yolo-autonomous-app
```

(Note: The container runs both FastAPI (Internal 8000) and Streamlit (Exposed 8501))
Access the App 👉 http://localhost:8501

Option 2: Run Locally (Python) 🐍

Clone the Repository
```
git clone [https://github.com/alperenndemirbas/YOLOv8-Autonomous-Driving](https://github.com/alperenndemirbas/YOLOv8-Autonomous-Driving)
cd YOLOv8-Autonomous-Driving
```

Install Dependencies
```
pip install -r requirements.txt
```

Start the Backend
```
uvicorn api:app --reload
```

Wait for the log: ✅ Model ready...

Start the Frontend (in a new terminal)
```
streamlit run streamlit_app.py
```

📡 API Usage
The backend exposes a REST API for prediction.

POST /predict

- Input: Multipart/form-data (Image file: jpg, png)

- Output: JSON Object

Example Response
```
{
  "filename": "highway.jpg",
  "detections": [
    {
      "box": [450, 320, 580, 410],
      "score": 0.92,
      "class_id": 0,
      "label": "car"
    },
    {
      "box": [120, 200, 300, 450],
      "score": 0.88,
      "class_id": 1,
      "label": "truck"
    }
  ]
}
```

📊 Training Results & Observations

![Model_Result](yolo_model_result.png)

The model was planned to be trained for 15 epochs, but training was manually stopped at Epoch 9 after observing stable convergence and diminishing performance gains.

- Rapid loss reduction after initial epochs

- Significant decrease in class loss, indicating successful learning

- No strong signs of overfitting observed up to Epoch 9

⚠️ Note: The model has not reached full convergence and can be further improved with additional epochs, stronger data augmentation, and better class balance.

🔍 Technical Analysis & Future Work
📊 Performance Analysis

- Status: Prototype / Proof of Concept.

- Current Success: Excellent detection of vehicles in daylight conditions.

- Optimization Gains: Switching to TFLite reduced container size and memory usage by approximately 60%, enabling deployment on free-tier cloud   instances.

📉 Confusion Matrix (CM) Analysis

The Confusion Matrix shows high accuracy for the Car class, while Pedestrian, Cyclist, and Traffic Light exhibit relatively lower performance.

Root Cause:
Class imbalance in the dataset, where the Car class is overrepresented.

📈 Future Improvements
- Data Augmentation: Increasing samples for underrepresented classes (Pedestrians/Cyclists).

- Scenario Diversity: Adding rain, fog, and night-time datasets.

- Training: Longer training with Early Stopping.

- Model Scaling: Fine-tuning larger YOLOv8 variants (Medium/Large) for detecting small/distant objects.

⚡ Impact of TFLite Conversion
Beyond speed and memory efficiency, the TFLite conversion provided:

- More stable inference on CPU-only environments.

- Faster cold-start times on serverless platforms.

📌 Conclusion
This project presents a lightweight, scalable, and real-time object detection system for autonomous driving scenarios. Transitioning from a heavy Keras architecture to an optimized TFLite microservice demonstrates a practical and efficient approach to deploying AI on edge devices and cloud platforms.
