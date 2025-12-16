# 🚗 YOLOv8 Autonomous Driving Object Detection System

This project is an **end-to-end AI microservice application** designed for object detection in autonomous driving scenarios.  
It follows a **modern, decoupled client–server architecture**, featuring a **FastAPI backend** that serves a custom-trained **YOLOv8** model and a **Streamlit frontend** for interactive testing and visualization.

---

## 📋 Overview

- **Model:** YOLOv8 (Medium backbone, pre-trained on COCO and fine-tuned on a Self-Driving Cars dataset)
- **Architecture:** Microservice-based (Client–Server)
- **Backend:** FastAPI for image processing and model inference
- **Frontend:** Streamlit for user interaction and visualization
- **Input:** Images and dashcam videos (up to 1920×1080)
- **Preprocessing:** Letterbox resizing with padding to preserve aspect ratio without distortion

---

## 🏗 Project Structure

yolo-project/
│
├── app/ # Core modules
│ ├── inference.py # Model loading and prediction logic
│ └── utils.py # Image preprocessing and visualization utilities
│
├── models/ # Trained AI models
│ └── yolov8_model_manuel_kayit.keras
│
├── api.py # FastAPI server (Backend)
├── streamlit_app.py # Streamlit client (Frontend)
├── requirements.txt # Project dependencies
└── README.md # Documentation

---

## 📥 Pre-trained Model

Due to GitHub file size limitations, the trained model is hosted on **Google Drive**.

👉 **Download YOLOv8 Final Model (.keras)**  
*(https://drive.google.com/file/d/1JZ0OmNuOIK8l4xxo5KoThcNpykMzsCtq/view?usp=drive_link)*

After downloading, place the file inside the following directory:

models/yolov8_model_manuel_kayit.keras

---

## 📂 Dataset

The model was trained on the **Self Driving Cars Dataset** sourced from Kaggle.

- **Source:** Kaggle – Self Driving Cars Dataset  
- **Classes:**  
  - Car  
  - Truck  
  - Pedestrian  
  - Bicyclist  
  - Traffic Light  

---

## 📊 Performance & Analysis

⚠️ **Project Status:** Prototype / Proof of Concept

This project demonstrates the integration of **YOLOv8** into a modern web-based microservice architecture.  
The model was trained for **10 epochs** due to computational limitations.

### ✅ Strengths
- **Near-range detection:** Strong performance for objects within approximately 0–30 meters
- **Production-ready architecture:** API-based design suitable for scaling and extension

### 🚧 Limitations
- **Class imbalance:** The model shows bias toward the *Car* class, as observed in the confusion matrix
- **Environmental sensitivity:** Optimized for daylight conditions; performance may degrade at night or in rainy environments

---

## 🛠️ Tech Stack

### Deep Learning
- TensorFlow 2.16.1  
- Keras 3.3.3  
- KerasCV 0.9.0  

### Backend
- FastAPI  
- Uvicorn  
- Python-Multipart  

### Frontend
- Streamlit  
- Requests  

### Image Processing
- OpenCV  
- Pillow  
- NumPy  

---

## 🚀 How to Run Locally

Follow the steps below to run the microservice architecture on your local machine.

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/alperenndemirbas/YOLOv8-Autonomous-Driving
cd YOLOv8-Autonomous-Driving
```
2️⃣ Environment Setup (Recommended)
```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
```

3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

4️⃣ Set Up the Model

Download the pre-trained model and ensure the directory structure is as follows:
```bash
models/yolov8_model_manuel_kayit.keras
```
5️⃣ Run the Backend (API)

Start the FastAPI server (this loads the model into memory):
```bash
uvicorn api:app --reload
```

The API will be available at:
👉 http://127.0.0.1:8000

6️⃣ Run the Frontend (UI)

In a second terminal, start the Streamlit application:
```bash
streamlit run streamlit_app.py
```

The UI will open automatically at:
👉 http://localhost:8501
