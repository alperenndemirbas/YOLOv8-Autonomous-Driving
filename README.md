🚗 YOLOv8 Autonomous Driving Object Detection System
This project is an end-to-end AI Microservice Application designed to detect objects in autonomous driving scenarios. It features a modern, decoupled architecture with a FastAPI backend serving a custom-trained YOLOv8 model and a Streamlit frontend for interactive user testing.

📋 Overview
Model: YOLOv8 (Medium Backbone pre-trained on COCO, fine-tuned on Self-Driving Cars dataset).

Architecture: Microservice pattern (Client-Server).

Backend: FastAPI handles image processing and model inference.

Frontend: Streamlit handles user input and visualization.

Input: Supports Images and Video feeds (1920x1080 Dashcam).

Preprocessing: Letterbox resizing with padding to maintain aspect ratio without distortion.

🏗 Project Structure
Plaintext

yolo-project/
│
├── app/                  # Core modules
│   ├── inference.py      # Model loading & prediction logic
│   └── utils.py          # Image preprocessing & visualization tools
│
├── models/               # Trained AI Models
│   └── yolov8_model_manuel_kayit.keras
│
├── api.py                # FastAPI Server (The "Kitchen")
├── streamlit_app.py      # Streamlit Client (The "Waiter")
├── requirements.txt      # Project dependencies
└── README.md             # Documentation
📥 Pre-trained Model
Since the trained model file exceeds GitHub's size limit, it is hosted on Google Drive. You must download it to run the project.

👉 Download YOLOv8 Final Model (.keras)

Place the downloaded file inside the models/ directory.

📂 Dataset
The model is trained on the Self Driving Cars dataset sourced from Kaggle.

Source: Kaggle - Self Driving Cars Dataset

Classes: Car, Truck, Pedestrian, Bicyclist, Traffic Light.

📊 Performance & Analysis
⚠️ Project Status: Prototype
This project serves as a Proof of Concept to demonstrate the integration of YOLOv8 with a modern web stack. The model was trained for 10 epochs due to computational constraints.

✅ Strengths:
Near Detection: Excellent performance on identifying vehicles and pedestrians within a close range (0-30 meters).

Production Architecture: The project uses a scalable API-based approach rather than a monolithic script.

🚧 Limitations:
Class Imbalance: As seen in the confusion matrix, the model has a bias towards the "Car" class.

Environmental Conditions: Optimized for daylight; performance may drop in night/rain conditions.

🛠️ Tech Stack
Deep Learning: TensorFlow 2.16.1, Keras 3.3.3, KerasCV 0.9.0

Backend: FastAPI, Uvicorn, Python-Multipart

Frontend: Streamlit, Requests

Image Processing: OpenCV, Pillow, NumPy

🚀 How to Run Locally
Follow these steps to set up the Microservice architecture.

1. Clone the Repository
Bash

git clone https://github.com/alperenndemirbas/YOLOv8-Autonomous-Driving
cd YOLOv8-Autonomous-Driving
2. Environment Setup (Recommended)
Bash

conda create -n yolo_env python=3.10 -y
conda activate yolo_env
3. Install Dependencies
Bash

pip install -r requirements.txt
4. Setup the Model
Download the model from the link above and ensure your folder looks like this: models/yolov8_model_manuel_kayit.keras

5. Run the Backend (API)
Open a terminal and start the API server. This loads the model into memory.

Bash

uvicorn api:app --reload
The server will start at http://127.0.0.1:8000

6. Run the Frontend (UI)
Open a second terminal and start the Streamlit interface.

Bash

streamlit run streamlit_app.py
The application will open in your default browser at http://localhost:8501

👤 Author: Alperen Demirbaş