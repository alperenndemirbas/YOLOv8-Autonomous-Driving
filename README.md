# 🚗 YOLOv8 Autonomous Driving Object Detection

This project implements a real-time object detection system for autonomous driving using **KerasCV** and **YOLOv8**. It detects cars, trucks, pedestrians, cyclists, and traffic lights from video feeds.

![Inference Example](inference_example.png)
*(Sample output from the model showing detected vehicles)*

## 📋 Overview

* **Model:** YOLOv8 (Medium Backbone pre-trained on COCO).
* **Framework:** TensorFlow / KerasCV.
* **Input:** 1920x1080 Dashcam Videos.
* **Preprocessing:** Resizing with padding (letterboxing) to maintain aspect ratio.

## 📥 Pre-trained Model

Since the trained model file exceeds GitHub's size limit, it is hosted on Google Drive. You can download it to skip the training process and test the inference immediately.

👉 **[Download YOLOv8 Final Model (.keras)](https://drive.google.com/file/d/1JZ0OmNuOIK8l4xxo5KoThcNpykMzsCtq/view?usp=sharing)**

## 📂 Dataset

The model is trained on the **Self Driving Cars** dataset sourced from Kaggle.
* **Source:** [Kaggle - Self Driving Cars Dataset](https://www.kaggle.com/datasets/alincijov/self-driving-cars)
* **Classes:** `Car`, `Truck`, `Pedestrian`, `Bicyclist`, `Traffic Light`.
* **Note:** The notebook is configured to automatically download this dataset using the Kaggle API.

## 📊 Performance & Analysis

### ⚠️ Project Status: Prototype / Proof of Concept
This project is developed as a **prototype** to demonstrate the capabilities of YOLOv8 on autonomous driving tasks. Due to computational constraints, the model was trained for **10 epochs** on a subset of the dataset.

### ✅ Strengths:
* **Near Detection:** The model performs excellently on identifying vehicles and pedestrians within a close range (0-30 meters).
* **Real-time Inference:** The inference pipeline is optimized for speed, suitable for real-time video processing.

### 🚧 Current Limitations & Areas for Improvement:
* **Distant Objects:** Accuracy drops for small objects appearing far in the horizon.
* **Class Imbalance:** As seen in the confusion matrix below, the model has a bias towards the "Car" class due to dataset imbalance.
* **Environmental Conditions:** The model is currently optimized for daylight conditions. Further training is required for night driving or rainy weather performance.

![Confusion Matrix](confusion_matrix.png)

### Key Observations:
1.  **Car Bias:** The model is highly accurate at detecting cars (Majority Class) but tends to misclassify Trucks and Pedestrians as "Cars". This is due to class imbalance and limited training time.
2.  **Robustness:** Despite the short training, the pipeline successfully localizes objects and handles video inference smoothly.

## 🛠️ Tech Stack
Python: 3.10

Deep Learning: TensorFlow 2.16.1, Keras 3.3.3, KerasCV 0.9.0

UI Framework: Streamlit

Computer Vision: OpenCV (with Letterbox Resizing for aspect-ratio preservation)

## 🚀 How to Run Locally
Follow these steps to set up and run the project on your local machine.

1. Clone the Repository
```bash
git clone https://github.com/alperenndemirbas/YOLOv8-Autonomous-Driving
cd YOLOv8-Autonomous-Driving
```
2. Environment Setup (Recommended)
This project requires specific package versions. It is highly recommended to use Miniconda to create an isolated environment.
```bash
conda create -n yolo_env python=3.10 -y
conda activate yolo_env
```
3. Install Dependencies
```bash
pip install -r requirements.txt
```
5. Setup the Model
👉 **[Download YOLOv8 Final Model (.keras)](https://drive.google.com/file/d/1JZ0OmNuOIK8l4xxo5KoThcNpykMzsCtq/view?usp=sharing)**

Create a folder named models in the root directory.

Place the downloaded file inside. The path should look like this: models/yolov8_model_manuel_kayit.keras

5. Run the Application
Start the Streamlit web interface:
```bash
streamlit run app.py
```
The application will open in your default web browser (usually at http://localhost:8501).
