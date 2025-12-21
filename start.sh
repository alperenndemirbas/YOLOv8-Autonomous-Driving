#!/bin/bash

# Orchestration script to run both FastAPI (Backend) and Streamlit (Frontend)
# within a single Docker container.

# 1. Start FastAPI Backend in the background
# It runs internally on port 8000.
echo "🚀 Starting FastAPI Backend..."
uvicorn api:app --host 0.0.0.0 --port 8000 &

# 2. Wait for API to initialize
# Gives the backend time to load the model before the frontend starts.
sleep 5

# 3. Start Streamlit Frontend
# Binds to the PORT variable provided by the cloud provider (Render)
# or defaults to 8501 if running locally.
echo "🎨 Starting Streamlit Frontend..."
streamlit run streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0