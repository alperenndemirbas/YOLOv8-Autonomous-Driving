# 1. Base Image: Use lightweight Python 3.10
FROM python:3.10-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. System Dependencies
# Install required system libraries for OpenCV (libgl1, libglib2.0)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements first to leverage Docker cache
COPY requirements.txt .

# 5. Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the rest of the application code
COPY . .

# 7. Expose ports for API (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# 8. Start Command
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]