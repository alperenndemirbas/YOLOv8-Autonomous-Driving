# 1. Use lightweight Python base image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install dependencies
# Copied separately to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application code
COPY . .

# 6. Make the entrypoint script executable
RUN chmod +x start.sh

# 7. Expose the default Streamlit port
EXPOSE 8501

# 8. Start the orchestration script (runs API and Frontend together)
CMD ["./start.sh"]