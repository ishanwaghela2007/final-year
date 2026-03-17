FROM python:3.10-slim

# Install system dependencies required for OpenCV and SQLite
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install fastapi uvicorn websockets

# Copy the rest of the ML application
COPY . .

# Expose the frontend-facing WebSocket API port
EXPOSE 8000

# Run the FastAPI server
CMD ["python", "src/ml_api.py"]
