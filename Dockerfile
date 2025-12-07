# Minimal image for the face detection/anonymization pipeline
FROM python:3.11-slim

# Install build deps for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project files
COPY . /app

# Create directories for mounts
RUN mkdir -p /app/photo_input /app/photo_detection_output /app/photo_detection_error /app/photo_anonymization_output /app/models

ENV PHOTO_INPUT=/app/photo_input
ENV PHOTO_DETECTION_OUTPUT=/app/photo_detection_output
ENV PHOTO_DETECTION_ERROR=/app/photo_detection_error
ENV PHOTO_ANONYMIZATION_OUTPUT=/app/photo_anonymization_output
ENV MODE=watch
ENV POLL_INTERVAL=5

# Default command: run the watchdog watcher that executes the pipeline
CMD ["python", "-u", "src/watcher.py"]
