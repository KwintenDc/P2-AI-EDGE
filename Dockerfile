# Use a base image that supports Python and OpenCV
FROM python:3.11-slim

# Install dependencies for OpenCV, OpenVINO, and X11 forwarding
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk-3-dev \
    libglib2.0-0 \
    libfontconfig1 \
    libx11-6 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first for better Docker cache usage
COPY requirements.txt /app/

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project to the container
COPY . /app

# Run the setup scripts (once during the build process)
RUN python scripts/save_model.py && python scripts/save_to_onnx.py

# Set the entrypoint to run the continuous script
CMD ["python", "scripts/index.py"]