# Use a base image that supports Python and OpenCV
FROM python:3.11-slim

# Install dependencies for OpenCV, OpenVINO, and X11 forwarding
RUN apt-get update && \
    apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libfontconfig1 \
    libx11-6 \
    libopencv-dev \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project to the container
COPY . /app

# Install Python dependencies from requirements.txt
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run the setup scripts (once during the build process)
RUN python scripts/save_model.py && python scripts/save_to_onnx.py

# Set the entrypoint to run the continuous script
CMD ["xvfb-run", "python", "scripts/index.py"]