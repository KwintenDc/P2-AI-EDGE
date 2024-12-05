# Semantic Segmentation with Segformer (Hugging Face, ONNX, and OpenVINO)
This repository provides a complete pipeline for semantic segmentation using the Segformer model. It supports efficient inference with ONNX and OpenVINO, allowing for real-time video segmentation and deployment flexibility.

## Features
- Uses Segformer for high-performance semantic segmentation.
- Converts PyTorch models to ONNX format for cross-platform compatibility.
- Uses Intel's OpenVINO Runtime for optimized performance.
- Performs segmentation on real-time camera feeds.
- Overlays segmentation maps on original video frames for better insights.

## Requirements
Ensure you have the following packages installed: 
- `transformers`
- `torch`
- `onnxruntime`
- `openvino`
- `Pillow`
- `matplotlib`
- `numpy`
- `opencv-python`

You can install these by installing the requirements.txt, `pip install -r requirements.txt`.

### Docker Support
This repository includes a Docker setup for easy deployment and portability.

## Usage
1. Clone the repository:
    ```
    git clone <repository-url>
    cd <repository-folder>
    ```
2. Run the script: For local execution:
    ```
    python save_model.py
    python save_to_onnx.py
    python index.py
    ```
3. Dockerized Execution:
    - Build the Docker container: 
        ```
        docker build -t segformer-segmentation .
        ```

    - Run the Docker container:
        Make sure the docker is allowed connection to the X server to display the video: 
        ```
        xhost +local:Docker
        ```
        Afterwards run it:
        ```
        docker run -it \
            --rm \
            --env DISPLAY=$DISPLAY \
            --volume /tmp/.X11-unix:/tmp/.X11-unix \
            --device /dev/video0:/dev/video0 \  # For webcam access if needed
            segformer-segmentation
        ```

        