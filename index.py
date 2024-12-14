from openvino.runtime import Core
import numpy as np
import cv2
from transformers import AutoImageProcessor
import time  # For calculating FPS

# Load the processor
processor = AutoImageProcessor.from_pretrained("./local_segformer_model")

# Load OpenVINO Core
core = Core()
print("Available devices:", core.available_devices)  
print("OpenVINO CPU version:", core.get_versions("CPU"))

# Load the ONNX model into OpenVINO
model_path = "segformer_model.onnx"
compiled_model = core.compile_model(model_path, device_name='CPU')

video_path = "images/sidewalk.mp4"

# Start capturing from the video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to access the video.")
    exit()

# Initialize variables for FPS calculation
prev_time = time.time()

while True:
    # Capture a frame from the video
    ret, frame = cap.read()
    if not ret:
        print("End of video or failed to capture frame.")
        break

    # Resize the frame to a smaller size for better performance (e.g., 160x120)
    resized_frame = cv2.resize(frame, (160, 120))

    # Convert the resized frame to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image, return_tensors="np")

    # Convert logits to a segmentation map
    segmentation_map = np.argmax(compiled_model([inputs['pixel_values']])[0], axis=1).squeeze()
    segmentation_map_resized = cv2.resize(segmentation_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    # Create a coloured segmentation map
    colored_segmentation_map = np.zeros_like(frame)
    sidewalk_class_id = 2  # Define the class ID for sidewalk
    colored_segmentation_map[segmentation_map_resized == sidewalk_class_id] = [0, 0, 255]

    # Create an overlayed image
    overlayed_image = cv2.addWeighted(frame, 0.7, colored_segmentation_map, 0.3, 0)

    # Scale factor (e.g., 50% of the original size)
    scale_percent = 50
    new_width = int(overlayed_image.shape[1] * scale_percent / 100)
    new_height = int(overlayed_image.shape[0] * scale_percent / 100)
    new_dim = (new_width, new_height)

    # Resize the images
    scaled_colored_segmentation_map = cv2.resize(colored_segmentation_map, new_dim, interpolation=cv2.INTER_NEAREST)
    scaled_overlayed_image = cv2.resize(overlayed_image, new_dim, interpolation=cv2.INTER_LINEAR)

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    # Add FPS text to the overlayed image
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(scaled_overlayed_image, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Visualize the scaled overlay with FPS
    cv2.imshow("Scaled Colored Segmentation Map", scaled_colored_segmentation_map)
    cv2.imshow("Scaled Overlayed Segmentation with FPS", scaled_overlayed_image)

    # Wait for 1 ms and check if the user pressed the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()