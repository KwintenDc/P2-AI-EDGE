from openvino.runtime import Core
import numpy as np
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor
from PIL import Image

# Load the processor
processor = AutoImageProcessor.from_pretrained("./local_segformer_model")

# Load OpenVINO Core
core = Core()
print("Available devices:", core.available_devices)  
print("OpenVINO CPU version:", core.get_versions("CPU"))

# Load the ONNX model into OpenVINO
model_path = "segformer_model.onnx"
compiled_model = core.compile_model(model_path, device_name='CPU')

video_path = "scripts/images/sidewalk.mp4"

# Start capturing from the camera (camera index 0 by default)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Resize the frame to a smaller size for better performance (e.g., 320x240)
    resized_frame = cv2.resize(frame, (160, 120))

    # Convert the resized frame to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    inputs = processor(images=image, return_tensors="np")

    # Prepare the input
    input_name = compiled_model.input(0).get_names()
    input_data = inputs['pixel_values']

    # Convert logits to a segmentation map
    segmentation_map = np.argmax(compiled_model([inputs['pixel_values']])[0], axis=1).squeeze()
    segmentation_map_resized = cv2.resize(segmentation_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

    colored_segmentation_map = np.zeros_like(frame)

    # Define the class ID for sidewalk
    sidewalk_class_id = 2  

    colored_segmentation_map[segmentation_map_resized == sidewalk_class_id] = [0, 0, 255]
    overlayed_image = cv2.addWeighted(frame, 0.7, colored_segmentation_map, 0.3, 0)

    # Visualize the segmentation map and overlay with OpenCV
    cv2.imshow("Colored Segmentation Map", colored_segmentation_map)
    cv2.imshow("Overlayed Segmentation", overlayed_image)

    # Wait for 1 ms and check if the user pressed the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()