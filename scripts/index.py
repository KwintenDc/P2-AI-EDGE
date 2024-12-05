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

# Start capturing from the camera (camera index 0 by default)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Convert the frame to RGB (OpenCV uses BGR by default)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image)  # Convert to PIL image for processing

    # Preprocess the image
    inputs = processor(images=image_pil, return_tensors="np")  # Use NumPy arrays for OpenVINO runtime

    # Prepare the input
    input_name = compiled_model.input(0).get_names()
    input_data = inputs['pixel_values']

    # Perform inference
    output_data = compiled_model([input_data])

    # Get the logits
    logits = output_data[0]  # Shape: (batch_size, num_labels, height/4, width/4)

    # Convert logits to a segmentation map
    segmentation_map = np.argmax(logits, axis=1).squeeze()

    # Convert segmentation_map to uint8 for OpenCV compatibility
    segmentation_map = segmentation_map.astype(np.uint8)

    # Resize the segmentation map to match the original image size
    segmentation_map_resized = np.array(Image.fromarray(segmentation_map).resize(image_pil.size, Image.NEAREST))

    # Convert the PIL image to a NumPy array
    image_np = np.array(image_pil)

    # Create a colormap using matplotlib and apply it to the segmentation map
    cmap = plt.get_cmap("jet")  # Use matplotlib's jet colormap
    segmentation_map_colored = cmap(segmentation_map_resized)  # Apply colormap

    # Convert to OpenCV-compatible format (Remove alpha channel and convert to uint8)
    segmentation_map_colored = (segmentation_map_colored[:, :, :3] * 255).astype(np.uint8)

    # Overlay the segmentation map on the original image using OpenCV
    overlayed_image = cv2.addWeighted(image_np, 0.5, segmentation_map_colored, 0.5, 0)

    # Visualize the segmentation map and overlay with OpenCV
    cv2.imshow("Segmentation Map", segmentation_map_colored)
    cv2.imshow("Overlayed Segmentation", overlayed_image)

    # Wait for 1 ms and check if the user pressed the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()