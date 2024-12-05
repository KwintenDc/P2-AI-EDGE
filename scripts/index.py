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

    # Initialize an empty image for coloring the segmentation map
    colored_segmentation_map = np.zeros_like(image_np)

    # Define the class ID for sidewalk (you need to know the correct ID from your model)
    sidewalk_class_id = 2  # Replace this with the correct class ID for sidewalk

    # Apply red color (255, 0, 0) to the sidewalk pixels
    colored_segmentation_map[segmentation_map_resized == sidewalk_class_id] = [0, 0, 255]  # Red color

    # Optionally: Adjust color brightness for other parts of the map (for example, darken the background)
    # You can apply other colors for different classes if needed

    # Create an overlay with the original image and the colored segmentation map
    overlayed_image = cv2.addWeighted(image_np, 0.7, colored_segmentation_map, 0.3, 0)

    # Visualize the segmentation map and overlay with OpenCV
    cv2.imshow("Colored Segmentation Map", colored_segmentation_map)
    cv2.imshow("Overlayed Segmentation", overlayed_image)

    # Wait for 1 ms and check if the user pressed the 'q' key to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()