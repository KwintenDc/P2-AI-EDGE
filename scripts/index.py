from openvino.runtime import Core
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor

# Load the processor
processor = AutoImageProcessor.from_pretrained("./local_segformer_model")

# Load the image
image_path = "scripts/images/sidewalk3.jpg"
image = Image.open(image_path)

# Preprocess the image
inputs = processor(images=image, return_tensors="np")  # Use NumPy arrays for OpenVINO runtime

# Load OpenVINO Core
core = Core()
print("Available devices:", core.available_devices)  
print("OpenVINO CPU version:", core.get_versions("CPU"))

# Load the ONNX model into OpenVINO
model_path = "segformer_model.onnx"
compiled_model = core.compile_model(model_path, device_name='CPU')

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
segmentation_map_resized = np.array(Image.fromarray(segmentation_map).resize(image.size, Image.NEAREST))

# Convert the PIL image to a NumPy array
image_np = np.array(image)

# Option 2: Using matplotlib's jet colormap and converting it to OpenCV

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

# Wait until a key is pressed to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
