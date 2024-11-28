from openvino.runtime import Core
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor

# Load the processor (ONNX runtime doesn't require model loading from Hugging Face, only the processor)
processor = AutoImageProcessor.from_pretrained("./local_segformer_model")

# Load the image
image_path = "images/sidewalk3.jpg"
image = Image.open(image_path)

# Preprocess the image
inputs = processor(images=image, return_tensors="np")  # Use NumPy arrays for OpenVINO runtime

# Load OpenVINO Core
core = Core()
print("Available devices:", core.available_devices)  # This should list devices, including "CPU" and possibly "GPU"
print("OpenVINO CPU version:", core.get_versions("CPU"))  # Get the version information for the GPU plugin

# Load the ONNX model into OpenVINO
model_path = "segformer_model.onnx"
compiled_model = core.compile_model(model_path, device_name='CPU')  # Use "CPU" if GPU is not available

# Prepare the input dictionary
input_name = compiled_model.input(0).get_names()  # Get the input name
input_data = inputs['pixel_values']  # The preprocessed input image tensor

# Perform inference
output_data = compiled_model([input_data])

# Get the logits (predictions)
logits = output_data[0]  # Shape: (batch_size, num_labels, height/4, width/4)

# Convert logits into a segmentation map by finding the highest-scoring class for each pixel.
segmentation_map = np.argmax(logits, axis=1).squeeze()  # Shape: [height, width]

# Convert segmentation_map to uint8 for PIL compatibility
segmentation_map = segmentation_map.astype(np.uint8)

# Resize segmentation map to match the original image size
segmentation_map_resized = np.array(Image.fromarray(segmentation_map).resize(image.size, Image.NEAREST))

# Step 5: Visualize the results
plt.figure(figsize=(10, 5))

# Segmentation map
plt.subplot(1, 2, 1)
plt.imshow(segmentation_map_resized, cmap="jet")  # Using a colormap for better visualization
plt.title("Segmentation Map")
plt.axis("off")

# Overlayed segmentation on original image
plt.subplot(1, 2, 2)
plt.imshow(image)
plt.imshow(segmentation_map_resized, cmap="jet", alpha=0.5)  # Overlay with transparency
plt.title("Overlayed Segmentation")
plt.axis("off")

plt.show()