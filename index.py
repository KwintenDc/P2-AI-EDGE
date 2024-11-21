# This script uses the ONNX model to perform inference on an input image, processes the output to generate a segmentation map,
# and visualizes the results with an overlay on the original image.

import onnxruntime as ort
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
inputs = processor(images=image, return_tensors="np")  # Use NumPy arrays for ONNX runtime

print(ort.get_available_providers())
# Specify that ONNX Runtime should use the TensorRT execution provider
providers = ['DNNLExecutionProvider', 'CPUExecutionProvider']

# Loads the ONNX model and specifies the inference backends (e.g., CPU or DNNL).
onnx_session = ort.InferenceSession("segformer_model.onnx", providers=providers)

# Prepare the input dictionary
input_name = onnx_session.get_inputs()[0].name
input_data = inputs['pixel_values']  # The preprocessed input image tensor

# Run inference
outputs = onnx_session.run(None, {input_name: input_data})

# Get the logits (predictions)
logits = outputs[0]  # shape: (batch_size, num_labels, height/4, width/4)

# Converts logits into a segmentation map by finding the highest-scoring class for each pixel.
segmentation_map =np.argmax(logits, axis=1).squeeze()  # Shape: [height, width]

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