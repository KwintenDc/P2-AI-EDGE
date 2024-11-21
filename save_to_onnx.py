# This script converts the locally saved PyTorch Segformer model into an ONNX model format, ensuring it is portable and compatible with ONNX Runtime.

import torch
from PIL import Image
from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Load the model and processor from the local directory
processor = AutoImageProcessor.from_pretrained("./local_segformer_model")
model = SegformerForSemanticSegmentation.from_pretrained("./local_segformer_model")

# Ensures the model operates in evaluation mode, disabling layers like dropout.
model.eval()

# Example image input (use the same image preprocessing as the original)
image_path = "images/sidewalk3.jpg"
image = Image.open(image_path)
# Preprocesses the image into a format the model expects (tensor of shape [batch_size, channels, height, width]).
inputs = processor(images=image, return_tensors="pt")

# Convert the model to ONNX format
dummy_input = inputs.pixel_values  # This is the input tensor for the model
onnx_output_path = "segformer_model.onnx"

# Converts the PyTorch model into the ONNX format for use in other frameworks like ONNX Runtime.
torch.onnx.export(
    model, 
    dummy_input, 
    onnx_output_path, 
    input_names=["pixel_values"], 
    output_names=["logits"],
    opset_version=12,  # Ensure the opset version is compatible with your ONNX runtime
    dynamic_axes={"pixel_values": {0: "batch_size", 2: "height", 3: "width"}, "logits": {0: "batch_size", 2: "height", 3: "width"}},
    # dynamic_axes allows for variable batch size and image dimensions
)
print(f"Model has been exported to {onnx_output_path}")