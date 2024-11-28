# This script fetches a pretrained semantic segmentation model (Segformer) and its processor from Hugging Face, 
# then saves them locally so they can be reused without downloading again.

from transformers import AutoImageProcessor, SegformerForSemanticSegmentation

# Define the model and processor from huggingface
model_name = "nickmuchi/segformer-b4-finetuned-segments-sidewalk"

# Function downloads the image processor associated with the specified model. 
# The processor includes preprocessing steps like resizing and normalization.
processor = AutoImageProcessor.from_pretrained(model_name)

# Saves the downloaded processor locally in the ./local_segformer_model folder.
processor.save_pretrained("./local_segformer_model")

# Downloads the Segformer model (pretrained on semantic segmentation tasks) from the Hugging Face Hub.
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

# Saves the downloaded model locally in the same folder.
model.save_pretrained("./local_segformer_model")