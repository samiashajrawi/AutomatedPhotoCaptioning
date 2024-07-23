from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image
# Initialize the processor and model from Hugging Face
processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base")
# Load an image
image = Image.open("icev4.jpg")
# Prepare the image
inputs = processor(image, return_tensors="pt")
# Generate captions
outputs = model.generate(**inputs)
caption = processor.decode(outputs[0], skip_special_tokens=True)

print("Generated Caption:", caption)
