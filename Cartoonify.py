import torch
from PIL import Image
from pathlib import Path
import os

# Load the pretrained model and the face2paint utility
generator = torch.hub.load("bryandlee/animegan2-pytorch:main", "generator", pretrained="paprika")
face2paint = torch.hub.load("bryandlee/animegan2-pytorch:main", "face2paint", size=512)

# Define input and output directories
input_dir = Path("dataset/real")
output_dir = Path("dataset/cartoon")
output_dir.mkdir(parents=True, exist_ok=True)

# Supported image extensions
image_extensions = (".jpg", ".jpeg", ".png")

# Process each image in the input directory
for image_path in input_dir.iterdir():
    if image_path.suffix.lower() in image_extensions:
        try:
            # Open and preprocess the image
            img = Image.open(image_path).convert("RGB")
            
            # Apply the cartoon effect
            cartoon_img = face2paint(generator, img)
            
            # Define output path
            output_path = output_dir / image_path.name
            
            # Save the cartoonized image
            cartoon_img.save(output_path)
            print(f"Saved cartoon image to {output_path}")
        
        except Exception as e:
            print(f"Failed to process {image_path.name}: {e}")
