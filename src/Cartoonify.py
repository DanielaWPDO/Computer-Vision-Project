import torch
from PIL import Image
from pathlib import Path
import config

# Load the pretrained model and the face2paint utility
generator = torch.hub.load(config.animegan, "generator", pretrained="paprika")
face2paint = torch.hub.load(config.animegan, "face2paint", size=512)

# Define input and output directories
input_dir = Path(config.REAL_DIR)
output_dir = Path(config.CARTOON_DIR)
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
