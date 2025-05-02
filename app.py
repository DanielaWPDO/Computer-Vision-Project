import torch
from PIL import Image
import config
import gradio as gr
from torchvision import transforms
from src.Preprocessing import transforms as preprocess

# Load cartoonizer
face2paint = torch.hub.load(config.animegan, "face2paint", size=512)
cartoon_model = torch.hub.load(config.animegan, "generator", pretrained="paprika")

# Load your classification model
classifier = torch.load("model.pt", map_location="cpu")
classifier.eval()


class_names = config.CLASS_NAMES

def predict(image, apply_cartoon):
    image = image.convert("RGB")
    cartoon_img = None

    # Step 1: Optional cartoonization
    if apply_cartoon:
        cartoon_img = face2paint(cartoon_model, image)
        image_for_model = cartoon_img
    else:
        image_for_model = image

    # Step 2: Preprocess for classification
    input_tensor = preprocess(image_for_model).unsqueeze(0)

    # Step 3: Prediction
    with torch.no_grad():
        output = classifier(input_tensor)
        predicted_class = class_names[output.argmax(dim=1).item()]

    # Step 4: Return predicted class and cartoon image (if applicable)
    return (predicted_class, cartoon_img if cartoon_img else image)

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Checkbox(label="Apply Cartoonizer")
    ],
    outputs=[
        gr.Label(label="Predicted Class"),
        gr.Image(type="pil", label="Processed Image")
    ],
    title="Cartoon vs Real Classifier",
    description="Upload an image and optionally cartoonize it. The model will predict if it's real or cartoon, and show the final image used for classification."
)

interface.launch()
