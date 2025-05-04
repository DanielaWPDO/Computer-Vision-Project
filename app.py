import torch
from torchvision import models
import torch.nn as nn
import config
import gradio as gr
from torchvision import transforms


# Preprocess
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to match input size expected by the model
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # Normalize pixel values
                         std=[0.5, 0.5, 0.5])
])

# Load cartoonizer
face2paint = torch.hub.load(config.animegan, "face2paint", size=512)
cartoon_model = torch.hub.load(config.animegan, "generator", pretrained="paprika")

# Load and fix the state_dict
raw_state_dict = torch.load("cnn_resnet18_model.pth", map_location="cpu")

# Handle wrapped keys with "model."
if "model" in raw_state_dict:
    raw_state_dict = raw_state_dict["model"]

# Clean keys: remove "model." prefix if present
from collections import OrderedDict
cleaned_state_dict = OrderedDict()
for k, v in raw_state_dict.items():
    new_key = k.replace("model.", "") if k.startswith("model.") else k
    cleaned_state_dict[new_key] = v

# Define model architecture
classifier = models.resnet18(pretrained=False)
classifier.fc = nn.Linear(classifier.fc.in_features, len(config.CLASS_NAMES))

# Load cleaned weights
classifier.load_state_dict(cleaned_state_dict)
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
    input_tensor = transform(image_for_model).unsqueeze(0)

    # Step 3: Prediction
    with torch.no_grad():
        output = classifier(input_tensor)
        predicted_class = class_names[output.argmax(dim=1).item()]

    # Step 4: Return predicted class and image
    return (predicted_class, cartoon_img if cartoon_img else image)

# Gradio interface

with gr.Blocks(theme=gr.themes.Base()) as interface:
    gr.Markdown(
        """
        # 🎨 Cartoon vs Real Classifier
        Upload an image and optionally apply a cartoon effect.  
        The model will then predict whether the image is **real** or **cartoon**!
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            image_input = gr.Image(type="pil", label="📷 Upload an Image")
            cartoon_checkbox = gr.Checkbox(label="🎭 Apply Cartoonizer")

            submit_btn = gr.Button("🔍 Classify Image")

        with gr.Column(scale=1):
            label_output = gr.Label(label="🧠 Predicted Class")
            image_output = gr.Image(type="pil", label="🖼️ Processed Image")

    submit_btn.click(fn=predict, inputs=[image_input, cartoon_checkbox], outputs=[label_output, image_output])
# Lauching
interface.launch()