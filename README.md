# 🧠 Cartoon vs Real: Image Classification using AnimeGAN

This project aims to classify images into two categories: `cartoon` or `real`. To achieve this, we use a combination of real-world image datasets and synthetic cartoon images generated from real photos using a deep learning model.

---


!!!!! ⚠️ Revérifie les liens github car tu vas changer le nom du projet!!!!!!!! N'oublie pas de modifier la partie Project Structure

## 📂 Project Overview

- **Data acquisition**: Real-world images are fetched via the [Pexels API](https://www.pexels.com/api/).
- **Cartoonification**: The script `cartoonify.py` transforms real photos into anime-style images using [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2).
- **PyTorch integration**: We use a PyTorch-compatible version of AnimeGAN by [bryanleed](https://github.com/bryandlee/animegan2-pytorch) to facilitate GPU acceleration and batch processing.
- **Classification**: The final step involves training a classifier to distinguish between cartoon and real images.

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/DanielaWPDO/Computer-Vision-Project.git
cd cartoon-vs-real
```
### 2. Install dependencies
Make sure you’re in a virtual environment, then:

```bash
pip install -r requirements.txt
```

### 🔐 API Key
To fetch real-world images using the Pexels API, you need to set your API key.
For downloading images execute this code:

```bash
python image_downloader.py --api-key "your_api_key_here"
```
This key will be used by the dataset loader to retrieve real images.

## 🖼️ Cartoonify Images
To generate cartoon images from real photos:

```bash

python cartoonify.py --input dataset\real --output dataset\cartoon 
```
This uses AnimeGANv2 with the PyTorch wrapper by [bryanleed](https://github.com/bryandlee/animegan2-pytorch). You may need to download pretrained model. This project uses `paprika` cartoon style but you can choose another among this list of styles (`celeba_distill`,`celeface_paint_512_v1ba_distill`,`face_paint_512_v2`)

## Training 

You can choose to use the already annoted dataset. To do so, just import `dataset_loader` library like that:

```bash
from src.dataset_loader import ImageDataset

dataset = ImageDataset(csv_file=config.CSV_PATH, transform=None, encode=True)
```
You can also use this library to print dataset values (image, label)

## Deployment

We use [Gradio](https://www.gradio.app/) to create an interactive web app for classifying images as either real or cartoon. The app also allows users to apply a cartoon effect to images before classification.

To run the app, just execute the following command:
```bash
python app.py
```

After running the script, you will see an Url as output. Click the URL to open the app in your browser. Upload an image, optionally apply a cartoon effect, and click "Classify Image" to see the predicted result. And that's it!!! Simple, Right??

## 📁 Project Structure
```
.
├── dataset/ # Dataset directory
│ ├── cartoon/ # Cartoon images
│ ├── real/ # Real images
│ ├── download_image.py # Image downloading script
│ └── labels.csv # Image labels
│
├── src/ # Source code
│ ├── Cartoonify.py # Cartoonification using AnimeGAN
│ ├── dataset_loader.py # Dataset loading utilities
│ ├── Labeller.py # Image labeling utilities
│ ├── Models_Training.ipynb # Model training script
│ ├── Models_Evaluation.ipynb # Model Evaluation script
│
├── app.py # Main application script
├── cm_resnet18_model.pth # Pretrained ResNet18 model
├── config.py # Configuration file
├── vit_model.pth # Pretrained Vision Transformer model
├── requirements.txt # Python dependencies
└── README.md


```