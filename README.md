# 🧠 Cartoon vs Real: Image Classification using AnimeGAN

This project aims to classify images into two categories: `cartoon` or `real`. To achieve this, we use a combination of real-world image datasets and synthetic cartoon images generated from real photos using a deep learning model.

---


!!!!! Revérifie les liens github car tu vas changer le nom du projet!!!!!!!!

## 📂 Project Overview

- **Data acquisition**: Real-world images are fetched via the [Pexels API](https://www.pexels.com/api/).
- **Cartoonification**: The script `cartoonify.py` transforms real photos into anime-style images using [AnimeGANv2](https://github.com/TachibanaYoshino/AnimeGANv2).
- **PyTorch integration**: We use a PyTorch-compatible version of AnimeGAN by [bryanleed](https://github.com/bryandlee/animegan2-pytorch) to facilitate GPU acceleration and batch processing.
- **Classification**: The final step involves training a classifier to distinguish between cartoon and real images.

---

## ⚙️ Setup

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/cartoon-vs-real.git
cd cartoon-vs-real
```
### 2. Install dependencies
Make sure you’re in a virtual environment, then:

```bash
pip install -r requirements.txt
```

### 🔐 API Key
To fetch real-world images using the Pexels API, you need to set your API key.
Create a `.env` file in the root of the project:

```bash
PEXELS_API_KEY=your_api_key_here
```
This key will be used by the dataset loader to retrieve real images.

⚠️ Do not share your key. This file is ignored by Git.

## 🖼️ Cartoonify Images
To generate cartoon images from real photos:

```bash

python cartoonify.py --input dataset\real --output dataset\cartoon 
```
This uses AnimeGANv2 with the PyTorch wrapper by [bryanleed](https://github.com/bryandlee/animegan2-pytorch). You may need to download pretrained model. This project uses `paprika` cartoon style but you can choose another among this list of styles (`celeba_distill`,`celeface_paint_512_v1ba_distill`,`face_paint_512_v2`)

## 📁 Project Structure
```
.
├── cartoonify.py         # Cartoonification using AnimeGAN
├── dataset_loader.py     # Image fetching using Pexels API
├── classifier.py         # (Upcoming) Image classification logic
├── .env.example          # Template for API key
├── requirements.txt      # Python dependencies
└── README.md
```