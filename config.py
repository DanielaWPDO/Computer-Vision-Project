# config.py

# Paths
CSV_PATH = "dataset/labels.csv"
REAL_DIR = "dataset/real"
CARTOON_DIR = "dataset/cartoon"

# Url
animegan= "bryandlee/animegan2-pytorch:main"

# Training settings
"""BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001"""

# Image settings
IMAGE_SIZE = (224, 224)

# Classes
CLASS_NAMES = ["real", "cartoon"]
