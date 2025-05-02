
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None, encode=False):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.encode = encode

        # Encode labels to integers if needed
        if self.encode:
            self.label_to_index = {label: idx for idx, label in enumerate(sorted(self.data['label'].unique()))}
            self.index_to_label = {v: k for k, v in self.label_to_index.items()}
        else:
            self.label_to_index = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        label_str = self.data.iloc[idx]['label']

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)

        # Encode label if needed
        if self.encode:
            label = self.label_to_index[label_str]
        else:
            label = label_str

        return image, label

    def get_raw_image(self, idx):
        img_path = self.data.iloc[idx]['filepath']
        label_str = self.data.iloc[idx]['label']
        image = Image.open(img_path).convert("RGB")
        return image, label_str
