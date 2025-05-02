from src.dataset_loader import ImageDataset
import config
from torch.utils.data import DataLoader
from torchvision import transforms

# Define image transformations (resize + tensor conversion + normalization)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load the dataset
dataset = ImageDataset(csv_file=config.CSV_PATH , transform=None, encode= True)

"""# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for i in range(len(dataset)):
    image, label = dataset[i]
    print(image)  # Tensor representing the image
    print(label)  # Label as integer (e.g., 0 or 1)
    break  # Remove break to see all"""


# Affichage
img, label = dataset.get_raw_image(154)
img.show()
print(label)
