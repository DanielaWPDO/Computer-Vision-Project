from src.dataset_loader import ImageDataset
import config
from torch.utils.data import DataLoader
from torchvision import transforms

transform = transforms.Compose([
    # Resize the image to 224x224 pixels (required input size for many CNN models like ResNet)
    transforms.Resize((224, 224)),
    # Randomly flip the image horizontally with a 50% chance (data augmentation)
    transforms.RandomHorizontalFlip(),
    # Randomly change brightness and contrast to simulate lighting variation (data augmentation)
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # Convert the PIL Image to a PyTorch tensor and scale pixel values to [0, 1]
    transforms.ToTensor(),
    # Normalize the image using mean and std per channel to center the data
    # This helps the model train faster and more stably
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])


"""
Load the dataset using a custom PyTorch Dataset class.

Args:
    csv_file (str): Path to the CSV file containing image filepaths and labels.
    transform (callable, optional): Optional torchvision transforms to apply on the images (e.g., resizing, normalization).
    encode (bool): If True, string labels will be encoded as integers (e.g., 'real' → 0, 'cartoon' → 1).

Returns:
    ImageDataset: A PyTorch Dataset object where each item is a tuple (image, label), 
                  with image optionally transformed and label either as a string or an encoded integer.
"""
dataset = ImageDataset(csv_file=config.CSV_PATH, transform=None, encode=True)


"""# Create a DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for i in range(len(dataset)):
    image, label = dataset[i]
    print(image)  # Tensor representing the image
    print(label)  # Label as integer (e.g., 0 or 1)
    break  # Remove break to see all"""


# Print dataset values for demo
index=154
img, label = dataset.get_raw_image(index)
img.show()
print(label)
