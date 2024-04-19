import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import os

class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_paths[idx])
        image = Image.open(img_name)
        
        if self.transform:
            image = self.transform(image)
        
        # Assuming the dataset structure is such that first 10 images are from one pose, and next 10 from another
        label = 0 if idx < len(self.image_paths)//2 else 1
        
        return image, label

# Define transform to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create dataset instance
dataset = ImageDataset(root_dir='path_to_your_dataset_folder', transform=transform)

# Example usage:
# Accessing an image and its label
image, label = dataset[0]
print("Image shape:", image.shape)
print("Label:", label)
