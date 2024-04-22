import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import time

device = (
    "Cuda" if torch.cuda.is_available() 
    else "MPS" if torch.backends.mps.is_available() 
    else "CPU"
)
print("Using ", device, "XD")

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, rootDir, transform=None, targetSize=(256,256)):
        self.rootDir = rootDir
        self.transform = transform
        self.targetSize = targetSize
        self.imagePaths = os.listdir(rootDir)

    def __len__(self):
        return len(self.imagePaths)
    
    def __getitem__(self, x):
        imgName = os.path.join(self.rootDir, self.imagePaths[x])
        image = Image.open(imgName).convert('RGB')
        image = Image.resize(self.targetSize, Image.BILINEAR)

        if self.transform:
            image = self.transform(image)
        
        '''
        Indsæt labelling herind
        '''






        