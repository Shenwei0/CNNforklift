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
        image = image.resize(self.targetSize, Image.BILINEAR)

        if self.transform:
            image = self.transform(image)
        
        if x < len(self.imagePaths)//3:
            label = 0
        elif x < len(self.imagePaths) and x > len(self.imagePaths)//3:
            label = 1
        else: 
            label = 2

        print(imgName, "and label: ", label)        

        return image, label



transform = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

Dataset = ImageDataset(rootDir='DATA_jet_car_ship', transform=transform)

trainLoader = DataLoader(Dataset, batch_size=1, shuffle=True)


















# for x, (images, labels) in enumerate(trainLoader):
#     for i in range(len(images)):
#         imgName = Dataset.imagePaths[x * trainLoader.batch_size + 1]
#         label = labels[i].item()
#         print("Image name: ", imgName, " and label: ", label)



# for batch_idx, (images, labels) in enumerate(trainLoader):
#     for i in range(len(images)):
#         imgName = Dataset.imagePaths[batch_idx * trainLoader.batch_size + i]
#         label = labels[i].item()
#         print("Image Name:", imgName, "| Label:", label)





        