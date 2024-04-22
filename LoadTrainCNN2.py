import torch
import torch.nn as nn
import torch.optim as optim
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