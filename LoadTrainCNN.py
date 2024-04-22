import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from PIL import Image
import os
import time


device = (
    "Cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} XD")

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, target_size=(256,256)):
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.image_paths = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, x):
        img_name = os.path.join(self.root_dir, self.image_paths[x])
        image = Image.open(img_name).convert('RGB')  # Konvertere til RGB to ensure consistency

        # Resize image to the target size
        image = image.resize(self.target_size, Image.BILINEAR)

        if self.transform:
            image = self.transform(image)
        
        # Herunder labeles datasættet

        label = 0 if x > len(self.image_paths)//2 else 1

        # label = 1
        # if x > len(self.image_paths)//2:
        #     label = 0
        # else: 
        #     1

        print(len(self.image_paths)//2)
        print(len(self.image_paths))
        print(img_name, "and ", label)
        #print(img_name)
        
        return image, label

# Define transform to apply to the images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create dataset instance
dataset = ImageDataset(root_dir='DATA_jet_car_ship', transform=transform)

# Create DataLoader
batch_size = 1  #Increase den her når datasættet bliver større
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define your neural network model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 128 * 128, 2)  # Justere input size afhængig af the resized image size

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(-1, 128 * 128 * 16)  # Adjust the size here accordingly
        x = self.fc(x)
        return x

model = SimpleCNN() # Initialize modellen

# Definer loss function og optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
def training():
    num_epochs = 40
    start = time.time()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")    
    print("Time used :", time.time() - start, "seconds")

    FILE = 'model.pth'
    torch.save(model.state_dict(), FILE)
    print(model.state_dict)

if __name__ == "__main__":
    training()
    
