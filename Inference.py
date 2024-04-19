import torch
from torchvision import transforms, models
from PIL import Image
from LoadTrainCNN import SimpleCNN

# Transformransform igen
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

test_image = Image.open('BMULTIPLA.png').convert('RGB') 
#test_image = Image.open('BLACKBIRD.png').convert('RGB')  

test_image = transform(test_image)  # Apply the same transform used during training
test_image = test_image.unsqueeze(0) # Add batch dimension to the test image as the model expects batches

model = SimpleCNN()

#model = models.vgg16()
model.load_state_dict(torch.load('model.pth')) # Load de trænede weights og biases igen så man ikke skal køre netværket igen
model.eval() # Sætter modellen til evaluation mode

# Lav inference
with torch.no_grad():
    output = model(test_image)

predicted_label = torch.argmax(output).item() # Hent predicted class label

print(output)
print("Predicted Label:", predicted_label)

#lol
